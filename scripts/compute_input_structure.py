"""Similarity structure of the AOJ input itself, and how the models relate to it.

All representations are compared with the same set kernel as compute_constituent_mmd_leading_n.py:
the MMD / RBF kernel between jets over the N leading-pT constituents of each jet, equal weights
(beta = 0), bandwidth = median pairwise constituent distance of that representation. CKNNA is then
computed between two such jet-by-jet kernels (constituent_kernels.cknna_sweep; neighbours by kernel
distance). The "input" representation is each constituent's preprocessed feature vector, i.e.
exactly what the backbone receives (truncated to the same 99 constituents).

1. random maps of the input: each constituent's input features go through a random per-constituent
   MLP (a random linear map to 128 dims, then `depth` random GELU layers, He-initialised, no
   particle mixing). Two independently seeded maps of the same depth are compared with each other
   ("map pair": the data-only analogue of random NTP vs random MPM) and each map with the input
   ("map vs input": how much of the input neighbourhoods a random map keeps).
2. what drives the neighbourhoods: CKNNA of the input, and of every model, against four scalar jet
   observables (multiplicity, jet pT, jet mass, leading-constituent pT fraction, all of the full
   jet). The observable kernel is an RBF of the standardised observable, so a jet's neighbours
   are the jets with the closest value. A tiny jitter breaks ties (many jets share the same
   multiplicity); because ties are broken at random, even a perfect encoder of the observable
   cannot reach 1, so each observable's "ceiling" (the observable vs itself with a different
   jitter) is reported too.
3. models vs input: random-init (3 seeds, same weights as the other scripts), end of pre-training
   (the snapshot fine-tuning started from) and end of fine-tuning (epoch 15), NTP and MPM, each
   against the input; plus the random NTP vs random MPM anchor.

Run from the repository root (writes analysis/cknna/<MAX_BATCHES>batches_input_structure/ and
analysis/embedding_alignment/input_structure/<MAX_BATCHES>batches/); --plot-only redoes the
tables and figures from the saved json.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import compute_constituent_mmd_kernel as mmd  # sets up sys.path / project root on import
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import compute_cknna_mpm_vs_ntp_on_aoj as base
import gabbro.metrics.constituent_kernels as ck
from plot_cknna import C_RND, save

MAX_BATCHES = 20
N_CUTS = (2, 5, 10, 20, None)               # N leading-pT constituents; None = the full jet
BETA, SIGMA_FACTOR = 0.0, 1.0
MAP_DEPTHS = (0, 1, 2, 4)                    # random GELU layers after the random linear map
MAP_WIDTH = 128                              # = backbone embedding_dim
N_MAP_SEEDS = 3
TOPKS = (5, 10, 25, 50)
N_SUB, N_DRAWS, N_PERM = 1500, 5, 1
OBSERVABLES = ("log_multiplicity", "log_jet_pt", "log_jet_mass", "leading_pt_fraction")
MODELS = ("ntp_random", "mpm_random", "ntp_pretrained", "mpm_pretrained", "ntp_finetuned", "mpm_finetuned")


# ------------------------------------------------------------------ representations
def input_inference(batches, pt_getter, n_jets, max_len):
    """The input features in the format of constituent_kernels.constituent_inference."""
    embs, jids, pts, seen = [], [], [], 0
    for b in batches:
        pf, valid = b["part_features"][:, :max_len], b["part_mask"][:, :max_len].bool()
        take = min(len(pf), n_jets - seen)
        valid = valid[:take]
        jet_idx = torch.arange(seen, seen + take)[:, None].expand_as(valid)
        embs.append(pf[:take][valid].float())
        jids.append(jet_idx[valid])
        pts.append(pt_getter(pf[:take])[valid].float())
        seen += take
        if seen >= n_jets:
            break
    E, jet_idx, pt = torch.cat(embs), torch.cat(jids), torch.cat(pts)
    bad_c = ~torch.isfinite(E).all(1) | ~torch.isfinite(pt) | (pt <= 0)
    E[bad_c], pt[bad_c] = 0.0, 1.0
    return E, jet_idx, pt, torch.unique(jet_idx[bad_c])


def random_map(inference, depth, seed):
    """Per-constituent random MLP: random linear map to MAP_WIDTH, then `depth` random GELU layers."""
    E, jet_idx, pt, bad = inference
    g = torch.Generator().manual_seed(seed)
    h = E.to(base.device) @ (torch.randn(E.shape[1], MAP_WIDTH, generator=g) / E.shape[1] ** 0.5).to(base.device)
    for _ in range(depth):
        W = torch.randn(MAP_WIDTH, MAP_WIDTH, generator=g) * (2 / MAP_WIDTH) ** 0.5
        h = torch.nn.functional.gelu(h @ W.to(base.device))
    return h.cpu(), jet_idx, pt, bad


def set_kernel(inference, n_leading, n_jets):
    """Jet-by-jet MMD kernel over the N leading constituents (all if n_leading is None)."""
    if n_leading is not None:
        inference = ck.leading_constituents(inference, n_leading)
    E, jet_idx, pt, _ = inference
    sigma = SIGMA_FACTOR * ck.median_heuristic_sigma(E)
    return ck.mmd_kernels(E, jet_idx, pt, n_jets, (BETA,), (sigma,), device=base.device)[(BETA, sigma)]


def observables(batches, getters, n_jets, max_len):
    """Full-jet observables per jet (massless constituents for the jet mass)."""
    out = {o: [] for o in OBSERVABLES}
    for b in batches:
        pf, m = b["part_features"][:, :max_len].double(), b["part_mask"][:, :max_len].bool()
        pt, eta, phi = (torch.where(m, getters[f](pf).double(), torch.zeros_like(pf[..., 0]))
                        for f in ("part_pt", "part_etarel", "part_phirel"))
        px, py, pz, e = pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta), pt * torch.cosh(eta)
        mass2 = e.sum(1) ** 2 - px.sum(1) ** 2 - py.sum(1) ** 2 - pz.sum(1) ** 2
        jet_pt = pt.sum(1)
        out["log_multiplicity"].append(m.sum(1).double().log())
        out["log_jet_pt"].append(jet_pt.log())
        out["log_jet_mass"].append(mass2.clamp_min(1e-6).sqrt().log())
        out["leading_pt_fraction"].append(pt[:, 0] / jet_pt)
    return {o: torch.cat(v)[:n_jets] for o, v in out.items()}


def scalar_kernel(x, seed):
    """RBF kernel of a standardised scalar (neighbours = closest value).

    A uniform jitter of at most a quarter of the smallest gap between distinct values breaks exact
    ties (e.g. equal multiplicity) at random without ever reordering distinct values.
    """
    z = (x - x.mean()) / x.std()
    gaps = torch.diff(torch.unique(z))
    g = torch.Generator().manual_seed(seed)
    z = z + 0.25 * gaps.min() * (2 * torch.rand(len(z), generator=g, dtype=torch.float64) - 1)
    return torch.exp(-0.5 * (z[:, None] - z[None, :]) ** 2)


def cknna(K, L, keep):
    Ks, Ls = (M[keep][:, keep].to(base.device) for M in (K, L))
    rows = ck.cknna_sweep(Ks, Ls, TOPKS, N_SUB, N_DRAWS, n_perm=N_PERM)
    del Ks, Ls
    return rows


def load_models(batches, pt_getter, n_jets):
    """Constituent inference (full jets) of the random-init, end-of-pre-training and end-of-fine-tuning models."""
    ck_ntp = mmd.checkpoints(base.NTP_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[0])
    ck_mpm = mmd.checkpoints(base.MPM_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[1])
    ck_ntp_ft = mmd.checkpoints(base.NTP_FINETUNE_RUN, True)
    ck_mpm_ft = mmd.checkpoints(base.MPM_FINETUNE_RUN, True)
    by_label = lambda paths, label: next(p for p in paths if base.epoch_label_from_ckpt_path(p) == label)  # noqa: E731
    makers = {}
    for seed in range(mmd.N_RANDOM_SEEDS):
        makers[("ntp_random", seed)] = lambda s=seed: base.random_init_model(base.first_real_ckpt(ck_ntp), s)
        makers[("mpm_random", seed)] = lambda s=seed: base.random_init_model(base.first_real_ckpt(ck_mpm), 1000 + s)
    for name, paths, label in [("ntp_pretrained", ck_ntp, "ft_src"), ("mpm_pretrained", ck_mpm, "ft_src"),
                               ("ntp_finetuned", ck_ntp_ft, 15), ("mpm_finetuned", ck_mpm_ft, 15)]:
        makers[(name, 0)] = lambda p=paths, lab=label: base.load_model(by_label(p, lab), base.first_real_ckpt(p))
    inferences = {}
    for key, make in makers.items():
        model = make()
        inferences[key] = ck.constituent_inference(batches, model, pt_getter, n_jets, base.device)
        del model
        mmd.free()
        print(f"inference done: {key}")
    return inferences


# ------------------------------------------------------------------ computation
def compute(out_json):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "experiment=example_experiment_backbone_and_head_allfeat_ntp_classifier_head",
            "feature_dict=feature_dict_all_without_cuts"])
    bs = cfg.data.batch_size.train
    n_jets = MAX_BATCHES * bs
    max_len = cfg.data.dataset_kwargs_common.pad_length - 1   # the models prepend a start token
    dataloader = base.get_aoj_dataloader(
        cfg, "/srv/beegfs/scratch/groups/rodem/lumea/datasets/aspenJets/RunG_batch0.h5",
        n_jets_per_file=(MAX_BATCHES + 1) * bs)
    batches = ck.collect_batches(dataloader, n_jets)
    fd = cfg.data.dataset_kwargs_common.feature_dict
    getters = {f: ck.make_pt_getter(fd, pt_key=f) for f in ("part_pt", "part_etarel", "part_phirel")}

    inp = input_inference(batches, getters["part_pt"], n_jets, max_len)
    obs = observables(batches, getters, n_jets, max_len)
    models = load_models(batches, getters["part_pt"], n_jets)

    keep = torch.ones(n_jets, dtype=torch.bool)
    for _, _, _, bad in [inp, *models.values()]:
        keep[bad.long()] = False
    for x in obs.values():
        keep &= torch.isfinite(x)
    print(f"dropping {int((~keep).sum())} of {n_jets} jets (non-finite input, embedding or observable)")
    for o, x in obs.items():
        print(f"{o:>20}: mean {x[keep].mean():.3f}, std {x[keep].std():.3f}")
    keep = keep.nonzero().squeeze(1)
    obs_kernels = {o: scalar_kernel(torch.nan_to_num(x), seed=0) for o, x in obs.items()}

    results = []

    def add(n, a, b, seed, rows):
        label = "all" if n is None else n
        results.extend({"n_leading": label, "rep_a": a, "rep_b": b, "seed": seed, **r} for r in rows)
        k10 = next(r for r in rows if r["topk"] == 10)
        print(f"  N={label!s:>3}  {a:>22} vs {b:<22} seed {seed}: CKNNA(k=10) = {k10['cknna']:.3f}")

    # ceilings: an observable against itself with a different tie-breaking jitter (independent of N)
    for o, x in obs.items():
        add(None, o, o, 0, cknna(obs_kernels[o], scalar_kernel(torch.nan_to_num(x), seed=1), keep))

    for n in N_CUTS:
        t0 = time.time()
        K_in = set_kernel(inp, n, n_jets)
        for o in OBSERVABLES:
            add(n, "input", o, 0, cknna(K_in, obs_kernels[o], keep))
        for depth in MAP_DEPTHS:
            for s in range(N_MAP_SEEDS):
                K_a = set_kernel(random_map(inp, depth, seed=1000 * depth + 2 * s), n, n_jets)
                K_b = set_kernel(random_map(inp, depth, seed=1000 * depth + 2 * s + 1), n, n_jets)
                add(n, f"map_depth{depth}", f"map_depth{depth}", s, cknna(K_a, K_b, keep))
                add(n, f"map_depth{depth}", "input", s, cknna(K_a, K_in, keep))
                del K_a, K_b
        K_models = {key: set_kernel(inf, n, n_jets) for key, inf in models.items()}
        for (name, s), K in K_models.items():
            add(n, name, "input", s, cknna(K, K_in, keep))
            for o in OBSERVABLES:
                add(n, name, o, s, cknna(K, obs_kernels[o], keep))
        for s in range(mmd.N_RANDOM_SEEDS):
            add(n, "ntp_random", "mpm_random", s, cknna(K_models[("ntp_random", s)], K_models[("mpm_random", s)], keep))
        del K_models, K_in
        gc.collect()
        print(f"=== N = {n} done ({time.time() - t0:.0f} s) ===")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"saved {len(results)} rows to {out_json}")


# ------------------------------------------------------------------ tables and figures
LABEL = {"input": "input", "ntp_random": "NTP random init", "mpm_random": "MPM random init",
         "ntp_pretrained": "NTP pre-trained", "mpm_pretrained": "MPM pre-trained",
         "ntp_finetuned": "NTP fine-tuned", "mpm_finetuned": "MPM fine-tuned",
         **{f"map_depth{d}": f"random map, depth {d}" for d in MAP_DEPTHS},
         "log_multiplicity": "log multiplicity", "log_jet_pt": "log jet pT", "log_jet_mass": "log jet mass",
         "leading_pt_fraction": "leading pT fraction"}
LINE = {"input": ("black", "-", "o"),
        "ntp_random": (C_RND, "-", "o"), "mpm_random": (C_RND, "--", "s"),
        "ntp_pretrained": ("#2E5EAA", "-", "o"), "mpm_pretrained": ("#2E5EAA", "--", "s"),
        "ntp_finetuned": ("#C0392B", "-", "o"), "mpm_finetuned": ("#C0392B", "--", "s")}


def values(results, a, b, n, k):
    """All CKNNA values (over seeds) for rep_a vs rep_b at one N cut and k."""
    return [r["cknna"] for r in results if r["rep_a"] == a and r["rep_b"] == b
            and str(r["n_leading"]) == str(n) and int(r["topk"]) == k]


def fmt(v):
    return "–" if not v else (f"{np.mean(v):.3f}" if len(v) == 1 else f"{np.mean(v):.3f} ({min(v):.2f}–{max(v):.2f})")


def tables(results, k):
    ns = ["all" if n is None else n for n in N_CUTS]
    head = "| | " + " | ".join(f"N = {n}" for n in ns) + " |\n|---|" + "---|" * len(ns)
    lines = [f"# Input structure, set kernel (beta = {BETA:g}), CKNNA at k = {k}",
             "mean over seeds (min–max) where there are several", "",
             "## 1. Random maps of the input (data-only proxy for the random-init alignment)", "", head]
    lines.append("| random NTP vs random MPM (anchor) | " +
                 " | ".join(fmt(values(results, "ntp_random", "mpm_random", n, k)) for n in ns) + " |")
    for d in MAP_DEPTHS:
        lines.append(f"| map pair, depth {d} | " +
                     " | ".join(fmt(values(results, f"map_depth{d}", f"map_depth{d}", n, k)) for n in ns) + " |")
    for d in MAP_DEPTHS:
        lines.append(f"| map vs input, depth {d} | " +
                     " | ".join(fmt(values(results, f"map_depth{d}", "input", n, k)) for n in ns) + " |")
    lines += ["", "## 2. Models vs input", "", head]
    for m in MODELS:
        lines.append(f"| {LABEL[m]} | " + " | ".join(fmt(values(results, m, "input", n, k)) for n in ns) + " |")
    for o in OBSERVABLES:
        ceiling = fmt(values(results, o, o, "all", k))
        lines += ["", f"## 3. vs {LABEL[o]} (ceiling, observable vs itself with other tie-breaking: {ceiling})",
                  "", head]
        for rep in ("input",) + MODELS:
            lines.append(f"| {LABEL[rep]} | " + " | ".join(fmt(values(results, rep, o, n, k)) for n in ns) + " |")
    return "\n".join(lines) + "\n"


def axis_n(ax):
    labels = ["all" if n is None else str(n) for n in N_CUTS]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel("N leading-pT constituents"); ax.grid(alpha=0.15, lw=0.4)


def line(ax, results, a, b, k, colour, ls, marker, label):
    ns = ["all" if n is None else n for n in N_CUTS]
    vals = [values(results, a, b, n, k) for n in ns]
    xs = [i for i, v in enumerate(vals) if v]
    vals = [v for v in vals if v]
    if not vals:
        return
    if max(len(v) for v in vals) > 1:
        ax.fill_between(xs, [min(v) for v in vals], [max(v) for v in vals], color=colour, alpha=0.15, lw=0)
    ax.plot(xs, [np.mean(v) for v in vals], ls=ls, marker=marker, color=colour, ms=3.5, lw=1.3, label=label)


def figures(results, out_dir, k):
    # 1. random maps: pair vs each other (left), each vs the input (right)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    colours = plt.cm.viridis(np.linspace(0, 0.85, len(MAP_DEPTHS)))
    for c, d in zip(colours, MAP_DEPTHS):
        line(axes[0], results, f"map_depth{d}", f"map_depth{d}", k, c, "-", "o", f"depth {d}")
        line(axes[1], results, f"map_depth{d}", "input", k, c, "-", "o", f"depth {d}")
    line(axes[0], results, "ntp_random", "mpm_random", k, C_RND, "--", "s", "random NTP vs random MPM")
    for m in ("ntp_random", "mpm_random"):
        line(axes[1], results, m, "input", k, *LINE[m], LABEL[m])
    axes[0].set_title("two random maps of the input", fontsize=8)
    axes[1].set_title("random map vs the input itself", fontsize=8)
    for ax in axes:
        axis_n(ax)
    axes[0].set_ylim(0, 1.0); axes[0].set_ylabel(f"set-kernel CKNNA at $k = {k}$")
    for ax in axes:
        ax.legend(frameon=False, fontsize=6)
    save(fig, out_dir / f"random_maps_k{k}")

    # 2. models vs input
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for m in MODELS:
        line(ax, results, m, "input", k, *LINE[m], LABEL[m])
    axis_n(ax); ax.set_ylim(0, None); ax.set_ylabel(f"CKNNA(model, input) at $k = {k}$")
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, out_dir / f"models_vs_input_k{k}")

    # 3. vs observables, one panel each; dotted = ceiling
    fig, axes = plt.subplots(1, len(OBSERVABLES), figsize=(2.6 * len(OBSERVABLES), 2.8), sharey=True)
    for ax, o in zip(axes, OBSERVABLES):
        for rep in ("input",) + MODELS:
            line(ax, results, rep, o, k, *LINE[rep], LABEL[rep])
        ceiling = values(results, o, o, "all", k)
        if ceiling:
            ax.axhline(np.mean(ceiling), color="black", ls=":", lw=0.9, label="ceiling (ties)")
        ax.set_title(f"vs {LABEL[o]}", fontsize=8); axis_n(ax)
    axes[0].set_ylim(0, None); axes[0].set_ylabel(f"set-kernel CKNNA at $k = {k}$")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, out_dir / f"vs_observables_k{k}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="only redo tables and figures from the saved json")
    args = ap.parse_args()
    out_json = Path("analysis/cknna") / f"{MAX_BATCHES}batches_input_structure" / "input_structure_results.json"
    if not args.plot_only:
        compute(out_json)
    with open(out_json) as fh:
        results = json.load(fh)
    out_dir = Path("analysis/embedding_alignment/input_structure") / f"{MAX_BATCHES}batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in TOPKS:
        (out_dir / f"summary_k{k}.md").write_text(tables(results, k))
        figures(results, out_dir, k)
    print(tables(results, 10))
    print(f"figures and tables in {out_dir}")
