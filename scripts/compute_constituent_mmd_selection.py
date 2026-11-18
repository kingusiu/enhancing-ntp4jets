"""Is the NTP / MPM alignment on the hardest constituents about pT, set size or multiplicity?

The backbones always see the full jet. From their constituent embeddings, N constituents per jet
are then selected in three ways (constituent_kernels.select_constituents):

    hardest   the N leading-pT constituents (as in compute_constituent_mmd_leading_n.py)
    softest   the N lowest-pT constituents
    random    N constituents at random per jet, the same ones in both models

and the set-kernel CKNNA (MMD over the selected constituents, beta = 0, sigma = median distance)
is computed separately within three multiplicity bins (tertiles of the jets) and for all jets.
If "hardest" stays above "softest" and "random" at the same N, the effect is about pT, not set
size; if the pattern is the same in every multiplicity bin, multiplicity does not explain it.
Within a bin all jets have similar multiplicity, so the bins also control for it directly.

Checkpoints: start, middle and end of pre-training and of fine-tuning, plus the random-init
baseline (causal NTP vs MPM, same seeds and weights as the other scripts). Run from the
repository root; --plot-only redoes the figures and tables from the saved json.
"""

import argparse
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
N_SELECT = (1, 2, 3, 5, 10, 20)
MODES = ("hardest", "softest", "random")
BETA, SIGMA_FACTOR = 0.0, 1.0
TOPKS = (5, 10, 25, 50)
N_SUB, N_DRAWS, N_PERM = 1500, 5, 1
N_BINS = 3
# (pair, model a, model b, epoch labels): start / middle / end of each training stage
STAGES = (("ntp_vs_mpm", "ntp", "mpm", ("untrained", 6, 13)),
          ("ntp_finetuned_vs_mpm_finetuned", "ntp_finetuned", "mpm_finetuned", ("init", 7, 15)))
RANDOM = "ntp_vs_mpm_random_seeds"
OUT_DIR = Path("analysis/cknna") / f"{MAX_BATCHES}batches_constituent_mmd_selection_beta{BETA}"
FIG_DIR = Path("analysis/embedding_alignment/constituent_mmd") / f"{MAX_BATCHES}batches_selection_beta{BETA}"


def kernel(inference, n, mode, n_jets):
    E, jet_idx, pt, _ = ck.select_constituents(inference, n, mode, seed=n)
    sigma = SIGMA_FACTOR * ck.median_heuristic_sigma(E)
    return ck.mmd_kernels(E, jet_idx, pt, n_jets, (BETA,), (sigma,), device=base.device)[(BETA, sigma)]


def compare(inf_a, inf_b, bins, n_jets):
    """Rows for every (mode, N, multiplicity bin) for one pair of models."""
    rows = []
    for mode in MODES:
        for n in N_SELECT:
            Ka, Kb = kernel(inf_a, n, mode, n_jets), kernel(inf_b, n, mode, n_jets)
            for bin_label, idx in bins.items():
                for r in ck.cknna_sweep(Ka[idx][:, idx].to(base.device), Kb[idx][:, idx].to(base.device),
                                        TOPKS, N_SUB, N_DRAWS, n_perm=N_PERM):
                    rows.append({"mode": mode, "n_select": n, "mult_bin": bin_label, **r})
            del Ka, Kb
    return rows


def report(rows):
    for b in dict.fromkeys(r["mult_bin"] for r in rows):
        parts = []
        for mode in MODES:
            v = [f"{r['cknna']:.2f}" for r in rows if r["mult_bin"] == b and r["mode"] == mode and r["topk"] == 10]
            parts.append(f"{mode}: " + " ".join(v))
        print(f"  {b:>12} k=10  " + " | ".join(parts))


def compute(out_json):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "experiment=example_experiment_backbone_and_head_allfeat_ntp_classifier_head",
            "feature_dict=feature_dict_all_without_cuts"])
    bs = cfg.data.batch_size.train
    n_jets = MAX_BATCHES * bs
    max_len = cfg.data.dataset_kwargs_common.pad_length - 1        # the models prepend a start token
    dataloader = base.get_aoj_dataloader(
        cfg, "/srv/beegfs/scratch/groups/rodem/lumea/datasets/aspenJets/RunG_batch0.h5",
        n_jets_per_file=(MAX_BATCHES + 1) * bs)
    batches = ck.collect_batches(dataloader, n_jets)
    pt_getter = ck.make_pt_getter(cfg.data.dataset_kwargs_common.feature_dict, pt_key="part_pt")

    ckpts = {
        "ntp": mmd.checkpoints(base.NTP_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[0]),
        "mpm": mmd.checkpoints(base.MPM_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[1]),
        "ntp_finetuned": mmd.checkpoints(base.NTP_FINETUNE_RUN, True),
        "mpm_finetuned": mmd.checkpoints(base.MPM_FINETUNE_RUN, True),
    }
    by_label = lambda paths, label: next(p for p in paths if base.epoch_label_from_ckpt_path(p) == label)  # noqa: E731
    jobs = []   # (pair, epoch label, checkpoint index, model makers)
    for pair, a, b, labels in STAGES:
        for label in labels:
            pa, pb = by_label(ckpts[a], label), by_label(ckpts[b], label)
            jobs.append((pair, label, ckpts[a].index(pa),
                         lambda pa=pa, pb=pb, a=a, b=b: (base.load_model(pa, base.first_real_ckpt(ckpts[a])),
                                                         base.load_model(pb, base.first_real_ckpt(ckpts[b])))))
    for s in range(mmd.N_RANDOM_SEEDS):
        jobs.append((RANDOM, f"seed{s}", s,
                     lambda s=s: (base.random_init_model(base.first_real_ckpt(ckpts["ntp"]), s),
                                  base.random_init_model(base.first_real_ckpt(ckpts["mpm"]), 1000 + s))))

    # multiplicity tertiles over the jets (constituents the models see, i.e. truncated like them)
    mult = torch.cat([b["part_mask"][:, :max_len].sum(1) for b in batches])[:n_jets]
    results, bins = [], None
    for pair, label, idx, make in jobs:
        t0 = time.time()
        model_a, model_b = make()
        inf_a = ck.constituent_inference(batches, model_a, pt_getter, n_jets, base.device)
        inf_b = ck.constituent_inference(batches, model_b, pt_getter, n_jets, base.device)
        del model_a, model_b
        mmd.free()
        if bins is None:   # the same jets are non-finite for every model (bad input), so fix the bins once
            keep = torch.ones(n_jets, dtype=torch.bool)
            for bad in (inf_a[3], inf_b[3]):
                keep[bad.long()] = False
            edges = torch.quantile(mult[keep].double(), torch.linspace(0, 1, N_BINS + 1, dtype=torch.float64))
            bins = {}
            for i in range(N_BINS):
                lo, hi = int(edges[i].ceil()), int(edges[i + 1].floor())
                sel = keep & (mult >= lo) & ((mult <= hi) if i == N_BINS - 1 else (mult < edges[i + 1]))
                bins[f"N {int(mult[sel].min())}-{int(mult[sel].max())}"] = sel.nonzero().squeeze(1)
            bins["all jets"] = keep.nonzero().squeeze(1)
            print("multiplicity bins:", {k: len(v) for k, v in bins.items()})
        rows = compare(inf_a, inf_b, bins, n_jets)
        print(f"=== {pair} | {label} ({time.time() - t0:.0f} s) ===")
        report(rows)
        results += [{"pair": pair, "checkpoint_idx": idx, "epoch": label, **r} for r in rows]
        del inf_a, inf_b

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"saved {len(results)} rows to {out_json}")


# ------------------------------------------------------------------ tables and figures
STATE = {  # row of the figure: (title, pair, epoch labels averaged over)
    "random init": (RANDOM, None),
    "end of pre-training (epoch 13)": ("ntp_vs_mpm", ("13",)),
    "end of fine-tuning (epoch 15)": ("ntp_finetuned_vs_mpm_finetuned", ("15",)),
}
MODE_STYLE = {"hardest": ("#C0392B", "-o", "N hardest"), "softest": ("#2E5EAA", "-s", "N softest"),
              "random": (C_RND, "--^", "N random")}


def values(results, pair, epochs, mode, n, b, k):
    return [r["cknna"] for r in results if r["pair"] == pair and r["mode"] == mode and r["n_select"] == n
            and r["mult_bin"] == b and int(r["topk"]) == k and (epochs is None or str(r["epoch"]) in epochs)]


def figure(results, k):
    bins = list(dict.fromkeys(r["mult_bin"] for r in results))
    fig, axes = plt.subplots(len(STATE), len(bins), figsize=(2.5 * len(bins), 2.2 * len(STATE)),
                             sharex=True, sharey=True, squeeze=False)
    xs = list(range(len(N_SELECT)))
    for row, (title, (pair, epochs)) in enumerate(STATE.items()):
        for col, b in enumerate(bins):
            ax = axes[row, col]
            for mode in MODES:
                colour, marker, label = MODE_STYLE[mode]
                v = [values(results, pair, epochs, mode, n, b, k) for n in N_SELECT]
                if not all(v):
                    continue
                if max(len(x) for x in v) > 1:
                    ax.fill_between(xs, [min(x) for x in v], [max(x) for x in v], color=colour, alpha=0.15, lw=0)
                ax.plot(xs, [np.mean(x) for x in v], marker, color=colour, ms=3, lw=1.2, label=label)
            if row == 0:
                ax.set_title(f"multiplicity {b}" if b != "all jets" else "all jets", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{title}\nCKNNA at $k = {k}$", fontsize=7)
            ax.grid(alpha=0.15, lw=0.4)
    for ax in axes[-1]:
        ax.set_xticks(xs); ax.set_xticklabels([str(n) for n in N_SELECT])
        ax.set_xlabel("N selected constituents")
    axes[0, 0].set_ylim(0, None)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=3)
    return fig


def table(results, k):
    bins = list(dict.fromkeys(r["mult_bin"] for r in results))
    lines = [f"set-kernel CKNNA (beta = {BETA:g}) at k = {k} on N selected constituents per jet; the backbones "
             "see the full jet. Random init: mean over seeds. Cells: hardest / softest / random.", ""]
    for title, (pair, epochs) in STATE.items():
        lines += [f"## {title}", "", "| N | " + " | ".join(bins) + " |", "|---|" + "---|" * len(bins)]
        for n in N_SELECT:
            cells = []
            for b in bins:
                cells.append(" / ".join(f"{np.mean(v):.2f}" if (v := values(results, pair, epochs, m, n, b, k))
                                        else "-" for m in MODES))
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    out_json = OUT_DIR / "cknna_mmd_selection_results.json"
    if not args.plot_only:
        compute(out_json)
    with open(out_json) as fh:
        results = json.load(fh)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for k in TOPKS:
        save(figure(results, k), FIG_DIR / f"selection_vs_n_k{k}")
        (FIG_DIR / f"summary_k{k}.md").write_text(table(results, k))
    print(table(results, 10))
    print(f"figures and tables in {FIG_DIR}")
