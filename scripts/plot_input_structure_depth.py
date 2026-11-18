"""How much of the input's neighbourhood structure survives random layers: CKNNA(representation, input).

x-axis: number of random non-linear layers between the input and the representation.
    random per-constituent maps of compute_input_structure.py   depth 0 (linear), 1, 2, 4 GELU layers
    random-init NTP / MPM backbones                             8 transformer blocks (one GELU MLP each)
The maps never mix constituents, the transformers do (attention), so the gap between the map curve
and the transformer points at the same depth is what particle mixing costs.

Three ways of comparing jets, all on the full jets (all constituents, start token excluded):
    set-based            MMD set kernel over all constituents, beta = 0 (from compute_input_structure.py)
    pooled               mean over constituents, unnormalized CKNNA (gabbro.metrics.cknna)
    pooled, normalized   mean over constituents, normalized CKNNA (clamp + L2, cknna_with_normalized)
The input is the 14 preprocessed constituent features the backbone receives; for the pooled versions
the input, maps and models are all mean-pooled the same way. Error bars: min-max over seeds.

Run from the repository root after compute_input_structure.py; --plot-only skips the pooled computation.
"""

import argparse
import json
from pathlib import Path

import compute_constituent_mmd_kernel as mmd  # sets up sys.path / project root on import
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import compute_cknna_mpm_vs_ntp_on_aoj as base
import compute_input_structure as inp_struct
import gabbro.metrics.cknna as cknna
import gabbro.metrics.cknna_with_normalized as cknna_norm
import gabbro.metrics.constituent_kernels as ck
from plot_cknna import save

TRANSFORMER_DEPTH = 8          # backbone n_blocks for both NTP and MPM
POOLED = {"pooled": cknna.sweep_topk, "pooled_normalized": cknna_norm.sweep_topk_normalized}
OUT_DIR = Path("analysis/cknna") / f"{inp_struct.MAX_BATCHES}batches_input_structure"
FIG_DIR = Path("analysis/embedding_alignment/input_structure") / f"{inp_struct.MAX_BATCHES}batches"


def pool(inference, n_jets):
    """Mean over each jet's constituents of a flat constituent_inference-style output."""
    E, jet_idx, _, _ = inference
    sums = torch.zeros(n_jets, E.shape[1], dtype=torch.float64).index_add_(0, jet_idx, E.double())
    return sums / torch.bincount(jet_idx, minlength=n_jets).clamp_min(1)[:, None]


def compute_pooled(out_json):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "experiment=example_experiment_backbone_and_head_allfeat_ntp_classifier_head",
            "feature_dict=feature_dict_all_without_cuts"])
    bs = cfg.data.batch_size.train
    n_jets = inp_struct.MAX_BATCHES * bs
    max_len = cfg.data.dataset_kwargs_common.pad_length - 1
    dataloader = base.get_aoj_dataloader(
        cfg, "/srv/beegfs/scratch/groups/rodem/lumea/datasets/aspenJets/RunG_batch0.h5",
        n_jets_per_file=(inp_struct.MAX_BATCHES + 1) * bs)
    batches = ck.collect_batches(dataloader, n_jets)
    pt_getter = ck.make_pt_getter(cfg.data.dataset_kwargs_common.feature_dict, pt_key="part_pt")

    inp = inp_struct.input_inference(batches, pt_getter, n_jets, max_len)
    reps = {}   # (name, seed) -> inference; same seeds and weights as compute_input_structure.py
    for d in inp_struct.MAP_DEPTHS:
        for s in range(inp_struct.N_MAP_SEEDS):
            reps[(f"map_depth{d}", s)] = inp_struct.random_map(inp, d, seed=1000 * d + 2 * s)
    ck_ntp = mmd.checkpoints(base.NTP_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[0])
    ck_mpm = mmd.checkpoints(base.MPM_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[1])
    for s in range(mmd.N_RANDOM_SEEDS):
        for name, model in (("ntp_random", base.random_init_model(base.first_real_ckpt(ck_ntp), s)),
                            ("mpm_random", base.random_init_model(base.first_real_ckpt(ck_mpm), 1000 + s))):
            reps[(name, s)] = ck.constituent_inference(batches, model, pt_getter, n_jets, base.device)
            del model
            mmd.free()

    keep = torch.ones(n_jets, dtype=torch.bool)
    for _, _, _, bad in [inp, *reps.values()]:
        keep[bad.long()] = False
    print(f"dropping {int((~keep).sum())} of {n_jets} jets with non-finite input or embedding")
    U_in = pool(inp, n_jets)[keep]

    results = []
    for (name, s), inf in reps.items():
        U = pool(inf, n_jets)[keep]
        for metric, sweep in POOLED.items():
            rows = sweep(U, U_in, topks=inp_struct.TOPKS, n_sub=inp_struct.N_SUB, n_draws=inp_struct.N_DRAWS)
            results += [{"metric": metric, "rep": name, "seed": s, **r} for r in rows]
            k10 = next(r for r in rows if r["topk"] == 10)
            print(f"{metric:>17} {name:>12} seed {s}: CKNNA(rep, input, k=10) = {k10['cknna']:.3f}")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"saved {len(results)} rows to {out_json}")


def collect(set_results, pooled_results, k):
    """{metric: {rep: [cknna per seed]}} for every representation vs the input, full jets."""
    out = {"set-based": {}, **{m: {} for m in POOLED}}
    for r in set_results:
        if r["rep_b"] == "input" and str(r["n_leading"]) == "all" and int(r["topk"]) == k:
            out["set-based"].setdefault(r["rep_a"], []).append(r["cknna"])
    for r in pooled_results:
        if int(r["topk"]) == k:
            out[r["metric"]].setdefault(r["rep"], []).append(r["cknna"])
    return out


def figure(vals, k):
    style = {"set-based": ("black", "set-based (all constituents)"),
             "pooled": ("#2E5EAA", "pooled"),
             "pooled_normalized": ("#D98C00", "pooled, normalized")}
    # the two random transformers both have 8 blocks; they get one tick each, side by side at depth 8
    x_tf = {"ntp_random": TRANSFORMER_DEPTH - 0.5, "mpm_random": TRANSFORMER_DEPTH + 0.5}
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for metric, by_rep in vals.items():
        colour, label = style[metric]
        depths = [d for d in inp_struct.MAP_DEPTHS if by_rep.get(f"map_depth{d}")]
        v = [by_rep[f"map_depth{d}"] for d in depths]
        ax.errorbar(depths, [np.mean(x) for x in v],
                    yerr=[[np.mean(x) - min(x) for x in v], [max(x) - np.mean(x) for x in v]],
                    fmt="-o", color=colour, ms=4, lw=1.3, capsize=2, label=f"{label}: random per-particle map")
        for rep, marker in (("ntp_random", "^"), ("mpm_random", "s")):
            x = by_rep.get(rep)
            if x:
                ax.errorbar([x_tf[rep]], [np.mean(x)],
                            yerr=[[np.mean(x) - min(x)], [max(x) - np.mean(x)]], fmt=marker, color=colour,
                            mfc="white", ms=5, capsize=2)
    for marker, name in (("^", "random NTP (causal)"), ("s", "random MPM")):
        ax.plot([], [], marker, color="gray", mfc="white", ls="none", label=f"{name}, 8 blocks")
    ax.axvspan(6.6, 9.4, color="gray", alpha=0.06, lw=0)
    ax.text(TRANSFORMER_DEPTH, 0.97, "random transformers", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6, color="gray")
    ax.set_xticks(list(inp_struct.MAP_DEPTHS) + list(x_tf.values()))
    ax.set_xticklabels([str(d) for d in inp_struct.MAP_DEPTHS] + [f"{TRANSFORMER_DEPTH}\nNTP", f"{TRANSFORMER_DEPTH}\nMPM"])
    ax.set_xlim(-0.6, 9.4); ax.set_ylim(0, 1.0)
    ax.set_xlabel("random non-linear layers after the input")
    ax.set_ylabel(f"CKNNA(representation, input) at $k = {k}$")
    ax.grid(alpha=0.15, lw=0.4)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=2)
    return fig


def table(vals, k):
    reps = [f"map_depth{d}" for d in inp_struct.MAP_DEPTHS] + ["ntp_random", "mpm_random"]
    names = [f"map, depth {d}" for d in inp_struct.MAP_DEPTHS] + ["random NTP (8 blocks)", "random MPM (8 blocks)"]
    lines = [f"CKNNA(representation, input), full jets, k = {k}; mean (min-max) over seeds", "",
             "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for metric, by_rep in vals.items():
        cells = [f"{np.mean(by_rep[r]):.3f} ({min(by_rep[r]):.2f}-{max(by_rep[r]):.2f})" if by_rep.get(r) else "-"
                 for r in reps]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--metrics", nargs="+", default=["set-based", *POOLED], choices=["set-based", *POOLED],
                    help="series to draw; a subset is saved with the metric names in the file name")
    args = ap.parse_args()
    pooled_json = OUT_DIR / "input_structure_pooled_results.json"
    if not args.plot_only:
        compute_pooled(pooled_json)
    with open(OUT_DIR / "input_structure_results.json") as fh:
        set_results = json.load(fh)
    with open(pooled_json) as fh:
        pooled_results = json.load(fh)
    for k in inp_struct.TOPKS:
        vals = {m: v for m, v in collect(set_results, pooled_results, k).items() if m in args.metrics}
        tag = "" if len(vals) == 1 + len(POOLED) else "_" + "_".join(m.replace("-", "_") for m in vals)
        save(figure(vals, k), FIG_DIR / f"depth_vs_input{tag}_k{k}")
        (FIG_DIR / f"depth_vs_input{tag}_k{k}.md").write_text(table(vals, k))
    print(table({m: v for m, v in collect(set_results, pooled_results, 10).items() if m in args.metrics}, 10))
    print(f"figures in {FIG_DIR}")
