"""Results figure: CKNNA between the NTP and MPM jet representations on AspenOpenJets.

Reads the evaluation output directly, e.g.

    python scripts/plot_cknna.py analysis/cknna/50batches_per-epoch/cknna_results.json \
        --metric cknna_pooled_unnormalized

Figures go to <out-root>/<metric>/<run-label>/ (default out-root analysis/embedding_alignment,
run-label = name of the json's parent directory), each as .pdf and .png:

    vs_k/cknna_vs_k.*            : alignment vs neighbourhood size k, mean over trained epochs with min-max band;
                     in grey the random-init baseline of the pre-training architectures
                     (mean and min-max over seeds).
    vs_epoch/cknna_vs_epoch_k<k>.* : alignment vs epoch at that k, with per-checkpoint uncertainties;
                     one figure for every k in the results. Pre-training and fine-tuning are
                     overlaid on one epoch axis. The first point ("start") is the checkpoint each
                     run starts from: random weights for pre-training, the loaded pre-trained
                     backbone ("init") for fine-tuning.

Expected JSON: a list of records. Alignment records carry
    pair, checkpoint_idx, epoch, topk, n, cknna, cknna_std, null, null_std, z
Records whose "pair" ends in _identity, _permutation or _identity_map are
self-comparison baselines and are skipped (--include-baselines overrides).
Records ending in _random_seeds are only used for the grey baseline in the vs-k plot.
Non-finite values (NaN, Infinity, nulls, missing keys) are dropped per point
rather than per checkpoint, so a partial run still plots: a missing cknna is a
gap in the line, and a missing uncertainty is drawn as a zero-length error bar
and treated as zero in the printed significance, which then overstates it.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})

C_RND = "#808080"
# known pairs get fixed colours and markers; anything else falls back to the cycle
STYLE = {
    "ntp_vs_mpm":                     ("#2E5EAA", "-o", "pre-trained"),
    "ntp_bidir_vs_mpm":               ("#1B7F5A", "-^", "pre-trained, NTP w/o causal mask"),
    "ntp_finetuned_vs_mpm_finetuned": ("#C0392B", "-s", "fine-tuned"),
    # single models against the jet multiplicity (see MULT_SUFFIX)
    "ntp_vs_multiplicity":            ("#2E5EAA", "-o", "NTP pre-trained"),
    "ntp_bidir_vs_multiplicity":      ("#1B7F5A", "-^", "NTP pre-trained, w/o causal mask"),
    "mpm_vs_multiplicity":            ("#7FA7E0", "--o", "MPM pre-trained"),
    "ntp_finetuned_vs_multiplicity":  ("#C0392B", "-s", "NTP fine-tuned"),
    "mpm_finetuned_vs_multiplicity":  ("#E59A90", "--s", "MPM fine-tuned"),
}
FALLBACK = ["#A929B4", "#D98C00", "#444444"]
BASELINE_SUFFIXES = ("_identity", "_permutation", "_identity_map")
RANDOM_SUFFIX = "_random_seeds"
# CKNNA of one model against log(multiplicity) alone; plotted separately from the model pairs
MULT_SUFFIX = "_vs_multiplicity"
# CKNNA of one model against its own mean-pooled input features; also plotted separately
INPUT_SUFFIX = "_vs_input"
REFERENCE_SUFFIXES = (MULT_SUFFIX, INPUT_SUFFIX)
# same colours and markers for a model against either reference
STYLE.update({k[: -len(MULT_SUFFIX)] + INPUT_SUFFIX: v for k, v in STYLE.items() if k.endswith(MULT_SUFFIX)})


# ------------------------------------------------------------------ loading
def _f(x):
    """float(x) if finite, else nan. Covers None, strings, NaN and inf."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def load(path, include_baselines=False, reference=None):
    """JSON records -> {pair: {"epochs": [...], "k": [...], arrays (n_ckpt, n_k)}}.

    reference=None returns the model pairs; a suffix from REFERENCE_SUFFIXES returns only the
    single-model-vs-that-reference records of trained checkpoints (random-init ones are
    summarised separately).
    """
    with open(path) as fh:
        records = json.load(fh)          # json.load accepts NaN / Infinity

    rows = [r for r in records if "topk" in r and "cknna" in r and not r["pair"].endswith(RANDOM_SUFFIX)
            and RANDOM_SUFFIX not in r["pair"]
            and (r["pair"].endswith(reference) if reference else not r["pair"].endswith(REFERENCE_SUFFIXES))]
    if not include_baselines:
        rows = [r for r in rows if not r["pair"].endswith(BASELINE_SUFFIXES)]
    if reference:
        return load_rows(rows)
    if not rows:
        raise SystemExit(f"no alignment records found in {path}")
    return load_rows(rows)


def load_rows(rows):
    out = {}
    for pair in dict.fromkeys(r["pair"] for r in rows):          # keeps file order
        sub = [r for r in rows if r["pair"] == pair]
        ks = sorted({int(r["topk"]) for r in sub})
        ckpts = sorted({int(r["checkpoint_idx"]) for r in sub})
        shape = (len(ckpts), len(ks))
        d = {key: np.full(shape, np.nan) for key in ("cknna", "cknna_std", "null", "null_std")}
        epochs = {}
        for r in sub:
            i, j = ckpts.index(int(r["checkpoint_idx"])), ks.index(int(r["topk"]))
            for key in d:
                d[key][i, j] = _f(r.get(key))
            epochs[i] = r.get("epoch", ckpts[i])
        d["k"] = np.array(ks)
        d["epochs"] = [str(epochs.get(i, ckpts[i])) for i in range(len(ckpts))]
        d["n"] = next((r.get("n") for r in sub if r.get("n") is not None), None)
        out[pair] = d
    return out


def load_random_seeds(path):
    """JSON records -> {pair: (k array, CKNNA array (n_seeds, n_k))} for the random-init pairs."""
    with open(path) as fh:
        records = json.load(fh)
    by_pair = {}
    for r in records:
        if "topk" in r and r.get("pair", "").endswith(RANDOM_SUFFIX) and "cknna" in r:
            pair = r["pair"][: -len(RANDOM_SUFFIX)]
            by_pair.setdefault(pair, {}).setdefault(int(r["checkpoint_idx"]), {})[int(r["topk"])] = _f(r.get("cknna"))
    out = {}
    for pair, by_seed in by_pair.items():
        ks = sorted({k for vals in by_seed.values() for k in vals})
        out[pair] = (np.array(ks), np.array([[vals.get(k, np.nan) for k in ks] for vals in by_seed.values()]))
    return out


def style_for(pair, i):
    if pair in STYLE:
        return STYLE[pair]
    return FALLBACK[i % len(FALLBACK)], "-^", pair.replace("_", " ")


# ------------------------------------------------------------------ plotting
def save(fig, stem):
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_vs_k(data, kk, random_seeds):
    """CKNNA vs k, mean over trained epochs, min-max band.

    In grey: the same for independently seeded random initialisations of the pre-training
    architectures, mean over seeds with a min-max band. Fine-tuning starts from the
    pre-trained backbone, not from random weights, so its random-init pair is not drawn.
    """
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for i, (pair, (k, y)) in enumerate(random_seeds.items()):
        if "finetuned" in pair or pair not in data or not np.isfinite(y).any():
            continue
        marker = style_for(pair, i)[1][-1]
        label = "random init" + (" (NTP w/o causal mask)" if "bidir" in pair else "")
        lo, hi = np.nanmin(y, 0), np.nanmax(y, 0)
        ok = np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(k[ok], lo[ok], hi[ok], color=C_RND, alpha=0.15, lw=0, zorder=1)
        ax.plot(k, np.nanmean(y, 0), "--" + marker, color=C_RND, ms=3, lw=1.0, mfc="white", zorder=4,
                label=label)
    for i, (pair, d) in enumerate(data.items()):
        colour, marker, label = style_for(pair, i)
        # untrained / init / pretraining-snapshot points are not trained epochs
        trained = [n for n, e in enumerate(d["epochs"]) if e.isdigit()] or list(range(len(d["epochs"])))
        y, k = d["cknna"][trained], d["k"]
        if not np.isfinite(y).any():
            continue
        lo, hi = np.nanmin(y, 0), np.nanmax(y, 0)
        ok = np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(k[ok], lo[ok], hi[ok], color=colour, alpha=0.20, lw=0, zorder=2)
        ax.plot(k, np.nanmean(y, 0), marker, color=colour, ms=3.5, lw=1.4, zorder=3, label=label)

    ax.set_xscale("log")
    ax.set_xticks(kk); ax.set_xticklabels([str(v) for v in kk])
    ax.set_xlabel("neighbourhood size $k$"); ax.set_ylabel("CKNNA")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6); ax.grid(alpha=0.15, which="both", lw=0.4)
    return fig


# the checkpoint each run starts from, drawn at x = 0: random weights for pre-training,
# the loaded pre-trained backbone for fine-tuning
START_EPOCH = {False: "untrained", True: "init"}
START_LABEL = {False: "random init", True: "pre-trained backbone"}


def x_of(pair, epoch):
    """x position of a checkpoint: 0 for the run's starting point, 1 + epoch for trained epochs.

    Other labels (the fine-tuning run's own random init, ft_src, best, last) are not drawn.
    """
    if epoch == START_EPOCH["finetuned" in pair]:
        return 0
    if epoch.isdigit():
        return 1 + int(epoch)
    return None


def plot_vs_epoch(data, topk, ylabel="CKNNA (NTP vs MPM)"):
    """CKNNA vs epoch at a fixed k, pre-training and fine-tuning overlaid on one epoch axis."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    xmax = 0
    for i, (pair, d) in enumerate(data.items()):
        if topk not in d["k"]:
            continue
        colour, marker, label = style_for(pair, i)
        j = list(d["k"]).index(topk)
        y, e = d["cknna"][:, j], np.where(np.isfinite(d["cknna_std"][:, j]), d["cknna_std"][:, j], 0.0)
        pts = sorted((x, n) for n, x in ((n, x_of(pair, ep)) for n, ep in enumerate(d["epochs"])) if x is not None)
        xs, idx = [x for x, _ in pts], [n for _, n in pts]
        start = START_LABEL["finetuned" in pair]
        ax.errorbar(xs, y[idx], yerr=e[idx], fmt=marker, color=colour, ms=4, lw=1.4, capsize=2,
                    label=f"{label} (start: {start})")
        xmax = max([xmax] + xs)

    ticks = [0] + list(range(1, xmax + 1, 2))
    ax.set_xticks(ticks); ax.set_xticklabels(["start"] + [str(x - 1) for x in ticks[1:]])
    ax.set_xlim(-0.6, xmax + 0.6); ax.set_ylim(0, None)
    ax.set_xlabel("epoch"); ax.set_ylabel(f"{ylabel} at $k = {topk}$")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6); ax.grid(alpha=0.15, lw=0.4)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?", default="analysis/cknna/50batches_per-epoch/cknna_results.json")
    ap.add_argument("--out-root", default="analysis/embedding_alignment",
                    help="figures go to <out-root>/<metric>/<run-label>/{vs_k,vs_epoch}/")
    ap.add_argument("--metric", default="cknna_pooled_unnormalized",
                    help="metric subdirectory, e.g. cknna_pooled_normalized, constituent_mmd")
    ap.add_argument("--run-label", default=None,
                    help="run subdirectory; defaults to the name of the json's parent directory")
    ap.add_argument("--include-baselines", action="store_true",
                    help="also plot the *_identity / *_permutation self-comparisons")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="drop pairs whose name contains any of these substrings, e.g. ntp_bidir")
    args = ap.parse_args()

    keep = lambda pair: not any(s in pair for s in args.exclude)  # noqa: E731
    data = {p: d for p, d in load(args.json, args.include_baselines).items() if keep(p)}
    random_seeds = {p: v for p, v in load_random_seeds(args.json).items() if keep(p)}
    kk = next(iter(data.values()))["k"]

    run_label = args.run_label or Path(args.json).resolve().parent.name
    out_dir = Path(args.out_root) / args.metric / run_label
    for sub in ("vs_k", "vs_epoch"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    save(plot_vs_k(data, kk, random_seeds), out_dir / "vs_k" / "cknna_vs_k")
    for k in kk:
        save(plot_vs_epoch(data, int(k)), out_dir / "vs_epoch" / f"cknna_vs_epoch_k{int(k)}")
    # single models against a reference: (suffix, figure dir, y label, summary heading)
    references = [(MULT_SUFFIX, "vs_multiplicity", "multiplicity", "log(multiplicity)"),
                  (INPUT_SUFFIX, "vs_input", "input", "its mean-pooled input features")]
    ref_data = {}
    for suffix, sub, what, _ in references:
        ref_data[suffix] = {p: d for p, d in load(args.json, reference=suffix).items() if keep(p)}
        if not ref_data[suffix]:
            continue
        (out_dir / sub).mkdir(exist_ok=True)
        for k in kk:
            save(plot_vs_epoch(ref_data[suffix], int(k), ylabel=f"CKNNA (model vs {what})"),
                 out_dir / sub / f"cknna_{sub}_k{int(k)}")
    print(f"saved figures to {out_dir}")

    # ---- numbers for the text, printed and written to <out_dir>/summary.txt
    lines = [f"source: {args.json}"]
    if random_seeds:
        lines.append("\nrandom init, mean (min-max) over seeds, per k:")
        for pair, (k, y) in random_seeds.items():
            lines.append(f"  {pair}: " + ", ".join(
                f"k={kv}: {np.nanmean(y[:, j]):.3f} ({np.nanmin(y[:, j]):.3f}-{np.nanmax(y[:, j]):.3f})"
                for j, kv in enumerate(k)))
    with open(args.json) as fh:
        records = json.load(fh)
    for suffix, _, _, heading in references:
        if not ref_data[suffix]:
            continue
        lines.append(f"\nCKNNA of each model vs {heading}, per checkpoint (k=10):")
        for pair, d in ref_data[suffix].items():
            j = list(d["k"]).index(10) if 10 in d["k"] else 0
            lines.append(f"  {pair}: " + ", ".join(f"{e}={v:.3f}" for e, v in zip(d["epochs"], d["cknna"][:, j])))
        rnd = [r for r in records if RANDOM_SUFFIX in r.get("pair", "") and r["pair"].endswith(suffix)
               and keep(r["pair"]) and r.get("topk") == 10]
        for pair in dict.fromkeys(r["pair"] for r in rnd):
            v = [_f(r["cknna"]) for r in rnd if r["pair"] == pair]
            lines.append(f"  {pair}: mean {np.nanmean(v):.3f} ({np.nanmin(v):.3f}-{np.nanmax(v):.3f}) over seeds")
    n = next((d["n"] for d in data.values() if d["n"]), "?")
    lines.append(f"n = {n} per draw;  k = {[int(v) for v in kk]}")
    for pair, d in data.items():
        y, s, nl, ns = d["cknna"], d["cknna_std"], d["null"], d["null_std"]
        sig = (y - nl) / np.sqrt(np.nan_to_num(s) ** 2 + np.nan_to_num(ns) ** 2)
        lines.append(f"\n{pair}  (epochs {', '.join(d['epochs'])})")
        lines.append(f"  CKNNA {np.nanmin(y):.3f}-{np.nanmax(y):.3f}   "
                     f"null {np.nanmin(nl):.4f}-{np.nanmax(nl):.4f}   "
                     f"sigma {np.nanmin(sig):.1f}-{np.nanmax(sig):.1f}")
        # the monotonicity check uses only checkpoints with a numeric epoch:
        # "best" is not a later training step and must not enter the trend
        ordered = [i for i, e in enumerate(d["epochs"]) if e.lstrip("-").isdigit()]
        for j, k in enumerate(d["k"]):
            col = y[:, j]
            seq = col[ordered]
            seq = seq[np.isfinite(seq)]
            trend = "rising" if len(seq) > 1 and np.all(np.diff(seq) > 0) else "not monotonic"
            lines.append(f"  k={k:>3}: " + ", ".join("nan" if not np.isfinite(v) else f"{v:.3f}"
                                                     for v in col) + f"   [{trend} over epochs]")
    if len(data) == 2:
        a, b = data.values()
        r = np.nanmean(b["cknna"], 0) / np.nanmean(a["cknna"], 0)
        lines.append("\nratio second/first per k: " +
                     ", ".join(f"k={k}: {v:.1f}" for k, v in zip(kk, r)))
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()