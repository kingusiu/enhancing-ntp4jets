"""Results figure: CKNNA between the NTP and MPM jet representations on AspenOpenJets.

Reads the evaluation output directly, e.g.

    python plot_cknna.py analysis/cknna/cknna_results.json -o fig/cknna_alignment

Two kinds of figure are written, each as .pdf and .png:

    <stem>_vs_k.*  : alignment vs neighbourhood size k, mean over trained epochs with min-max band.
    <stem>_k<k>.*  : alignment vs checkpoint at that k, with per-checkpoint uncertainties;
                     one figure for every k in the results. The x-axis is one timeline:
                     untrained | pre-training epochs | init | fine-tuning epochs. "init" is the
                     pretraining snapshot that fine-tuning started from (both curves meet there).
                     Grey bands are the permutation null; hatched bands are the min-max over
                     independently seeded random initialisations (pairs ending in _random_seeds).

Expected JSON: a list of records. Alignment records carry
    pair, checkpoint_idx, epoch, topk, n, cknna, cknna_std, null, null_std, z
Records whose "pair" ends in _identity, _permutation or _identity_map are
self-comparison baselines and are skipped (--include-baselines overrides).
Records ending in _random_seeds are only used for the hatched bands.
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

C_NUL = "#666666"
# known pairs get fixed colours and markers; anything else falls back to the cycle
STYLE = {
    "ntp_vs_mpm":                     ("#2E5EAA", "-o", "pre-trained"),
    "ntp_finetuned_vs_mpm_finetuned": ("#C0392B", "-s", "fine-tuned"),
}
FALLBACK = ["#1B7F5A", "#A929B4", "#D98C00", "#444444"]
BASELINE_SUFFIXES = ("_identity", "_permutation", "_identity_map")
RANDOM_SUFFIX = "_random_seeds"


# ------------------------------------------------------------------ loading
def _f(x):
    """float(x) if finite, else nan. Covers None, strings, NaN and inf."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def load(path, include_baselines=False):
    """JSON records -> {pair: {"epochs": [...], "k": [...], arrays (n_ckpt, n_k)}}."""
    with open(path) as fh:
        records = json.load(fh)          # json.load accepts NaN / Infinity

    rows = [r for r in records if "topk" in r and "cknna" in r and not r["pair"].endswith(RANDOM_SUFFIX)]
    if not include_baselines:
        rows = [r for r in rows if not r["pair"].endswith(BASELINE_SUFFIXES)]
    if not rows:
        raise SystemExit(f"no alignment records found in {path}")

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


def load_random_bands(path):
    """JSON records -> {pair: {k: (min, max) of CKNNA over random-init seeds}}."""
    with open(path) as fh:
        records = json.load(fh)
    by_pair = {}
    for r in records:
        if "topk" in r and r.get("pair", "").endswith(RANDOM_SUFFIX):
            pair = r["pair"][: -len(RANDOM_SUFFIX)]
            by_pair.setdefault(pair, {}).setdefault(int(r["topk"]), []).append(_f(r.get("cknna")))
    return {pair: {k: (np.nanmin(v), np.nanmax(v)) for k, v in by_k.items() if np.isfinite(v).any()}
            for pair, by_k in by_pair.items()}


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


def plot_vs_k(data, nulls, kk):
    """CKNNA vs k, mean over trained epochs, min-max band."""
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
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

    lo, hi = np.nanmin(nulls, 0), np.nanmax(nulls, 0)
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(kk[ok], lo[ok], hi[ok], color=C_NUL, alpha=0.30, lw=0, zorder=1)
    ax.plot(kk, np.nanmean(nulls, 0), "-", color=C_NUL, lw=1.0, zorder=2, label="permutation null")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(kk); ax.set_xticklabels([str(v) for v in kk])
    ax.set_xlabel("neighbourhood size $k$"); ax.set_ylabel("CKNNA")
    finite = np.concatenate([d["cknna"][np.isfinite(d["cknna"])].ravel() for d in data.values()]
                            + [nulls[np.isfinite(nulls)].ravel()])
    ax.set_ylim(0.5 * finite.min(), 2.0 * finite.max())
    ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=0.15, which="both", lw=0.4)
    return fig


def make_axis(data):
    """x positions on one timeline: untrained | pre-training epochs | init | fine-tuning epochs."""
    def epochs(finetuned):
        return sorted({int(e) for p, d in data.items() if ("finetuned" in p) == finetuned
                       for e in d["epochs"] if e.isdigit()})

    pre, ft = epochs(False), epochs(True)
    x_init = 1 + len(pre)
    pos = {False: {e: 1 + i for i, e in enumerate(pre)}, True: {e: x_init + 1 + i for i, e in enumerate(ft)}}
    ticks = {0: "untrained", x_init: "init"}
    for finetuned in (False, True):
        ticks.update({x: str(e) for e, x in pos[finetuned].items() if e % 2 == 0})
    return pos, x_init, ticks


def x_of(pair, epoch, pos, x_init):
    """x position of a checkpoint label, or None if it has no place on the timeline."""
    if epoch == "untrained":
        return 0
    if epoch in ("init", "ft_src"):
        return x_init
    if epoch.isdigit():
        return pos["finetuned" in pair].get(int(epoch))
    return None


def plot_vs_epoch(data, nulls, kk, topk, bands):
    """CKNNA vs checkpoint at a fixed k, with per-checkpoint uncertainties."""
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    pos, x_init, ticks = make_axis(data)
    for i, (pair, d) in enumerate(data.items()):
        if topk not in d["k"]:
            continue
        colour, marker, label = style_for(pair, i)
        j = list(d["k"]).index(topk)
        y, e = d["cknna"][:, j], np.where(np.isfinite(d["cknna_std"][:, j]), d["cknna_std"][:, j], 0.0)
        xs = [x_of(pair, ep, pos, x_init) for ep in d["epochs"]]
        line = [n for n, x in enumerate(xs) if x is not None and x != 0]
        ax.errorbar([xs[n] for n in line], y[line], yerr=e[line], fmt=marker, color=colour,
                    ms=4, lw=1.4, capsize=2, label=label)
        # the random initialisation is a separate starting point, not part of the training curve
        for n in (n for n, x in enumerate(xs) if x == 0):
            ax.errorbar([0], [y[n]], yerr=[e[n]], fmt=marker[-1], color=colour, ms=4, capsize=2)
        if topk in bands.get(pair, {}):
            lo, hi = bands[pair][topk]
            ax.axhspan(lo, hi, facecolor="none", edgecolor=colour, hatch="////", alpha=0.35, lw=0,
                       label=f"random init ({label})")

    band = nulls[:, list(kk).index(topk)]
    ax.axhspan(np.nanmin(band), np.nanmax(band), color=C_NUL, alpha=0.30, lw=0, label="permutation null")
    ax.axvline(x_init, color=C_NUL, lw=0.6, ls="--")
    for text, x, ha in (("pre-training", x_init - 0.3, "right"), ("fine-tuning", x_init + 0.3, "left")):
        ax.text(x, 0.98, text, transform=ax.get_xaxis_transform(), ha=ha, va="top", fontsize=6, color=C_NUL)
    xmax = max(list(ticks) + [x for p in pos.values() for x in p.values()])
    ax.set_xticks(list(ticks)); ax.set_xticklabels(list(ticks.values()), rotation=60, ha="right")
    ax.set_xlim(-0.6, xmax + 0.6); ax.set_ylim(0, None)
    ax.set_xlabel("checkpoint (epoch)"); ax.set_ylabel(f"CKNNA at $k = {topk}$")
    ax.legend(frameon=False, loc="best", fontsize=6, ncol=2); ax.grid(alpha=0.15, lw=0.4)
    return fig


def main():
    dir_substr = '50batches_per-epoch'
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?", default=f"analysis/cknna/{dir_substr}/cknna_results.json")
    ap.add_argument("-o", "--out", default=f"fig/{dir_substr}/cknna_alignment",
                    help="output stem; .pdf and .png are written")
    ap.add_argument("--include-baselines", action="store_true",
                    help="also plot the *_identity / *_permutation self-comparisons")
    args = ap.parse_args()

    data = load(args.json, args.include_baselines)
    bands = load_random_bands(args.json)
    nulls = np.vstack([d["null"] for d in data.values()])
    kk = next(iter(data.values()))["k"]

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    save(plot_vs_k(data, nulls, kk), f"{stem}_vs_k")
    for k in kk:
        save(plot_vs_epoch(data, nulls, kk, int(k), bands), f"{stem}_k{int(k)}")

    # ---- numbers for the text
    n = next((d["n"] for d in data.values() if d["n"]), "?")
    print(f"n = {n} per draw;  k = {[int(v) for v in kk]}")
    for pair, d in data.items():
        y, s, nl, ns = d["cknna"], d["cknna_std"], d["null"], d["null_std"]
        sig = (y - nl) / np.sqrt(np.nan_to_num(s) ** 2 + np.nan_to_num(ns) ** 2)
        print(f"\n{pair}  (epochs {', '.join(d['epochs'])})")
        print(f"  CKNNA {np.nanmin(y):.3f}-{np.nanmax(y):.3f}   "
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
            print(f"  k={k:>3}: " + ", ".join("nan" if not np.isfinite(v) else f"{v:.3f}"
                                              for v in col) + f"   [{trend} over epochs]")
    if len(data) == 2:
        a, b = data.values()
        r = np.nanmean(b["cknna"], 0) / np.nanmean(a["cknna"], 0)
        print("\nratio second/first per k: " +
              ", ".join(f"k={k}: {v:.1f}" for k, v in zip(kk, r)))


if __name__ == "__main__":
    main()