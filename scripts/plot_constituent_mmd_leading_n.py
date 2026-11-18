"""Figures for compute_constituent_mmd_leading_n.py: constituent-MMD CKNNA on the N leading constituents.

    python scripts/plot_constituent_mmd_leading_n.py \
        analysis/cknna/10batches_per-epoch_constituent_mmd_leading_n/cknna_mmd_leading_n_results.json \
        --full analysis/cknna/10batches_per-epoch_constituent_mmd/beta1.0_sigmax1.0.json

writes to <out-root>/<run-label>/ (default analysis/embedding_alignment/constituent_mmd/10batches_leading_n):

    <cut>/cknna_vs_epoch_by_n_k<k>.*  pre-trained | fine-tuned vs epoch, one line per N; the
                                      all-constituents run (--full, same beta and sigma) dashed red
    cknna_vs_n_k<k>.*                 one panel per cut vs N: random init (mean, min-max over
                                      seeds), pre-trained and fine-tuned (mean, min-max over trained
                                      epochs); the last point ("all") is the --full run
    <cut>/summary_k<k>.md             the same numbers as a table, with ratios to random init and to
                                      the fine-tuning start

Only the embedding cut is plotted by default: the input cut feeds the backbone truncated jets it
never saw in training (--cuts input embedding to include it).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_cknna import C_RND, STYLE, save, x_of

PAIRS = ("ntp_vs_mpm", "ntp_finetuned_vs_mpm_finetuned")
RANDOM = "ntp_vs_mpm_random_seeds"
RANDOM_BIDIR = "ntp_bidir_vs_mpm_random_seeds"   # NTP without its causal mask: same attention type as MPM
C_ALL = "#D7263D"  # bright red: the all-constituents reference stands out from the viridis N lines


def series(records, pair, k, **match):
    """{epoch label: cknna} for one pair at one k, restricted to records matching `match`."""
    return {str(r["epoch"]): r["cknna"] for r in records
            if r["pair"] == pair and int(r["topk"]) == k and all(r.get(f) == v for f, v in match.items())}


def trained(s):
    return [v for e, v in s.items() if e.isdigit()]


def plot_vs_epoch(records, full, cut, ns, k, beta):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(ns)))
    for ax, pair in zip(axes, PAIRS):
        for colour, n in zip(colours, ns):
            s = series(records, pair, k, cut=cut, n_leading=n)
            pts = sorted((x_of(pair, e), v) for e, v in s.items() if x_of(pair, e) is not None)
            ax.plot([x for x, _ in pts], [v for _, v in pts], "-o", color=colour, ms=2.5, lw=1.1, label=f"N = {n}")
        if full:
            s = series(full, pair, k)
            pts = sorted((x_of(pair, e), v) for e, v in s.items() if x_of(pair, e) is not None)
            ax.plot([x for x, _ in pts], [v for _, v in pts], "--", color=C_ALL, lw=1.8, label="all constituents")
        xmax = max(x for line in ax.get_lines() for x in line.get_xdata())
        ticks = [0] + list(range(1, int(xmax) + 1, 2))
        ax.set_xlim(-0.8, xmax + 0.5)
        ax.set_xticks(ticks); ax.set_xticklabels(["start"] + [str(x - 1) for x in ticks[1:]])
        ax.get_xticklabels()[0].set_horizontalalignment("right")
        ax.set_title(STYLE[pair][2], fontsize=8); ax.set_xlabel("epoch"); ax.grid(alpha=0.15, lw=0.4)
    axes[0].set_ylim(0, None)
    axes[0].set_ylabel(f"constituent-MMD CKNNA at $k = {k}$, $\\beta = {beta:g}$\n({cut} cut, N leading constituents)")
    axes[1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def plot_vs_n(records, full, ns, k, beta, cuts):
    fig, axes = plt.subplots(1, len(cuts), figsize=(3.4 * len(cuts), 2.6), sharey=True, squeeze=False)
    axes = axes[0]
    # N values are not evenly spaced, so place them at evenly spaced positions; "all" after a gap
    x_all = len(ns) + 0.5
    curves = [(RANDOM, None, C_RND, "--o", "random init")]
    if any(r["pair"] == RANDOM_BIDIR for r in records):
        curves.append((RANDOM_BIDIR, None, C_RND, ":^", "random init, NTP w/o causal mask"))
    curves += [(p, None, *STYLE[p]) for p in PAIRS]
    for ax, cut in zip(axes, cuts):
        for pair, _, colour, marker, label in curves:
            pick = (lambda s: list(s.values())) if pair in (RANDOM, RANDOM_BIDIR) else trained
            vals = [pick(series(records, pair, k, cut=cut, n_leading=n)) for n in ns]
            xs = list(range(len(ns)))
            if full and pick(series(full, pair, k)):
                vals.append(pick(series(full, pair, k))); xs.append(x_all)
            if not any(vals):
                continue
            xs, vals = [x for x, v in zip(xs, vals) if v], [v for v in vals if v]
            ax.fill_between(xs, [min(v) for v in vals], [max(v) for v in vals], color=colour, alpha=0.18, lw=0)
            ax.plot(xs, [np.mean(v) for v in vals], marker, color=colour, ms=3.5, lw=1.3, label=label,
                    mfc="white" if pair in (RANDOM, RANDOM_BIDIR) else colour)
        ax.set_xticks(list(range(len(ns))) + ([x_all] if full else []))
        ax.set_xticklabels([str(n) for n in ns] + (["all"] if full else []))
        ax.set_title(f"{cut} cut", fontsize=8); ax.set_xlabel("N leading-pT constituents")
        ax.grid(alpha=0.15, lw=0.4)
    axes[0].set_ylim(0, None)
    axes[0].set_ylabel(f"constituent-MMD CKNNA at $k = {k}$, $\\beta = {beta:g}$")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=3)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--full", default=None,
                    help="all-constituents results at the same beta and sigma factor ('' to leave out); default: "
                         "analysis/cknna/<batches>_per-epoch_constituent_mmd/beta<beta>_sigmax<sigma>.json, "
                         "with <batches> (e.g. 20batches) taken from the json's parent directory")
    ap.add_argument("--out-root", default="analysis/embedding_alignment/constituent_mmd")
    ap.add_argument("--run-label", default=None,
                    help="defaults to e.g. 10batches_leading_n, from the json's parent directory")
    ap.add_argument("--cuts", nargs="+", default=["embedding"], choices=["input", "embedding"],
                    help="which cuts to plot (default: embedding only)")
    args = ap.parse_args()

    with open(args.json) as fh:
        records = json.load(fh)
    beta, sigma = float(records[0]["beta"]), float(records[0]["sigma_factor"])
    if args.full is None:
        batches = Path(args.json).resolve().parent.name.split("_")[0]   # same jets as the sweep
        args.full = f"analysis/cknna/{batches}_per-epoch_constituent_mmd/beta{beta}_sigmax{sigma}.json"
    full = None
    if args.full and not Path(args.full).exists():
        print(f"no all-constituents reference at {args.full}, plotting without it")
    if args.full and Path(args.full).exists():
        with open(args.full) as fh:
            full = json.load(fh)
    ns = sorted({int(r["n_leading"]) for r in records})
    ks = sorted({int(r["topk"]) for r in records})
    cuts = [c for c in ("input", "embedding") if c in args.cuts and any(r["cut"] == c for r in records)]

    run_label = args.run_label or Path(args.json).resolve().parent.name.replace(
        "_per-epoch_constituent_mmd_leading_n", "_leading_n")
    out = Path(args.out_root) / run_label
    for cut in cuts:
        (out / cut).mkdir(parents=True, exist_ok=True)
    for k in ks:
        save(plot_vs_n(records, full, ns, k, beta, cuts), out / f"cknna_vs_n_k{k}")
        for cut in cuts:
            save(plot_vs_epoch(records, full, cut, ns, k, beta), out / cut / f"cknna_vs_epoch_by_n_k{k}")
        for cut in cuts:
            (out / cut / f"summary_k{k}.md").write_text(summary_table(records, full, cut, ns, k, beta))
    print(f"saved to {out}")


def summary_table(records, full, cut, ns, k, beta):
    """Markdown table per N: means over trained epochs and seeds, ratios to random init and to the
    fine-tuning start (the pre-trained backbone fine-tuning starts from)."""
    def mean(s):
        return np.mean(s) if s else np.nan

    bidir = any(r["pair"] == RANDOM_BIDIR for r in records)   # columns only if that baseline was computed
    lines = [f"constituent-MMD CKNNA, {cut} cut, k = {k}, beta = {beta:g}; trained curves are means over "
             "trained epochs, random init over seeds; ft start = fine-tuning start (pre-trained backbone)"
             + ("; rnd bidir = random init with NTP's causal mask off" if bidir else "") + "; '-' = not computed", "",
             "| N | random init | " + ("rnd bidir | " if bidir else "") + "pre-trained | pre-trained / random | "
             + ("pre-trained / rnd bidir | " if bidir else "") + "fine-tuned | fine-tuned / random | "
             + ("fine-tuned / rnd bidir | " if bidir else "") + "ft start | fine-tuned / ft start |",
             "|---|---|---|---|---|---|---|---|" + ("---|---|---|" if bidir else "")]
    f = lambda v, p=3: "-" if np.isnan(v) else f"{v:.{p}f}"  # noqa: E731
    rows = [(str(n), dict(cut=cut, n_leading=n), records) for n in ns]
    if full:
        rows.append(("all", {}, full))
    for label, match, recs in rows:
        rnd = mean(list(series(recs, RANDOM, k, **match).values()))
        rnd_bi = mean(list(series(recs, RANDOM_BIDIR, k, **match).values()))
        pre = mean(trained(series(recs, PAIRS[0], k, **match)))
        ft_s = series(recs, PAIRS[1], k, **match)
        ft, start = mean(trained(ft_s)), ft_s.get("init", np.nan)
        cells = [f(rnd)] + ([f(rnd_bi)] if bidir else []) + [f(pre), f(pre / rnd, 2)] \
            + ([f(pre / rnd_bi, 2)] if bidir else []) + [f(ft), f(ft / rnd, 2)] \
            + ([f(ft / rnd_bi, 2)] if bidir else []) + [f(start), f(ft / start, 2)]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
