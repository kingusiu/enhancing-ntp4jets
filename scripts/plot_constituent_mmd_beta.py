"""Constituent-MMD CKNNA vs the pT weighting exponent beta, one panel per k.

Reads the combined output of compute_constituent_mmd_kernel.py, e.g.

    python scripts/plot_constituent_mmd_beta.py \
        analysis/cknna/20batches_per-epoch_constituent_mmd/cknna_mmd_results.json --sigma-factor 1.0

and writes <out-root>/<run-label>_beta_sweep/cknna_vs_beta_sigmax<f>.{pdf,png}. Per beta: random init
(mean, min-max over seeds), pre-trained and fine-tuned (mean, min-max over trained epochs).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_cknna import C_RND, STYLE, save  # same look as the other CKNNA figures

TRAINED = {str(e) for e in range(100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--sigma-factor", type=float, default=1.0)
    ap.add_argument("--out-root", default="analysis/embedding_alignment/constituent_mmd")
    ap.add_argument("--run-label", default=None,
                    help="defaults to the json's parent directory name without '_per-epoch_constituent_mmd'")
    args = ap.parse_args()

    with open(args.json) as fh:
        records = [r for r in json.load(fh) if float(r["sigma_factor"]) == args.sigma_factor]
    betas = sorted({float(r["beta"]) for r in records})
    ks = sorted({int(r["topk"]) for r in records})

    curves = [  # (pair, which checkpoints, colour, marker, label)
        ("ntp_vs_mpm_random_seeds", lambda e: True, C_RND, "--o", "random init"),
        ("ntp_vs_mpm", lambda e: e in TRAINED, *STYLE["ntp_vs_mpm"]),
        ("ntp_finetuned_vs_mpm_finetuned", lambda e: e in TRAINED, *STYLE["ntp_finetuned_vs_mpm_finetuned"]),
    ]
    fig, axes = plt.subplots(1, len(ks), figsize=(2.0 * len(ks), 2.4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), ks):
        for pair, sel, colour, marker, label in curves:
            vals = [[r["cknna"] for r in records if r["pair"] == pair and float(r["beta"]) == b
                     and int(r["topk"]) == k and sel(str(r["epoch"]))] for b in betas]
            mean = np.array([np.mean(v) for v in vals])
            ax.fill_between(betas, [min(v) for v in vals], [max(v) for v in vals], color=colour, alpha=0.18, lw=0)
            ax.plot(betas, mean, marker, color=colour, ms=3.5, lw=1.3, label=label,
                    mfc="white" if pair.endswith("random_seeds") else colour)
        ax.set_title(f"$k = {k}$", fontsize=8)
        ax.set_xticks(betas); ax.set_xlabel(r"pT weight exponent $\beta$")
        ax.grid(alpha=0.15, lw=0.4)
    # the axes share y, so set the limit once all panels are drawn
    np.atleast_1d(axes)[0].set_ylim(0, None)
    np.atleast_1d(axes)[0].set_ylabel(f"constituent-MMD CKNNA ($\\sigma \\times {args.sigma_factor:g}$)")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5 * len(ks), -0.25),
                                  ncol=3, fontsize=6)

    run_label = args.run_label or Path(args.json).resolve().parent.name.replace("_per-epoch_constituent_mmd", "")
    out = Path(args.out_root) / f"{run_label}_beta_sweep"
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out / f"cknna_vs_beta_sigmax{args.sigma_factor:g}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
