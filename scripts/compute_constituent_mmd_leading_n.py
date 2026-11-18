"""Constituent-MMD CKNNA on the N leading-pT constituents only, swept over N.

Tests whether the NTP / MPM alignment comes from the hardest constituents: if the models
agree on the leading particles and diverge on the soft ones, alignment should be high for
small N and fall as softer constituents are added. With BETA = 0 the constituents are only
selected by pT and then weighted equally; BETA = 1 additionally weights them by pT. Two ways of cutting
(CUTS), by default only the embedding cut is run:

    input      the backbone only sees the N leading constituents of each jet
    embedding  the backbone sees the full jet; only the embeddings of the N leading
               constituents enter the kernel

Everything else as in compute_constituent_mmd_kernel.py (same pairs, checkpoints and
random-init seeds), at a single beta and sigma factor. Run from the repository root;
results go to analysis/cknna/<MAX_BATCHES>batches_per-epoch_constituent_mmd_leading_n<Nmin>to<Nmax>_beta<BETA>/
as cknna_mmd_leading_n_results.json and one plot_cknna.py-readable json per (cut, N):
<cut>/n<N>.json. Plot with scripts/plot_constituent_mmd_leading_n.py.
"""

import argparse
import json
import time
from pathlib import Path

import compute_constituent_mmd_kernel as mmd  # sets up sys.path / project root on import
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import compute_cknna_mpm_vs_ntp_on_aoj as base
import gabbro.metrics.constituent_kernels as ck

MAX_BATCHES = 20
N_LEADING = (1, 2, 3, 4, 5, 7, 10, 20, 46, 70)  # finer steps where alignment changes fastest
# "input" feeds the backbone truncated jets it never saw in training, so it tests behaviour on
# out-of-distribution inputs rather than the representation of real jets; add it back if wanted
CUTS = ("embedding",)
BETA = 0.0  # uniform weights: the N leading constituents are selected by pT, then count equally
SIGMA_FACTOR = 1.0
# random-init baselines, same weights for every N: (pair name, causal-mask override for NTP).
# ("ntp_bidir_vs_mpm_random_seeds", False) would compare two random draws of what is then the same
# architecture, i.e. seed-to-seed variation rather than NTP vs MPM, so it is not run
RANDOM_PAIRS = (("ntp_vs_mpm_random_seeds", None),)


def compare_all_n(model_a, model_b, batches, cut_batches, pt_getter, n_jets):
    """Rows for every (cut, N) for one pair of models; full-jet inference is shared by all N."""
    rows = []
    full = [ck.constituent_inference(batches, m, pt_getter, n_jets, base.device) for m in (model_a, model_b)]
    for cut in CUTS:
        for n in N_LEADING:
            if cut == "embedding":
                inf = [ck.leading_constituents(x, n) for x in full]
            else:
                inf = [ck.constituent_inference(cut_batches[n], m, pt_getter, n_jets, base.device)
                       for m in (model_a, model_b)]
            for r in ck.compare_constituents_mmd(
                    *inf, n_jets=n_jets, betas=(BETA,), sigma_factors=(SIGMA_FACTOR,), topks=mmd.TOPKS,
                    n_sub=mmd.N_SUB, n_draws=mmd.N_DRAWS, n_perm=mmd.N_PERM, device=base.device):
                rows.append({"cut": cut, "n_leading": n, **r})
    return rows


def report(rows):
    for cut in CUTS:
        vals = [f"{r['n_leading']}:{r['cknna']:.3f}" for r in rows if r["cut"] == cut and r["topk"] == 10]
        print(f"  {cut:>9} k=10  " + "  ".join(vals))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-only", action="store_true",
                    help="only (re)compute the random-init pairs and merge them into the existing results json")
    args = ap.parse_args()

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "experiment=example_experiment_backbone_and_head_allfeat_ntp_classifier_head",
            "feature_dict=feature_dict_all_without_cuts"])
    bs = cfg.data.batch_size.train
    n_jets = MAX_BATCHES * bs
    dataloader = base.get_aoj_dataloader(
        cfg, "/srv/beegfs/scratch/groups/rodem/lumea/datasets/aspenJets/RunG_batch0.h5",
        n_jets_per_file=(MAX_BATCHES + 1) * bs)
    batches = ck.collect_batches(dataloader, n_jets)
    cut_batches = {n: ck.cut_batches_to_leading(batches, n) for n in N_LEADING} if "input" in CUTS else {}
    pt_getter = ck.make_pt_getter(cfg.data.dataset_kwargs_common.feature_dict, pt_key="part_pt")

    # "leading" means the first constituents in input order, so check that this is descending pT
    n_sorted, n_total, mult = 0, 0, []
    for b in batches:
        pt, mask = pt_getter(b["part_features"]), b["part_mask"].bool()
        for row, m in zip(pt, mask):
            p = row[m]
            n_sorted += bool(torch.all(p[1:] <= p[:-1] + 1e-3))
            n_total += 1
            mult.append(int(m.sum()))
    mult = torch.tensor(mult, dtype=torch.float)
    print(f"jets with descending-pT constituent order: {n_sorted} of {n_total}")
    print(f"multiplicity: mean {mult.mean():.1f}, median {mult.median():.0f}, min {mult.min():.0f}, "
          f"jets with fewer than {max(N_LEADING)} constituents: {int((mult < max(N_LEADING)).sum())}")

    ckpts = {
        "ntp": mmd.checkpoints(base.NTP_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[0]),
        "mpm": mmd.checkpoints(base.MPM_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[1]),
        "ntp_finetuned": mmd.checkpoints(base.NTP_FINETUNE_RUN, True),
        "mpm_finetuned": mmd.checkpoints(base.MPM_FINETUNE_RUN, True),
    }
    pairs = [("ntp_vs_mpm", "ntp", "mpm"), ("ntp_finetuned_vs_mpm_finetuned", "ntp_finetuned", "mpm_finetuned")]

    results = []
    for pair_name, a, b in ([] if args.random_only else pairs):
        ref_a, ref_b = base.first_real_ckpt(ckpts[a]), base.first_real_ckpt(ckpts[b])
        paths_b = {base.epoch_label_from_ckpt_path(p): p for p in ckpts[b]}
        for i, path_a in enumerate(ckpts[a]):
            epoch = base.epoch_label_from_ckpt_path(path_a)
            if epoch not in paths_b:
                continue
            t0 = time.time()
            model_a, model_b = base.load_model(path_a, ref_a), base.load_model(paths_b[epoch], ref_b)
            rows = compare_all_n(model_a, model_b, batches, cut_batches, pt_getter, n_jets)
            del model_a, model_b
            mmd.free()
            print(f"=== {pair_name} | epoch {epoch} ({time.time() - t0:.0f} s) ===")
            report(rows)
            results += [{"pair": pair_name, "checkpoint_idx": i, "epoch": epoch, **r} for r in rows]

    ref_ntp, ref_mpm = base.first_real_ckpt(ckpts["ntp"]), base.first_real_ckpt(ckpts["mpm"])
    for pair_name, causal in RANDOM_PAIRS:
        for seed in range(mmd.N_RANDOM_SEEDS):
            t0 = time.time()
            model_a = base.random_init_model(ref_ntp, seed, causal=causal)
            model_b = base.random_init_model(ref_mpm, 1000 + seed)
            rows = compare_all_n(model_a, model_b, batches, cut_batches, pt_getter, n_jets)
            del model_a, model_b
            mmd.free()
            print(f"=== {pair_name} | seed {seed} ({time.time() - t0:.0f} s) ===")
            report(rows)
            results += [{"pair": pair_name, "checkpoint_idx": seed, "epoch": f"seed{seed}", **r} for r in rows]

    # the N range is part of the name, so sweeps over different N do not overwrite each other
    out = Path("analysis/cknna") / (f"{MAX_BATCHES}batches_per-epoch_constituent_mmd_leading_n"
                                    f"{min(N_LEADING)}to{max(N_LEADING)}_beta{BETA}")
    out.mkdir(parents=True, exist_ok=True)
    if args.random_only:
        # keep everything except the records just recomputed (same random pair and cut)
        with open(out / "cknna_mmd_leading_n_results.json") as f:
            old = json.load(f)
        redone = {(name, cut) for name, _ in RANDOM_PAIRS for cut in CUTS}
        results = [r for r in old if (r["pair"], r["cut"]) not in redone] + results
    with open(out / "cknna_mmd_leading_n_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    for cut in CUTS:
        (out / cut).mkdir(exist_ok=True)
        for n in N_LEADING:
            with open(out / cut / f"n{n}.json", "w") as f:
                json.dump([r for r in results if r["cut"] == cut and r["n_leading"] == n], f, indent=2, default=str)
    print(f"saved {len(results)} rows to {out}")
