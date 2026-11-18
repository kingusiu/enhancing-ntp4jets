"""
How much of a model's (normalized) jet neighbourhoods is particle multiplicity alone?

Counterpart of the `*_vs_multiplicity` records of compute_cknna_mpm_vs_ntp_on_aoj.py for the
normalized metric. Those records rank the neighbours of the 1-d feature log(N) by the largest
inner product of the centred feature, which in 1-d picks the most extreme same-sign jets rather
than jets of similar multiplicity, and a 1-d feature cannot be L2-normalised. Here:

    model side:        clamp + L2-normalise (cknna_with_normalized.prepare_normalized), linear
                       kernel -> neighbours by cosine similarity, as in the normalized CKNNA
    multiplicity side: RBF kernel on log(N), neighbours by distance = jets of closest multiplicity

and CKNNA is computed from the two kernels (constituent_kernels.cknna_sweep). As a sanity anchor
the NTP vs MPM random-init pair is also run through the same kernel pipeline.

usage: python scripts/compute_cknna_vs_multiplicity_normalized.py --max-batches 20
"""

import sys

deps_path = "/home/users/w/wozniak/.local/ntp4jets-deps"
if deps_path not in sys.path:
    sys.path.insert(0, deps_path)

import argparse
import gc
import json
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import gabbro.metrics.cknna_with_normalized as cknna_norm
from gabbro.metrics.constituent_kernels import cknna_sweep
from scripts.compute_cknna_mpm_vs_ntp_on_aoj import (
    INIT_CKPT_NAME, MPM_FINETUNE_RUN, MPM_PRETRAIN_RUN, N_RANDOM_SEEDS, NTP_FINETUNE_RUN,
    NTP_PRETRAIN_RUN, PRETRAIN_INIT_NAMES, PRETRAIN_INIT_EPOCH, do_inference_on_aoj,
    epoch_label_from_ckpt_path, first_real_ckpt, get_aoj_dataloader, inference_without_causal_mask,
    jet_multiplicity, load_model, pick_checkpoints_from_model_dir, random_init_embeddings,
)

TOPKS = (5, 10, 25, 50)


def normalized_kernel(U):
    """Linear kernel of clamped, L2-normalised embeddings (unit diagonal: kNN = cosine kNN)."""
    A = cknna_norm.prepare_normalized(U)
    return A @ A.T


def rbf_kernel_1d(x):
    """RBF kernel of a 1-d feature, bandwidth = its standard deviation."""
    d2 = (x[:, None] - x[None, :]) ** 2
    return torch.exp(-d2 / (2 * x.var()))


def ckpt_by_label(run, label, max_epoch, init_ckpt=None):
    paths = pick_checkpoints_from_model_dir(run, max_epoch=max_epoch, init_ckpt=init_ckpt)
    path = next(p for p in paths if epoch_label_from_ckpt_path(p) == label)
    return path, first_real_ckpt(paths)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batches", type=int, default=20, help="AOJ batches per model")
    max_batches = ap.parse_args().max_batches

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg_aoj = compose(
            config_name="train",
            overrides=["experiment=example_experiment_backbone_and_head_allfeat_ntp_classifier_head",
                       "feature_dict=feature_dict_all_without_cuts"],
        )
    aoj_file = "/srv/beegfs/scratch/groups/rodem/lumea/datasets/aspenJets/RunG_batch0.h5"
    dataloader = get_aoj_dataloader(cfg_aoj, aoj_file, n_jets_per_file=(max_batches + 1) * cfg_aoj.data.batch_size.train)

    ####### pooled embeddings: random init (several seeds) and the trained end points

    ntp_src, ntp_ref = ckpt_by_label(NTP_PRETRAIN_RUN, "ft_src", PRETRAIN_INIT_EPOCH,
                                     Path(NTP_PRETRAIN_RUN) / "checkpoints" / PRETRAIN_INIT_NAMES[0])
    mpm_src, mpm_ref = ckpt_by_label(MPM_PRETRAIN_RUN, "ft_src", PRETRAIN_INIT_EPOCH,
                                     Path(MPM_PRETRAIN_RUN) / "checkpoints" / PRETRAIN_INIT_NAMES[1])

    latents = {}  # name -> list of (n_jets, D), one entry per seed (random) or a single entry
    for seed in range(N_RANDOM_SEEDS):
        print(f"random init, seed {seed}")
        latents.setdefault("random/ntp_causal", []).append(
            random_init_embeddings(ntp_ref, seed, dataloader, max_batches))
        # same seed, same weights, attention mask switched off
        latents.setdefault("random/ntp_bidir", []).append(
            random_init_embeddings(ntp_ref, seed, dataloader, max_batches, causal=False))
        latents.setdefault("random/mpm", []).append(
            random_init_embeddings(mpm_ref, 1000 + seed, dataloader, max_batches))

    model = load_model(ntp_src, ntp_ref)
    assert model.backbone.apply_causal_mask, "NTP pretraining snapshot expected to be causal"
    latents["pretrained/ntp_causal"] = [do_inference_on_aoj(dataloader, model, max_batches=max_batches)]
    latents["pretrained/ntp_bidir"] = [inference_without_causal_mask(dataloader, model, max_batches)]
    del model
    model = load_model(mpm_src, mpm_ref)
    latents["pretrained/mpm"] = [do_inference_on_aoj(dataloader, model, max_batches=max_batches)]
    del model

    for name, run in [("finetuned/ntp", NTP_FINETUNE_RUN), ("finetuned/mpm", MPM_FINETUNE_RUN)]:
        path, ref = ckpt_by_label(run, 15, 15, Path(run) / INIT_CKPT_NAME)
        model = load_model(path, ref)
        print(f"{name}: {path.name}, causal={model.backbone.apply_causal_mask}")
        latents[name] = [do_inference_on_aoj(dataloader, model, max_batches=max_batches)]
        del model
    gc.collect(); torch.cuda.empty_cache()

    # drop jets that are non-finite for any model, so all rows stay paired
    keep = torch.ones(len(latents["random/mpm"][0]), dtype=torch.bool)
    for Us in latents.values():
        for U in Us:
            keep &= torch.isfinite(U).all(dim=1)
    print(f"dropping {int((~keep).sum())} of {len(keep)} jets with non-finite embeddings")
    latents = {k: [U[keep].double() for U in Us] for k, Us in latents.items()}

    # log(N) with a tiny jitter to break ties between jets of equal multiplicity
    mult = jet_multiplicity(dataloader, max_batches)[keep]
    g = torch.Generator().manual_seed(0)
    log_n = mult.log() + 1e-3 * torch.randn(len(mult), generator=g, dtype=torch.float64)
    L_mult = rbf_kernel_1d(log_n)

    ####### CKNNA of each model against multiplicity, and the NTP vs MPM anchor

    results = []
    for name, Us in latents.items():
        for i, U in enumerate(Us):
            rows = cknna_sweep(normalized_kernel(U), L_mult, topks=TOPKS)
            for row in rows:
                results.append({"model": name, "seed": i, "reference": "log_multiplicity", **row})
            print(f"{name} seed {i} vs log N: " + ", ".join(f"k={r['topk']}: {r['cknna']:.3f}" for r in rows))

    for a in ("ntp_causal", "ntp_bidir"):
        for i in range(N_RANDOM_SEEDS):
            rows = cknna_sweep(normalized_kernel(latents[f"random/{a}"][i]),
                               normalized_kernel(latents["random/mpm"][i]), topks=TOPKS)
            for row in rows:
                results.append({"model": f"random/{a}", "seed": i, "reference": "random/mpm", **row})
            print(f"random {a} vs mpm seed {i}: " + ", ".join(f"k={r['topk']}: {r['cknna']:.3f}" for r in rows))

    ####### summary: mean (min-max) over seeds

    out_dir = Path("analysis/embedding_alignment/cknna_pooled_normalized") / \
        f"{max_batches}batches_per-epoch_no-start-token" / "vs_multiplicity"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cknna_vs_multiplicity.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [f"normalized model kernel vs RBF kernel on log(N); {max_batches} AOJ batches, "
             f"n = {results[0]['n']} per draw, mean (min-max) over seeds"]
    for model_name in latents:
        for ref in ("log_multiplicity", "random/mpm"):
            sel = [r for r in results if r["model"] == model_name and r["reference"] == ref]
            if not sel:
                continue
            parts = []
            for k in TOPKS:
                v = torch.tensor([r["cknna"] for r in sel if r["topk"] == k])
                null = sum(r["null"] for r in sel if r["topk"] == k) / len(v)
                parts.append(f"k={k}: {v.mean():.3f} ({v.min():.3f}-{v.max():.3f}) [null {null:.3f}]")
            lines.append(f"  {model_name:22s} vs {ref:16s} " + ", ".join(parts))
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
