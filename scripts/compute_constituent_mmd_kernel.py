"""Constituent-level CKNNA between the NTP and MPM backbones on AspenOpenJets.

Instead of mean-pooling each jet to one vector, every jet is kept as its set of
constituent embeddings (start token stripped), and jets are compared with a
pT^beta-weighted MMD / RBF set kernel (gabbro.metrics.constituent_kernels):

    K(A, B) = sum_i sum_j z_i z'_j exp(-||e_i - e'_j||^2 / (2 sigma^2)),  z_i ~ pT_i^beta

CKNNA is then computed on these jet-by-jet kernels, for every (beta, sigma factor);
sigma is set per model by the median heuristic times the factor. Pairs are the same
as in compute_cknna_mpm_vs_ntp_on_aoj.py (pre-trained and fine-tuned NTP vs MPM, matched
by epoch), plus a random-init baseline with the pre-training architectures and the same
seeds as there.

Run from the repository root; results go to
    analysis/cknna/<MAX_BATCHES>batches_per-epoch_constituent_mmd/
as one combined json and one json per (beta, sigma factor), which plot_cknna.py reads:
    python scripts/plot_cknna.py <dir>/beta1.0_sigmax1.0.json --metric constituent_mmd \
        --run-label <MAX_BATCHES>batches_beta1.0_sigmax1.0
"""

import gc
import json
import sys
import time
from pathlib import Path

# see compute_cknna_mpm_vs_ntp_on_aoj.py: compatible numpy/uproot versions for this kernel
deps_path = "/home/users/w/wozniak/.local/ntp4jets-deps"
if deps_path not in sys.path:
    sys.path.insert(0, deps_path)

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import compute_cknna_mpm_vs_ntp_on_aoj as base
import gabbro.metrics.constituent_kernels as ck

torch.backends.cuda.matmul.allow_tf32 = True  # fast distance matmuls; kernels are summed in float32 anyway

MAX_BATCHES = 20
BETAS = (0.0, 0.5, 1.0, 2.0)  # 0: uniform constituent weights, 1: pT fractions
SIGMA_FACTORS = (0.5, 1.0, 2.0)
TOPKS = (5, 10, 25, 50)
N_SUB, N_DRAWS, N_PERM = 1500, 5, 1
N_RANDOM_SEEDS = 3
FINETUNE_MAX_EPOCH = 15


def checkpoints(run, finetune, pretrain_snapshot=None):
    """Same checkpoint selection as compute_cknna_mpm_vs_ntp_on_aoj.py."""
    run = Path(run)
    if finetune:
        return base.pick_checkpoints_from_model_dir(
            run, max_epoch=FINETUNE_MAX_EPOCH, init_ckpt=run / base.INIT_CKPT_NAME, init_first=True)
    return base.pick_checkpoints_from_model_dir(
        run, max_epoch=base.PRETRAIN_INIT_EPOCH, init_ckpt=run / "checkpoints" / pretrain_snapshot)


def compare(model_a, model_b, batches, pt_getter, n_jets):
    return ck.compare_pair_mmd(
        model_a, model_b, batches, pt_getter, n_jets=n_jets, betas=BETAS, sigma_factors=SIGMA_FACTORS,
        topks=TOPKS, n_sub=N_SUB, n_draws=N_DRAWS, n_perm=N_PERM, device=base.device,
    )


def report(rows):
    for r in rows:
        if r["topk"] == 10:
            print(f"  beta={r['beta']:<4} sigma x{r['sigma_factor']:<4} "
                  f"CKNNA={r['cknna']:.4f}+-{r['cknna_std']:.4f}  null={r['null']:.4f}")


def free():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
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

    # read the jets once, so every model sees the same jets in the same order
    batches = ck.collect_batches(dataloader, n_jets)
    pt_getter = ck.make_pt_getter(cfg.data.dataset_kwargs_common.feature_dict, pt_key="part_pt")

    # sanity check of the pT inversion: values should look like GeV and be pT-ordered
    pt = pt_getter(batches[0]["part_features"])[:5]
    mask = batches[0]["part_mask"][:5].bool()
    print("recovered constituent pT, first 5 jets (leading 5 constituents):")
    for row, m in zip(pt, mask):
        print("   ", [round(v, 2) for v in row[m][:5].tolist()])

    ckpts = {
        "ntp": checkpoints(base.NTP_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[0]),
        "mpm": checkpoints(base.MPM_PRETRAIN_RUN, False, base.PRETRAIN_INIT_NAMES[1]),
        "ntp_finetuned": checkpoints(base.NTP_FINETUNE_RUN, True),
        "mpm_finetuned": checkpoints(base.MPM_FINETUNE_RUN, True),
    }
    pairs = [("ntp_vs_mpm", "ntp", "mpm"), ("ntp_finetuned_vs_mpm_finetuned", "ntp_finetuned", "mpm_finetuned")]

    # ---- MMD-kernel CKNNA per epoch-matched pair; models are loaded two at a time
    results = []
    for pair_name, a, b in pairs:
        ref_a, ref_b = base.first_real_ckpt(ckpts[a]), base.first_real_ckpt(ckpts[b])
        paths_b = {base.epoch_label_from_ckpt_path(p): p for p in ckpts[b]}
        for i, path_a in enumerate(ckpts[a]):
            epoch = base.epoch_label_from_ckpt_path(path_a)
            if epoch not in paths_b:
                continue
            t0 = time.time()
            model_a, model_b = base.load_model(path_a, ref_a), base.load_model(paths_b[epoch], ref_b)
            rows = compare(model_a, model_b, batches, pt_getter, n_jets)
            del model_a, model_b
            free()
            print(f"=== MMD {pair_name} | epoch {epoch} ({time.time() - t0:.0f} s) ===")
            report(rows)
            results += [{"pair": pair_name, "checkpoint_idx": i, "epoch": epoch, **r} for r in rows]

    # ---- random-init baseline, pre-training architectures; seeds as in the pooled script,
    #      so these are the same random networks as its ntp_vs_mpm_random_seeds
    ref_ntp, ref_mpm = base.first_real_ckpt(ckpts["ntp"]), base.first_real_ckpt(ckpts["mpm"])
    for seed in range(N_RANDOM_SEEDS):
        t0 = time.time()
        model_a, model_b = base.random_init_model(ref_ntp, seed), base.random_init_model(ref_mpm, 1000 + seed)
        rows = compare(model_a, model_b, batches, pt_getter, n_jets)
        del model_a, model_b
        free()
        print(f"=== MMD ntp_vs_mpm_random_seeds | seed {seed} ({time.time() - t0:.0f} s) ===")
        report(rows)
        results += [{"pair": "ntp_vs_mpm_random_seeds", "checkpoint_idx": seed, "epoch": f"seed{seed}", **r}
                    for r in rows]

    out = Path("analysis/cknna") / f"{MAX_BATCHES}batches_per-epoch_constituent_mmd"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cknna_mmd_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    for beta in BETAS:
        for factor in SIGMA_FACTORS:
            variant = [r for r in results if r["beta"] == beta and r["sigma_factor"] == factor]
            with open(out / f"beta{beta}_sigmax{factor}.json", "w") as f:
                json.dump(variant, f, indent=2, default=str)
    print(f"saved {len(results)} rows to {out}")
