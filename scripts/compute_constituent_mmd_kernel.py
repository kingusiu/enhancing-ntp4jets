# Paste into the __main__ block of the CKNNA script, after the four checkpoint lists
# (ntp_models, mpm_models, ntp_finetuned_models, mpm_finetuned_models) and their epoch
# labels exist. Requires constituent_kernels.py on the PYTHONPATH.

import json
from pathlib import Path

import torch

import constituent_kernels as ck

torch.backends.cuda.matmul.allow_tf32 = True  # fast distance matmuls; kernels are summed in float32 anyway

N_JETS = 2000           # jets with constituent-level kernels; CKNNA subsamples 1500 of them per draw
BETAS = (0.0, 0.5, 1.0, 2.0)
SIGMA_FACTORS = (0.5, 1.0, 2.0)

# ---- 0. read the jets once, so every model sees the same jets in the same order
batches = ck.collect_batches(dataloader_aoj, N_JETS)
pt_getter = ck.make_pt_getter(cfg_aoj.data.dataset_kwargs_common.feature_dict, pt_key="part_pt")

# sanity check of the pT inversion: values should look like GeV and be pT-ordered
_pt = pt_getter(batches[0]["part_features"])[:5]
_m = batches[0]["part_mask"][:5].bool()
print("recovered constituent pT, first 5 jets (leading 5 constituents):")
for row, m in zip(_pt, _m):
    print("   ", [round(v, 2) for v in row[m][:5].tolist()])

# ---- 1. weight diagnostics for the 'untrained' points
print("untrained NTP vs untrained MPM:", ck.compare_weights(ntp_models[0], mpm_models[0]))
print("finetune-NTP 'untrained' vs pretrained NTP last:", ck.compare_weights(ntp_finetuned_models[0], ntp_models[-1]))
print("finetune-NTP 'untrained' vs pretrained NTP untrained:", ck.compare_weights(ntp_finetuned_models[0], ntp_models[0]))

# ---- 2. MMD/RBF CKNNA per epoch-matched pair, sweeping beta and sigma
pairs = [
    ("ntp_vs_mpm", ntp_models, mpm_models, ntp_epochs, mpm_epochs),
    ("ntp_finetuned_vs_mpm_finetuned", ntp_finetuned_models, mpm_finetuned_models,
     ntp_finetuned_epochs, mpm_finetuned_epochs),
]
mmd_results = []
for pair_name, models_a, models_b, epochs_a, epochs_b in pairs:
    idx_b = {e: j for j, e in enumerate(epochs_b)}
    for i, epoch in enumerate(epochs_a):
        if epoch not in idx_b:
            continue
        print(f"=== MMD {pair_name} | epoch {epoch} ===")
        rows = ck.compare_pair_mmd(
            models_a[i], models_b[idx_b[epoch]], batches, pt_getter,
            n_jets=N_JETS, betas=BETAS, sigma_factors=SIGMA_FACTORS, device=device,
        )
        for r in rows:
            mmd_results.append({"pair": pair_name, "epoch": epoch, **r})
            if r["topk"] == 10:
                print(f"  beta={r['beta']:<4} sigma x{r['sigma_factor']:<4} "
                      f"CKNNA={r['cknna']:.4f}+-{r['cknna_std']:.4f}  null={r['null']:.4f}")

# ---- 3. pooled reference without the start token, through the same reference CKNNA,
#         plus two confound checks: does the embedding norm track multiplicity, and how
#         well does multiplicity ALONE align with each model?
mult = torch.cat([b["part_mask"].sum(1) for b in batches])[:N_JETS].double()
g = torch.Generator().manual_seed(0)
mult_feat = (mult.log() + 1e-3 * torch.randn(len(mult), generator=g, dtype=torch.float64))[:, None]  # jitter breaks ties
K_mult = ck.linear_kernel(mult_feat - mult_feat.mean(), normalize=False)

for pair_name, models_a, models_b, epochs_a, epochs_b in pairs:
    idx_b = {e: j for j, e in enumerate(epochs_b)}
    for i, epoch in enumerate(epochs_a):
        if epoch not in idx_b:
            continue
        U = ck.pooled_inference(batches, models_a[i], N_JETS, device)
        V = ck.pooled_inference(batches, models_b[idx_b[epoch]], N_JETS, device)
        finite = torch.isfinite(U).all(1) & torch.isfinite(V).all(1)   # same jets dropped from both
        if (~finite).any():
            print(f"  dropping {int((~finite).sum())} jets with non-finite pooled embeddings")
        U, V = U[finite], V[finite]
        Km = K_mult[finite][:, finite]
        for name, X in (("a", U), ("b", V)):
            norm = (X - X.mean(0)).norm(dim=1).double()
            corr = torch.corrcoef(torch.stack([norm, mult[finite]]))[0, 1].item()
            for r in ck.cknna_sweep(ck.linear_kernel(X - X.mean(0), False), Km, topks=(10, 50)):
                mmd_results.append({"pair": f"{pair_name}_model_{name}_vs_multiplicity", "epoch": epoch,
                                    "norm_mult_corr": corr, **r})
        for normalize in (True, False):
            for r in ck.cknna_sweep(ck.linear_kernel(U, normalize), ck.linear_kernel(V, normalize)):
                mmd_results.append({"pair": pair_name, "epoch": epoch, "beta": "pooled",
                                    "sigma_factor": f"linear_norm={normalize}", **r})

out = Path("analysis/cknna") / "constituent_mmd"
out.mkdir(parents=True, exist_ok=True)
with open(out / "cknna_mmd_results.json", "w") as f:
    json.dump(mmd_results, f, indent=2, default=str)
print(f"saved {len(mmd_results)} rows to {out / 'cknna_mmd_results.json'}")