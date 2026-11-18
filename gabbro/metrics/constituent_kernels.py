"""
Constituent-level jet kernels and a reference CKNNA that works on any precomputed kernel.

Contents
--------
1. Reference CKNNA on kernel matrices (for checking gabbro.metrics.cknna and for the
   MMD kernels, which have no feature vectors).
2. pT^beta-weighted MMD / RBF set kernel between jets, computed blockwise on the GPU,
   with a sweep over beta (pT weighting) and sigma (RBF bandwidth) sharing one
   distance computation.
3. Inference helpers: cache the dataloader once, extract per-constituent embeddings
   (start token stripped) together with the constituent pT.
4. Diagnostics: weight comparison between checkpoints, pooled embeddings without the
   start token.

Kernel definition
-----------------
For jets A, B with constituent embeddings e_i, e'_j and weights
    z_i = pT_i^beta / sum_k pT_k^beta        (beta = 0: uniform, beta = 1: pT fractions)
    K(A, B) = sum_i sum_j z_i z'_j exp(-||e_i - e'_j||^2 / (2 sigma^2))
This is <mu_A, mu_B> for the weighted kernel mean embedding mu_A = sum_i z_i phi(e_i),
so K is positive semidefinite for every beta and sigma. The induced distance
K(A,A) + K(B,B) - 2 K(A,B) is the squared MMD between the two jets.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch


# =============================================================================
# 1. Reference CKNNA on kernels
# =============================================================================

def _center(A: torch.Tensor) -> torch.Tensor:
    """H A H with H = I - 11^T/n, without building H."""
    return A - A.mean(0, keepdim=True) - A.mean(1, keepdim=True) + A.mean()


def _hsic(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Biased HSIC up to the constant 1/(n-1)^2, which cancels in CKNNA.

    Uses the Frobenius form <HAH, HBH> = <HAH, B>. For symmetric B this equals the
    usual trace form tr(A H B H). The kNN masks make A and B non-symmetric, and then
    only the Frobenius form guarantees _hsic(A, A) >= 0, so the square root in the
    CKNNA denominator can never be taken of a negative number.
    """
    return (_center(A) * B).sum()


def knn_mask_from_kernel(K: torch.Tensor, k: int) -> torch.Tensor:
    """Row-wise k-nearest-neighbour mask (self excluded), with distances induced by K.

    d^2(i, j) = K_ii + K_jj - 2 K_ij. For a linear kernel on raw features this is the
    squared Euclidean distance; for the MMD kernel it is the squared MMD. Taking the
    top-k of K itself instead is only equivalent when K has a constant diagonal
    (e.g. L2-normalised features); otherwise it prefers large-norm "hub" points.
    """
    d = K.diagonal()
    D2 = d[:, None] + d[None, :] - 2.0 * K
    D2 = D2.clone()
    D2.fill_diagonal_(float("inf"))
    idx = D2.topk(k, dim=1, largest=False).indices
    M = torch.zeros_like(K)
    M.scatter_(1, idx, 1.0)
    return M


def cknna_from_kernels(K: torch.Tensor, L: torch.Tensor, k: int) -> float:
    """CKNNA between two kernel matrices over the same samples, in the same order.

    Numerator: alignment restricted to pairs that are k-nearest neighbours in BOTH
    spaces. Denominators: each kernel's alignment with itself over its OWN kNN pairs
    (as in the platonic-rep reference code). With the joint mask in the denominators
    as well, any accidental overlap would score close to 1, and a permutation null
    would not sit near k/n.
    """
    K = K.double()
    L = L.double()
    mK = knn_mask_from_kernel(K, k)
    mL = knn_mask_from_kernel(L, k)
    m = mK * mL
    kl = _hsic(m * K, m * L)
    kk = _hsic(mK * K, mK * K)
    ll = _hsic(mL * L, mL * L)
    return (kl / torch.sqrt(kk * ll)).item()


def linear_kernel(X: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Inner-product kernel of feature rows, optionally after L2 normalisation."""
    X = X.double()
    if normalize:
        X = torch.nn.functional.normalize(X, dim=1)
    return X @ X.T


def cknna_sweep(
    K: torch.Tensor,
    L: torch.Tensor,
    topks: Sequence[int] = (5, 10, 25, 50),
    n_sub: int | None = 1500,
    n_draws: int = 5,
    n_perm: int = 5,
    seed: int = 0,
) -> list[dict]:
    """CKNNA with subsampling uncertainty and a permutation null, from kernels.

    Each draw takes the same random subset of samples for K and L. The null permutes
    the samples of L within the subset, which breaks the correspondence but keeps
    each kernel's own structure.
    """
    n = K.shape[0]
    assert L.shape[0] == n, "K and L must cover the same samples"
    g = torch.Generator().manual_seed(seed)
    n_sub = n if n_sub is None else min(n_sub, n)

    vals = {k: [] for k in topks}
    nulls = {k: [] for k in topks}
    for _ in range(n_draws):
        idx = torch.randperm(n, generator=g)[:n_sub] if n_sub < n else torch.arange(n)
        Ks, Ls = K[idx][:, idx], L[idx][:, idx]
        perms = [torch.randperm(n_sub, generator=g) for _ in range(n_perm)]
        for k in topks:
            vals[k].append(cknna_from_kernels(Ks, Ls, k))
            for p in perms:
                nulls[k].append(cknna_from_kernels(Ks, Ls[p][:, p], k))

    rows = []
    for k in topks:
        v = torch.tensor(vals[k])
        nl = torch.tensor(nulls[k])
        null_std = nl.std().item() if len(nl) > 1 else float("nan")
        rows.append({
            "topk": k,
            "n": n_sub,
            "cknna": v.mean().item(),
            "cknna_std": v.std().item() if len(v) > 1 else 0.0,
            "null": nl.mean().item(),
            "null_std": null_std,
            # report NaN, never +-inf, when the null spread is zero or undefined
            "z": (v.mean().item() - nl.mean().item()) / null_std
            if null_std == null_std and null_std > 0 else float("nan"),
        })
    return rows


# =============================================================================
# 2. pT-weighted MMD / RBF kernels between jets
# =============================================================================

def jet_weights(pt: torch.Tensor, jet_idx: torch.Tensor, n_jets: int, beta: float) -> torch.Tensor:
    """Per-constituent weights z_i = pT_i^beta / sum_{k in jet} pT_k^beta (flat arrays).

    Computed in log space so large beta and wide pT ranges do not overflow.
    beta = 0 gives uniform weights 1/N.
    """
    logw = beta * torch.log(pt.double().clamp_min(1e-12))
    jet_max = torch.full((n_jets,), -float("inf"), dtype=logw.dtype, device=pt.device)
    jet_max = jet_max.scatter_reduce(0, jet_idx, logw, reduce="amax")
    w = torch.exp(logw - jet_max[jet_idx])
    norm = torch.zeros(n_jets, dtype=w.dtype, device=pt.device).index_add_(0, jet_idx, w)
    return (w / norm[jet_idx]).float()


def median_heuristic_sigma(E: torch.Tensor, n_sample: int = 4000, seed: int = 0) -> float:
    """Median pairwise Euclidean distance between (a random subset of) constituent embeddings."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(E.shape[0], generator=g)[: min(n_sample, E.shape[0])]
    X = E[idx].double()
    sq = (X * X).sum(1)
    D2 = (sq[:, None] + sq[None, :] - 2 * X @ X.T).clamp_min(0)
    iu = torch.triu_indices(len(idx), len(idx), offset=1)
    return D2[iu[0], iu[1]].sqrt().median().item()


@torch.no_grad()
def mmd_kernels(
    E: torch.Tensor,
    jet_idx: torch.Tensor,
    pt: torch.Tensor,
    n_jets: int,
    betas: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    sigmas: Sequence[float] = (1.0,),
    block_jets: int = 128,
    device: torch.device | str = "cpu",
) -> dict[tuple[float, float], torch.Tensor]:
    """Jet-by-jet kernels K[(beta, sigma)] (n_jets x n_jets, float64, on CPU).

    Inputs are flat over constituents and must be grouped by jet (all constituents of
    jet 0, then jet 1, ...), which is what `constituent_inference` returns.
        E       (M, D) constituent embeddings
        jet_idx (M,)   jet index of each constituent, non-decreasing, 0..n_jets-1
        pt      (M,)   constituent pT (any overall scale; it cancels in the weights)

    Cost is dominated by one M x M squared-distance computation, done blockwise and
    shared by every (beta, sigma) combination.
    """
    assert torch.all(jet_idx[1:] >= jet_idx[:-1]), "constituents must be grouped by jet"
    counts = torch.bincount(jet_idx, minlength=n_jets)
    assert torch.all(counts > 0), "every jet needs at least one constituent"
    offsets = torch.zeros(n_jets + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, 0)

    E = E.float().to(device)
    jet_idx = jet_idx.to(device)
    W = {b: jet_weights(pt.to(device), jet_idx, n_jets, b) for b in betas}
    sqn = (E * E).sum(1)

    out = {(b, s): torch.zeros(n_jets, n_jets, dtype=torch.float64) for b in betas for s in sigmas}
    starts = list(range(0, n_jets, block_jets))
    for bi, i0 in enumerate(starts):
        i1 = min(i0 + block_jets, n_jets)
        a0, a1 = offsets[i0].item(), offsets[i1].item()
        loc_i = jet_idx[a0:a1] - i0
        for j0 in starts[bi:]:
            j1 = min(j0 + block_jets, n_jets)
            c0, c1 = offsets[j0].item(), offsets[j1].item()
            loc_j = jet_idx[c0:c1] - j0

            D2 = (sqn[a0:a1, None] + sqn[None, c0:c1] - 2.0 * E[a0:a1] @ E[c0:c1].T).clamp_min_(0)
            for s in sigmas:
                R = torch.exp(D2 * (-0.5 / s**2))
                for b in betas:
                    wi, wj = W[b][a0:a1], W[b][c0:c1]
                    # sum over constituents of jet j (columns), then of jet i (rows)
                    T = torch.zeros(a1 - a0, j1 - j0, device=device).index_add_(1, loc_j, R * wj[None, :])
                    blk = torch.zeros(i1 - i0, j1 - j0, device=device).index_add_(0, loc_i, T * wi[:, None])
                    blk = blk.double().cpu()
                    out[(b, s)][i0:i1, j0:j1] = blk
                    if j0 != i0:
                        out[(b, s)][j0:j1, i0:i1] = blk.T
    return out


# =============================================================================
# 3. Data and inference helpers
# =============================================================================

def collect_batches(dataloader, n_jets: int) -> list[dict]:
    """Read the first batches covering n_jets once and keep them on the CPU.

    Every model then sees exactly the same jets in the same order, and the dataset is
    not re-read (or re-shuffled by workers) once per checkpoint.
    """
    batches, seen = [], 0
    for batch in dataloader:
        batches.append({k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in batch.items()})
        seen += batch["part_features"].shape[0]
        if seen >= n_jets:
            break
    if seen < n_jets:
        raise ValueError(f"dataloader yielded only {seen} jets, {n_jets} requested")
    n_bad = 0
    for b in batches:
        nonfinite = ~torch.isfinite(b["part_features"]).all(-1)          # (B, N)
        n_bad += int((nonfinite & b["part_mask"].bool()).any(-1).sum())
    print(f"collect_batches: {seen} jets, {n_bad} with non-finite input features on real constituents")
    return batches


def make_pt_getter(feature_dict, pt_key: str = "part_pt") -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function part_features (B, N, F) -> pT (B, N), inverting the preprocessing.

    ASSUMPTIONS - check them against your gabbro version:
      * the columns of part_features follow the order of the feature_dict keys;
      * the forward transform is x' = (func(x) - subtract_by) * multiply_by.
    Only pT ratios within a jet matter: an overall scale cancels in the weights, so
    for func = log the subtract_by value is irrelevant, and a wrong multiply_by only
    rescales beta. For func = signed_log (sign(x) * log1p(|x| + eps), see
    gabbro.utils.arrays) the inverse is expm1, which is not scale-covariant, so there
    subtract_by and multiply_by must be right.
    """
    names = list(feature_dict.keys())
    col = names.index(pt_key)
    spec = feature_dict[pt_key] or {}
    mult = float(spec.get("multiply_by", 1.0) or 1.0)
    sub = float(spec.get("subtract_by", 0.0) or 0.0)
    func = spec.get("func", None)

    def pt_getter(part_features: torch.Tensor) -> torch.Tensor:
        y = part_features[..., col].float() / mult + sub
        if func in (None, "None", "null"):
            return y
        if func in ("np.log", "log", "torch.log", "numpy.log"):
            return torch.exp(y)
        if func == "signed_log":  # inverse of gabbro.utils.arrays.signed_log, eps = 1e-5
            return torch.sign(y) * (torch.expm1(y.abs()) - 1e-5)
        raise ValueError(f"no inverse implemented for func={func!r} of {pt_key}; add it in make_pt_getter")

    return pt_getter


def _uses_start_token(model) -> bool:
    return getattr(model.hparams, "start_token_type", None) in ["trainable", "zero_pad"]


def _prepare_inputs(model, part_features, part_mask):
    """Truncate and prepend the start token exactly as in do_inference_on_aoj.

    Returns (inputs, mask_for_backbone, truncated_features, truncated_mask, n_prefix).
    """
    if not _uses_start_token(model):
        return part_features, part_mask, part_features, part_mask, 0
    B = part_features.shape[0]
    max_len = model.hparams.backbone_cfg.max_sequence_len
    pf, pm = part_features[:, : max_len - 1], part_mask[:, : max_len - 1]
    start = model.start_token_continuous.unsqueeze(0).repeat(B, 1, 1)
    pf_in = torch.cat([start, pf], dim=1)
    pm_in = torch.cat([torch.ones(B, 1, dtype=pm.dtype, device=pm.device), pm], dim=1)
    return pf_in, pm_in, pf, pm, 1


@torch.no_grad()
def constituent_inference(batches, model, pt_getter, n_jets: int, device):
    """Flat per-constituent embeddings (start token removed) for the first n_jets jets.

    Returns E (M, D) float32, jet_idx (M,) long, pt (M,) float32 - all on the CPU and
    grouped by jet, ready for `mmd_kernels` - and bad_jets: indices of jets with any
    non-finite embedding or pT. Their constituents are replaced by finite placeholders
    so the kernel computation stays finite; drop those jets before computing CKNNA.
    """
    model.eval()
    embs, jids, pts = [], [], []
    seen = 0
    for batch in batches:
        pf = batch["part_features"].to(device)
        pm = batch["part_mask"].to(device).int()
        pf_in, pm_in, pf_tr, pm_tr, n_prefix = _prepare_inputs(model, pf, pm)
        latent = model.backbone(pf_in, mask=pm_in)[:, n_prefix:]
        pt = pt_getter(pf_tr)

        take = min(pf.shape[0], n_jets - seen)
        valid = pm_tr[:take].bool()
        jet_idx = torch.arange(seen, seen + take, device=device)[:, None].expand(take, valid.shape[1])
        embs.append(latent[:take][valid].float().cpu())
        jids.append(jet_idx[valid].cpu())
        pts.append(pt[:take][valid].float().cpu())
        seen += take
        if seen >= n_jets:
            break

    E, jet_idx, pt = torch.cat(embs), torch.cat(jids), torch.cat(pts)
    n_nonpos = int((pt <= 0).sum())
    if n_nonpos:
        print(f"WARNING: {n_nonpos} constituents with pT <= 0 after inverting the preprocessing")
    bad_c = ~torch.isfinite(E).all(1) | ~torch.isfinite(pt) | (pt <= 0)
    bad_jets = torch.unique(jet_idx[bad_c])
    if len(bad_jets):
        print(f"WARNING: {int(bad_c.sum())} bad constituents in {len(bad_jets)} jets; these jets are dropped")
        E[bad_c] = 0.0
        pt[bad_c] = 1.0
    return E, jet_idx, pt, bad_jets


@torch.no_grad()
def pooled_inference(batches, model, n_jets: int, device, include_start_token: bool = False) -> torch.Tensor:
    """Masked mean over constituents. By default the start token is EXCLUDED.

    For a causal model the start token's output is the same vector c for every jet,
    so including it adds c/(N+1) to each pooled embedding: a term that depends only on
    the multiplicity N.
    """
    model.eval()
    pooled, seen = [], 0
    for batch in batches:
        pf = batch["part_features"].to(device)
        pm = batch["part_mask"].to(device).int()
        pf_in, pm_in, _, _, n_prefix = _prepare_inputs(model, pf, pm)
        latent = model.backbone(pf_in, mask=pm_in)
        if not include_start_token:
            latent, pm_in = latent[:, n_prefix:], pm_in[:, n_prefix:]
        m = pm_in.float().unsqueeze(-1)
        take = min(pf.shape[0], n_jets - seen)
        pooled.append(((latent * m).sum(1) / m.sum(1).clamp_min(1))[:take].float().cpu())
        seen += take
        if seen >= n_jets:
            break
    return torch.cat(pooled)


# =============================================================================
# 4. Diagnostics and driver
# =============================================================================

def compare_weights(model_a, model_b, prefix: str = "backbone.") -> dict:
    """Compare parameters with a common name prefix between two models.

    Use it to answer: do the untrained NTP and MPM share their initial weights (same
    seed)? Is the fine-tuning run's 'untrained' state equal to a pretrained
    checkpoint, or a fresh random init?
    """
    sa, sb = model_a.state_dict(), model_b.state_dict()
    keys = [k for k in sa if k.startswith(prefix) and k in sb and sa[k].shape == sb[k].shape]
    only_a = [k for k in sa if k.startswith(prefix) and k not in sb]
    identical = sum(torch.equal(sa[k].cpu(), sb[k].cpu()) for k in keys)
    max_diff = max((float((sa[k].float().cpu() - sb[k].float().cpu()).abs().max()) for k in keys), default=float("nan"))
    return {"n_compared": len(keys), "n_identical": identical, "max_abs_diff": max_diff, "n_only_in_a": len(only_a)}


def compare_pair_mmd(
    model_a,
    model_b,
    batches,
    pt_getter,
    n_jets: int = 2000,
    betas: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    sigma_factors: Sequence[float] = (0.5, 1.0, 2.0),
    topks: Sequence[int] = (5, 10, 25, 50),
    n_sub: int = 1500,
    n_draws: int = 5,
    n_perm: int = 5,
    device="cpu",
    seed: int = 0,
) -> list[dict]:
    """CKNNA between two models on pT^beta-weighted MMD kernels, for every (beta, sigma factor).

    sigma is set per model by the median heuristic and multiplied by each factor, so the
    factor means the same relative bandwidth in both models' embedding spaces.
    """
    inferences = [constituent_inference(batches, model, pt_getter, n_jets, device) for model in (model_a, model_b)]
    return compare_constituents_mmd(*inferences, n_jets=n_jets, betas=betas, sigma_factors=sigma_factors,
                                    topks=topks, n_sub=n_sub, n_draws=n_draws, n_perm=n_perm,
                                    device=device, seed=seed)


def leading_constituents(inference, n_leading: int):
    """Keep only the first n_leading constituents of every jet of a `constituent_inference` output.

    "First" is the order of the input, which for AOJ / JetClass is descending pT. Jets with
    fewer constituents keep all of them. bad_jets is passed through unchanged.
    """
    E, jet_idx, pt, bad_jets = inference
    counts = torch.bincount(jet_idx)
    first = torch.cumsum(counts, 0) - counts                 # index of each jet's first constituent
    keep = torch.arange(len(jet_idx)) - first[jet_idx] < n_leading
    return E[keep], jet_idx[keep], pt[keep], bad_jets


def select_constituents(inference, n: int, mode: str = "hardest", seed: int = 0):
    """Keep n constituents of every jet of a `constituent_inference` output.

    mode: "hardest"  the first n in input order (descending pT for AOJ / JetClass)
          "softest"  the last n
          "random"   n chosen at random per jet; the choice depends only on `seed` and the jet
                     structure, so two models' outputs (same jets) get the same constituents
    Jets with at most n constituents keep all of them, whatever the mode. The kept
    constituents stay grouped by jet and in input order. bad_jets is passed through.
    """
    E, jet_idx, pt, bad_jets = inference
    counts = torch.bincount(jet_idx)
    first = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(jet_idx)) - first[jet_idx]          # position within the jet
    if mode == "hardest":
        keep = pos < n
    elif mode == "softest":
        keep = pos >= counts[jet_idx] - n
    elif mode == "random":
        g = torch.Generator().manual_seed(seed)
        key = torch.rand(len(jet_idx), generator=g, dtype=torch.float64)
        order = torch.argsort(jet_idx.double() * 2 + key)      # grouped by jet, random within jet
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(order))
        keep = rank - first[jet_idx] < n
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return E[keep], jet_idx[keep], pt[keep], bad_jets


def cut_batches_to_leading(batches, n_leading: int) -> list[dict]:
    """Copies of `batches` in which only the first n_leading valid constituents of each jet remain.

    The removed constituents are masked out and their features zeroed, so the model never
    sees them (unlike `leading_constituents`, which cuts after inference).
    """
    out = []
    for b in batches:
        mask = b["part_mask"].bool()
        keep = mask & (mask.int().cumsum(dim=1) <= n_leading)
        # torch.where, not a product: a NaN feature times 0 would stay NaN
        features = torch.where(keep.unsqueeze(-1), b["part_features"], torch.zeros_like(b["part_features"]))
        out.append({**b, "part_mask": keep.to(b["part_mask"].dtype), "part_features": features})
    return out


def compare_constituents_mmd(
    inference_a,
    inference_b,
    n_jets: int,
    betas: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    sigma_factors: Sequence[float] = (0.5, 1.0, 2.0),
    topks: Sequence[int] = (5, 10, 25, 50),
    n_sub: int = 1500,
    n_draws: int = 5,
    n_perm: int = 5,
    device="cpu",
    seed: int = 0,
) -> list[dict]:
    """The kernel and CKNNA part of `compare_pair_mmd`, on two `constituent_inference` outputs."""
    kernels, sigmas, bad = [], [], []
    for E, jid, pt, bad_jets in (inference_a, inference_b):
        bad.append(bad_jets)
        s0 = median_heuristic_sigma(E, seed=seed)
        sigmas.append(s0)
        Ks = mmd_kernels(E, jid, pt, n_jets, betas, [f * s0 for f in sigma_factors], device=device)
        kernels.append({(b, f): Ks[(b, f * s0)] for b in betas for f in sigma_factors})
        del Ks

    # drop jets that are non-finite in EITHER model, from both kernels
    keep = torch.ones(n_jets, dtype=torch.bool)
    keep[torch.cat(bad).long()] = False
    keep = keep.nonzero().squeeze(1)

    rows = []
    for b in betas:
        for f in sigma_factors:
            Ka = kernels[0][(b, f)][keep][:, keep].to(device)
            Kb = kernels[1][(b, f)][keep][:, keep].to(device)
            for r in cknna_sweep(Ka, Kb, topks, n_sub, n_draws, n_perm=n_perm, seed=seed):
                rows.append({"beta": b, "sigma_factor": f, "sigma_a": sigmas[0], "sigma_b": sigmas[1],
                             "n_jets_used": len(keep), **r})
    return rows


# =============================================================================
# Self-tests (CPU, synthetic data):  python constituent_kernels.py
# =============================================================================

def _self_test():
    torch.manual_seed(0)
    n, d = 600, 32
    X = torch.randn(n, d)

    # CKNNA: identity, invariances, permutation null, partial alignment
    K = linear_kernel(X, normalize=False)
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    assert abs(cknna_from_kernels(K, K, 10) - 1) < 1e-9
    assert abs(cknna_from_kernels(K, linear_kernel(3.0 * X @ Q, normalize=False), 10) - 1) < 1e-6
    p = torch.randperm(n)
    null = cknna_from_kernels(K, K[p][:, p], 10)
    noisy = cknna_from_kernels(K, linear_kernel(X + 0.7 * torch.randn(n, d), normalize=False), 10)
    print(f"CKNNA identity=1, rotation+scale=1, permutation={null:.4f} (k/n={10/n:.4f}), noisy={noisy:.3f}")
    assert abs(null) < 0.05 and 0.1 < noisy < 0.95

    # MMD: blockwise kernels against a direct per-pair computation
    n_jets, D = 23, 8
    counts = torch.randint(1, 15, (n_jets,))
    jid = torch.repeat_interleave(torch.arange(n_jets), counts)
    E = torch.randn(len(jid), D)
    pt = torch.rand(len(jid)) * 100 + 0.5
    betas, sigmas = (0.0, 1.0, 2.0), (0.7, 2.0)
    Ks = mmd_kernels(E, jid, pt, n_jets, betas, sigmas, block_jets=5)
    for b in betas:
        for s in sigmas:
            ref = torch.zeros(n_jets, n_jets, dtype=torch.float64)
            for i in range(n_jets):
                for j in range(n_jets):
                    ei, ej = E[jid == i].double(), E[jid == j].double()
                    wi, wj = pt[jid == i].double() ** b, pt[jid == j].double() ** b
                    wi, wj = wi / wi.sum(), wj / wj.sum()
                    R = torch.exp(-torch.cdist(ei, ej) ** 2 / (2 * s**2))
                    ref[i, j] = wi @ R @ wj
            err = (Ks[(b, s)] - ref).abs().max().item()
            eig_min = torch.linalg.eigvalsh(Ks[(b, s)]).min().item()
            assert err < 1e-5, err
            assert eig_min > -1e-6, eig_min
    print("MMD blockwise == direct computation for all (beta, sigma); all kernels PSD")

    # beta = 0 gives uniform weights
    w0 = jet_weights(pt, jid, n_jets, 0.0)
    assert torch.allclose(w0, 1.0 / counts[jid].float())
    # an overall pT scale cancels
    assert torch.allclose(jet_weights(pt, jid, n_jets, 1.5), jet_weights(37 * pt, jid, n_jets, 1.5), atol=1e-6)
    print("weights: beta=0 uniform, invariant to overall pT scale")

    # leading constituents: per jet the first min(n, count) constituents, in order
    Ec, jc, pc, _ = leading_constituents((E, jid, pt, torch.tensor([])), 3)
    assert torch.equal(torch.bincount(jc, minlength=n_jets), counts.clamp(max=3))
    assert torch.equal(Ec, torch.cat([E[jid == j][:3] for j in range(n_jets)]))
    mask = torch.arange(6)[None, :] < torch.tensor([[2], [6]])
    feats = torch.randn(2, 6, 4)
    feats[1, 5, 0] = float("nan")
    cut = cut_batches_to_leading([{"part_mask": mask.int(), "part_features": feats}], 3)[0]
    assert cut["part_mask"].sum(1).tolist() == [2, 3] and torch.isfinite(cut["part_features"]).all()
    print("leading constituents: kept the first n per jet, after and before inference")

    # select_constituents: hardest == leading_constituents, softest = last n, random = n distinct
    # ones per jet, identical for the same seed, all kept in input order
    inf = (E, jid, pt, torch.tensor([]))
    assert torch.equal(select_constituents(inf, 3, "hardest")[0], Ec)
    Es, js = select_constituents(inf, 3, "softest")[:2]
    assert torch.equal(Es, torch.cat([E[jid == j][-3:] for j in range(n_jets)]))
    Er, jr = select_constituents(inf, 3, "random", seed=5)[:2]
    assert torch.equal(torch.bincount(jr, minlength=n_jets), counts.clamp(max=3))
    assert torch.equal(Er, select_constituents(inf, 3, "random", seed=5)[0])
    assert torch.all(jr[1:] >= jr[:-1])
    for j in range(n_jets):   # random picks are distinct constituents of the right jet, in input order
        rows = [torch.nonzero((E[jid == j] == e).all(1)).item() for e in Er[jr == j]]
        assert rows == sorted(set(rows))
    print("select_constituents: hardest / softest / random per jet, reproducible, input order kept")


if __name__ == "__main__":
    _self_test()