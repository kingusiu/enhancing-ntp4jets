"""
CKNNA (centered kernel nearest-neighbour alignment) for jet foundation model
embeddings, with the null and identity-map baselines needed to read it.

Inputs are per-constituent embeddings of shape (B, N, D) -- e.g. (B, 150, 128) --
which are pooled to one vector per jet before any alignment is computed.

Definition (Huh et al. 2024, appendix). With centered linear kernels
K = <phi_i, phi_j> and L = <psi_i, psi_j> over the same n jets,

    CKA   = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    CKNNA = HSIC(a.K, a.L) / sqrt(HSIC(a_K.K, a_K.K) * HSIC(a_L.L, a_L.L))

where a_ij = 1 iff j is among the topk nearest neighbours of i in BOTH spaces,
a_K is the kNN mask of K alone, a_L that of L alone, and '.' is elementwise.
Using the mutual mask in the numerator and each space's own mask in the
denominators is what makes the metric discriminative; see the note in cknna().
As topk -> n-1 all three masks fill in and CKNNA -> CKA.

Two variants are provided:

  cknna / sweep_topk                       raw features, feature-centered (as before)
  cknna_normalized / sweep_topk_normalized the authors' preprocessing (paper,
                                           Appendix C; platonic-rep measure_alignment.py):
                                           clamp outlier elements at the 95th percentile
                                           of |value|, L2-normalise each embedding, no
                                           feature centering. Neighbours are then ranked
                                           by cosine similarity, so per-jet norms (e.g.
                                           multiplicity-driven ones) cannot dominate them.

HSIC form. The kNN-masked kernels are not symmetric, and for those the original
estimator in this file (hsic_form="frobenius", pairing sum(K * L)) and the one in
platonic-rep (hsic_form="reference", trace form sum(K * L.T), sum(K @ L)) give
noticeably different CKNNA values (~10% on synthetic data). The normalized variant
defaults to "reference"; with it, cknna_normalized reproduces platonic-rep's
AlignmentMetrics.cknna up to the 1e-6 they add to the denominator.
"""

from __future__ import annotations

import numpy as np
import torch

Tensor = torch.Tensor


# ---------------------------------------------------------------- pooling
def pool_jets(x: Tensor, mask: Tensor | None = None, mode: str = "mean") -> Tensor:
    """(B, N, D) per-constituent embeddings -> (B, D) per-jet embeddings.

    mask: (B, N) bool or float, True/1 for real constituents, False/0 for padding.
          Strongly recommended: JetClass jets are padded to N=150, and averaging
          the padding tokens in makes the pooled vector depend on multiplicity in
          a way that shows up as spurious alignment (both models see the same
          multiplicities, so both are biased the same way).
    """
    if x.ndim != 3:
        raise ValueError(f"expected (B, N, D), got {tuple(x.shape)}")

    if mask is None:
        m = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
    else:
        m = mask.to(dtype=x.dtype, device=x.device)
    m = m.unsqueeze(-1)  # (B, N, 1)

    if mode == "mean":
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)
    if mode == "sum":
        return (x * m).sum(1)
    if mode == "max":
        return x.masked_fill(m == 0, float("-inf")).max(1).values
    if mode == "first":  # leading-pT constituent, or a jet/class token at index 0
        return x[:, 0]
    raise ValueError(f"unknown pooling mode: {mode}")


# ---------------------------------------------------------------- kernels
def _prepare(feats: Tensor, center: bool = True) -> Tensor:
    """(n, D) -> float64, feature-centered.

    For a linear kernel, subtracting the feature mean is exactly equivalent to
    double-centering the kernel matrix (H K H), and is cheaper.
    """
    f = feats.to(torch.float64)
    if center:
        f = f - f.mean(dim=0, keepdim=True)
    return f


def _knn_mask(K: Tensor, topk: int) -> Tensor:
    """Boolean (n, n): True where j is among the topk nearest neighbours of i.

    Neighbours are taken by kernel value (largest = nearest), diagonal excluded.
    The mask is NOT symmetrised: a_ij follows the definition 'j in knn(i)'.
    """
    n = K.shape[0]
    if not 1 <= topk <= n - 1:
        raise ValueError(f"topk must be in [1, n-1]; got topk={topk}, n={n}")
    Kd = K.clone()
    Kd.fill_diagonal_(float("-inf"))
    idx = Kd.topk(topk, dim=-1).indices
    mask = torch.zeros_like(K, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def _hsic(K: Tensor, L: Tensor, unbiased: bool = True, form: str = "frobenius") -> Tensor:
    """Hilbert-Schmidt independence criterion between two kernel matrices.

    HSIC does the centering itself, so the kernels passed in are the raw
    (optionally masked) Gram matrices.

    form="frobenius": pairs K and L elementwise (original behaviour of this file).
    form="reference": trace form tr(K L), as in platonic-rep metrics.hsic_unbiased.
    Identical for symmetric K, L; they differ only for kNN-masked kernels.
    """
    m = K.shape[0]
    if unbiased:  # Song et al. estimator, diagonal removed
        Kt = K.clone().fill_diagonal_(0)
        Lt = L.clone().fill_diagonal_(0)
        if form == "reference":
            term1 = (Kt * Lt.T).sum()
            term3 = 2 * (Kt @ Lt).sum() / (m - 2)
        else:
            term1 = (Kt * Lt).sum()
            term3 = 2 * (Kt.sum(0) @ Lt.sum(0)) / (m - 2)
        term2 = Kt.sum() * Lt.sum() / ((m - 1) * (m - 2))
        return (term1 + term2 - term3) / (m * (m - 3))
    H = torch.eye(m, dtype=K.dtype, device=K.device) - 1.0 / m
    return torch.trace(K @ H @ L @ H)


def cka(feats_a: Tensor, feats_b: Tensor, center: bool = True, unbiased: bool = True) -> float:
    """Linear CKA between two (n, D) representations of the same n jets."""
    A, B = _prepare(feats_a, center), _prepare(feats_b, center)
    K, L = A @ A.T, B @ B.T
    num = _hsic(K, L, unbiased)
    den = torch.sqrt(_hsic(K, K, unbiased) * _hsic(L, L, unbiased))
    return float(num / den)


def cknna(
    feats_a: Tensor,
    feats_b: Tensor,
    topk: int = 10,
    center: bool = True,
    unbiased: bool = True,
    hsic_form: str = "frobenius",
) -> float:
    """CKNNA between two (n, D) representations of the same n jets.

    Row i of feats_a and row i of feats_b must be the same jet.

    The numerator restricts to MUTUAL neighbours (the intersection of the two
    kNN graphs); each normalisation term restricts to that space's OWN kNN
    graph. The asymmetry is the whole point: the ratio is high only when the two
    spaces agree about who is near whom. Using the joint mask in the
    denominators as well collapses the metric -- the selected entries are large
    in both kernels by construction, so the ratio sits near 1 for any pair of
    spaces, matched or shuffled.
    """
    A, B = _prepare(feats_a, center), _prepare(feats_b, center)
    K, L = A @ A.T, B @ B.T

    mask_k, mask_l = _knn_mask(K, topk), _knn_mask(L, topk)
    joint = mask_k & mask_l
    if not joint.any():
        return float("nan")

    num = _hsic(joint * K, joint * L, unbiased, hsic_form)
    den = torch.sqrt(_hsic(mask_k * K, mask_k * K, unbiased, hsic_form)
                     * _hsic(mask_l * L, mask_l * L, unbiased, hsic_form))
    return float(num / den)


# ---------------------------------------------------------------- normalized variant
def remove_outliers(feats: Tensor, q: float = 0.95, exact: bool = True) -> Tensor:
    """Clamp every element to +-t, with t the q-quantile of |elements|.

    Mirrors platonic-rep metrics.remove_outliers. exact=True (their default via
    compute_alignment(precise=True)) takes one global quantile over all elements;
    exact=False takes the per-sample quantile and averages it over samples.
    The threshold is one number per model, so it limits single large coordinates
    ("emergent outliers") without changing the overall scale.
    """
    if q >= 1:
        return feats
    a = feats.abs()
    if exact:
        flat = a.reshape(-1)
        k = min(int(q * flat.numel()), flat.numel() - 1)   # index into the ascending sort
        t = flat.kthvalue(k + 1).values                      # kthvalue is 1-based
    else:
        t = torch.quantile(a.flatten(start_dim=1), q, dim=1).mean()
    return feats.clamp(-t, t)


def prepare_normalized(feats: Tensor, q: float = 0.95, exact: bool = True) -> Tensor:
    """(n, D) -> float64: outlier clamp, then L2 normalisation of each row.

    Order as in the reference: prepare_features (clamp) first, then
    F.normalize(p=2, dim=-1) in compute_score. Apply this to the FULL set of jets
    of one model, before any subsampling, so the clamp threshold is the same for
    every draw.
    """
    f = remove_outliers(feats.to(torch.float64), q=q, exact=exact)
    return torch.nn.functional.normalize(f, p=2, dim=-1)


def finite_rows(*feats: Tensor) -> Tensor:
    """Boolean (n,): True for jets whose embedding is finite in EVERY given model.

    One NaN/inf embedding turns every kernel it enters into NaN, so such jets have
    to be dropped from all models together, keeping the rows paired.
    """
    keep = torch.ones(feats[0].shape[0], dtype=torch.bool, device=feats[0].device)
    for f in feats:
        keep &= torch.isfinite(f).all(dim=1).to(keep.device)
    return keep


def cknna_normalized(
    feats_a: Tensor,
    feats_b: Tensor,
    topk: int = 10,
    q: float = 0.95,
    exact: bool = True,
    unbiased: bool = True,
    hsic_form: str = "reference",
) -> float:
    """CKNNA with the authors' preprocessing: clamp at the q-quantile, L2-normalise.

    Features are NOT centered after normalising (center=False), as in the reference
    code: centering unit vectors gives them different lengths again (their distance
    from the mean direction), which would bring norm effects back into the
    neighbour ranking. HSIC still centers the masked kernels internally.
    Jets that are non-finite in either model are dropped from both.
    """
    keep = finite_rows(feats_a, feats_b)
    A = prepare_normalized(feats_a[keep], q, exact)
    B = prepare_normalized(feats_b[keep], q, exact)
    return cknna(A, B, topk=topk, center=False, unbiased=unbiased, hsic_form=hsic_form)


def sweep_topk_normalized(
    feats_a: Tensor,
    feats_b: Tensor,
    topks=(5, 10, 25, 50, 100),
    q: float = 0.95,
    exact: bool = True,
    hsic_form: str = "reference",
    **kw,
) -> list[dict]:
    """sweep_topk with the authors' preprocessing, applied once to the full sets.

    Drops jets that are non-finite in either model, clamps and L2-normalises each
    model's embeddings over all remaining jets, then subsamples as sweep_topk does
    (same seed, so the same draws as the raw version when no jet is dropped).
    Each row also reports how many jets were dropped.
    """
    kw.pop("center", None)
    keep = finite_rows(feats_a, feats_b)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"sweep_topk_normalized: dropping {n_dropped} jets with non-finite embeddings")
    A = prepare_normalized(feats_a[keep], q, exact)
    B = prepare_normalized(feats_b[keep], q, exact)
    rows = sweep_topk(A, B, topks=topks, center=False, hsic_form=hsic_form, **kw)
    for r in rows:
        r.update({"normalized": True, "q": q, "hsic_form": hsic_form, "n_dropped": n_dropped})
    return rows


# ---------------------------------------------------------------- baselines
def permutation_null(
    feats_a: Tensor,
    feats_b: Tensor,
    topk: int = 10,
    n_repeats: int = 20,
    seed: int = 0,
    center: bool = True,
    hsic_form: str = "frobenius",
) -> dict:
    """CKNNA with the jet correspondence destroyed: the floor to measure against.

    Shuffling the rows of B breaks the pairing while leaving both marginal
    geometries untouched, so whatever CKNNA survives is what the metric reports
    for two unrelated spaces of this shape and anisotropy. Report the measured
    value against mean + a few sigma of this, never as a bare number.

    Note there is no point in a random-rotation null for CKNNA: the linear kernel
    K = A A^T is invariant under orthogonal transforms of A, so rotating B leaves
    CKNNA exactly unchanged. Rotation is only a meaningful null for the
    coordinate-wise baselines below.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = feats_a.shape[0]
    vals = []
    for _ in range(n_repeats):
        perm = torch.randperm(n, generator=g).to(feats_b.device)
        vals.append(cknna(feats_a, feats_b[perm], topk=topk, center=center, hsic_form=hsic_form))
    v = np.asarray(vals, dtype=float)
    std = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return {"mean": float(v.mean()), "std": std, "values": v}


def identity_baselines(feats_a: Tensor, feats_b: Tensor, seed: int = 0) -> dict:
    """What you get with no learned map at all. Only defined when d_a == d_b.

    Reported quantities:
      paired_cos        mean cos(u_i, v_i) over jets -- the identity 'translation'
      rotated_cos       same after a random orthogonal map applied to B
      anisotropy_a/b    mean cos between two random distinct jets within one space.
                        This is the number that makes paired_cos readable: in an
                        anisotropic space, unrelated embeddings routinely sit at
                        cos ~ 0.5+, so a translation scoring 0.6 may be doing
                        nothing at all.
      top1 / mean_rank  retrieval of the true partner among all n jets under the
                        identity map. Chance mean_rank is (n+1)/2.
    """
    if feats_a.shape[1] != feats_b.shape[1]:
        raise ValueError("identity baseline needs equal dimensions")

    A = torch.nn.functional.normalize(feats_a.to(torch.float64), dim=-1)
    B = torch.nn.functional.normalize(feats_b.to(torch.float64), dim=-1)
    n = A.shape[0]

    S = A @ B.T                      # (n, n) cosine similarities
    paired = S.diagonal()
    off = ~torch.eye(n, dtype=torch.bool, device=S.device)

    ranks = (S > paired.unsqueeze(1)).sum(dim=1) + 1  # rank of the true partner
    Sa, Sb = A @ A.T, B @ B.T

    g = torch.Generator(device="cpu").manual_seed(seed)
    d = A.shape[1]
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    Brot = torch.nn.functional.normalize(feats_b.to(torch.float64) @ Q.to(B.device), dim=-1)

    return {
        "paired_cos": float(paired.mean()),
        "unpaired_cos": float(S[off].mean()),
        "rotated_cos": float((A * Brot).sum(-1).mean()),
        "anisotropy_a": float(Sa[off].mean()),
        "anisotropy_b": float(Sb[off].mean()),
        "top1": float((ranks == 1).to(torch.float64).mean()),
        "top5": float((ranks <= 5).to(torch.float64).mean()),
        "mean_rank": float(ranks.to(torch.float64).mean()),
        "chance_mean_rank": (n + 1) / 2,
    }


# ---------------------------------------------------------------- estimation
def cknna_with_errors(
    feats_a: Tensor,
    feats_b: Tensor,
    topk: int = 10,
    n_sub: int = 4096,
    n_draws: int = 10,
    seed: int = 0,
    center: bool = True,
    hsic_form: str = "frobenius",
) -> dict:
    """CKNNA over repeated random subsets, with the matched permutation null.

    Kernels are n x n, so the full dataset is not an option: n = 4096 in float64
    is ~130 MB per kernel, n = 20000 is ~3 GB. Subsampling also gives error bars,
    which a single number does not.

    CKNNA depends on topk and on n (the neighbourhood is topk/n of the sample),
    so fix n across all configurations you intend to compare, and sweep topk.
    """
    n = feats_a.shape[0]
    n_sub = min(n_sub, n)
    g = torch.Generator(device="cpu").manual_seed(seed)

    obs, null = [], []
    for d in range(n_draws):
        idx = torch.randperm(n, generator=g)[:n_sub].to(feats_a.device)
        a, b = feats_a[idx], feats_b[idx]
        obs.append(cknna(a, b, topk=topk, center=center, hsic_form=hsic_form))
        null.append(permutation_null(a, b, topk=topk, n_repeats=1, seed=seed + d, center=center,
                                     hsic_form=hsic_form)["mean"])

    o, u = np.asarray(obs), np.asarray(null)
    u_std = u.std(ddof=1) if len(u) > 1 else 0.0
    z = (o.mean() - u.mean()) / u_std if u_std > 0 else float("inf")
    return {
        "topk": topk,
        "n": n_sub,
        "cknna": float(o.mean()),
        "cknna_std": float(o.std(ddof=1)) if len(o) > 1 else float("nan"),
        "null": float(u.mean()),
        "null_std": float(u_std),
        "z": float(z),
    }


def sweep_topk(feats_a, feats_b, topks=(5, 10, 25, 50, 100), **kw) -> list[dict]:
    """CKNNA vs topk. A result that only appears at one topk is not a result."""
    return [cknna_with_errors(feats_a, feats_b, topk=k, **kw) for k in topks]


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    torch.manual_seed(0)
    n, N, D = 3000, 150, 128

    # Synthetic stand-in: a shared latent cause seen through two different
    # random nonlinear encoders, plus independent noise. Replace with your
    # frozen-backbone outputs.
    z = torch.randn(n, 32)
    W1, W2 = torch.randn(32, D) / 32**0.5, torch.randn(32, D) / 32**0.5
    u_jet = torch.tanh(z @ W1) + 0.5 * torch.randn(n, D)
    v_jet = torch.tanh(z @ W2) + 0.5 * torch.randn(n, D)

    # ... expanded to (B, N, D) with padding, to exercise the pooling path
    lengths = torch.randint(20, N + 1, (n,))
    mask = (torch.arange(N)[None, :] < lengths[:, None])
    u = u_jet[:, None, :] + 0.1 * torch.randn(n, N, D)
    v = v_jet[:, None, :] + 0.1 * torch.randn(n, N, D)

    U, V = pool_jets(u, mask), pool_jets(v, mask)

    print("--- sanity ---")
    print(f"self-CKNNA (U vs U):      {cknna(U, U, topk=10):.4f}   (must be 1)")
    Q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
    print(f"rotation invariance:      {cknna(U, (V.double() @ Q).float(), topk=10):.4f}"
          f"  == {cknna(U, V, topk=10):.4f}")
    print(f"CKNNA at topk=n-1:        {cknna(U, V, topk=U.shape[0] - 1):.4f}"
          f"  ~ CKA {cka(U, V):.4f}")

    print("\n--- alignment ---")
    for row in sweep_topk(U, V, topks=(5, 10, 25, 50), n_sub=1500, n_draws=5):
        print(f"topk={row['topk']:>4}  CKNNA={row['cknna']:.4f}+-{row['cknna_std']:.4f}   "
              f"null={row['null']:.4f}+-{row['null_std']:.4f}   z={row['z']:.1f}")

    print("\n--- alignment, normalized (clamp q=0.95 + L2) ---")
    for row in sweep_topk_normalized(U, V, topks=(5, 10, 25, 50), n_sub=1500, n_draws=5):
        print(f"topk={row['topk']:>4}  CKNNA={row['cknna']:.4f}+-{row['cknna_std']:.4f}   "
              f"null={row['null']:.4f}+-{row['null_std']:.4f}   z={row['z']:.1f}")

    print("\n--- shared per-jet scale only (no shared structure) ---")
    s = torch.exp(0.8 * torch.randn(1500, 1))
    Us = s * (2 * torch.randn(1, D) + torch.randn(1500, D))
    Vs = s * (2 * torch.randn(1, D) + torch.randn(1500, D))
    print(f"raw CKNNA: {cknna(Us, Vs, topk=10):.4f}   normalized CKNNA: {cknna_normalized(Us, Vs, topk=10):.4f}")

    print("\n--- identity-map baselines ---")
    print("(the synthetic features here are near-isotropic, so anisotropy ~ 0;")
    print(" real backbone embeddings will not be, which is exactly why you measure it)")
    for k, val in identity_baselines(U, V).items():
        print(f"{k:>18}: {val:.4f}" if isinstance(val, float) else f"{k:>18}: {val}")