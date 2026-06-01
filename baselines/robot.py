"""RobOT baseline (Shen et al., NeurIPS 2021)
Paper: "Accurate Point Cloud Registration with Robust Optimal Transport"
Code:  https://github.com/uncbiag/robot

This script provides a clean, standalone PyTorch/KeOps implementation of the Rigid RobOT baseline, tailored for coordinate-only registration of point clouds normalized to the unit sphere.

Implementation origin:
  This port follows the official code paths used by rigid `ot_mapping`:
    robot/modules_reg/module_gradient_flow.py
      wasserstein_barycenter_mapping
    robot/modules_reg/module_gradflow_prealign.py
      GradFlowPreAlign.__call__

Remarks:
- The official demo uses FPFH features before OT; we drop them for a fair coordinate-only baseline and run OT directly on normalized 3D coords.
- Weighted Procrustes keeps the upstream w^2 quirk (weights applied to both centered factors) so numbers stay compatible with the reference code.
- Hyperparameters from partial_prealign_reg.py: blur=0.1, scaling=0.9, debias=False. We set reach=1.0 for uniform comparison.
- Barycentric mapping uses dense tensors by default (~3x faster than KeOps for N,M < 5000). KeOps fallback kept as _barycentric_mapping_keops for large point clouds.
"""

import numpy as np
import torch
from geomloss import SamplesLoss
from pykeops.torch import LazyTensor

BLUR = 0.1  # OT entropy scale
REACH = 1.0  # Use None for balanced OT; use finite value for unbalanced RobOT.
SCALING = 0.9  # epsilon-scaling ratio in GeomLoss
NITER = 50
REL_FTOL = 1e-5


def _barycentric_mapping_keops(TY, X, w_src, w_tgt, blur, reach, scaling):
    """KeOps LazyTensor version. Slower for N,M < 5000; kept as fallback for large point clouds."""
    geomloss = SamplesLoss(
        "sinkhorn",
        p=2,
        blur=blur,
        reach=reach,
        scaling=scaling,
        debias=False,
        potentials=True,
    )
    F, G = geomloss(w_src, TY, w_tgt, X)  # (1, N), (1, M)
    F = F.reshape(-1)  # (N,)
    G = G.reshape(-1)  # (M,)

    N, D = TY.shape
    M = X.shape[0]

    a_i = LazyTensor(w_src.view(N, 1, 1))
    b_j = LazyTensor(w_tgt.view(1, M, 1))
    x_i = LazyTensor(TY.view(N, 1, D))
    y_j = LazyTensor(X.view(1, M, D))
    F_i = LazyTensor(F.view(N, 1, 1))
    G_j = LazyTensor(G.view(1, M, 1))

    # log P_ij = log a_i + log b_j + F_i/blur² + G_j/blur² − ½‖x−y‖²/blur²
    sqrt2_blur = float(np.sqrt(2.0)) * blur
    xx_i = x_i / sqrt2_blur
    yy_j = y_j / sqrt2_blur
    f_i = a_i.log() + F_i / (blur**2)
    g_j = b_j.log() + G_j / (blur**2)
    C_ij = ((xx_i - yy_j) ** 2).sum(-1)
    log_P_ij = f_i + g_j - C_ij  # (N, M, 1) LazyTensor

    mapped = log_P_ij.sumsoftmaxweight(y_j, dim=1)  # (N, D)
    # logsumexp for stability (blur=0.1 ⇒ 1/blur²=100 amplifies log_P extremes);
    # subtract max since Kabsch only needs relative weights
    log_row_mass = log_P_ij.logsumexp(dim=1).view(-1)
    row_mass = (log_row_mass - log_row_mass.max()).exp()  # (N,)
    return mapped, row_mass


def _barycentric_mapping(TY, X, w_src, w_tgt, blur, reach, scaling):
    """Return (mapped, row_mass) from OT dual potentials. Dense tensor version, ~3x faster than KeOps for N,M < 5000."""
    geomloss_fn = SamplesLoss(
        "sinkhorn",
        p=2,
        blur=blur,
        reach=reach,
        scaling=scaling,
        debias=False,
        potentials=True,
    )
    F, G = geomloss_fn(w_src, TY, w_tgt, X)
    F = F.reshape(-1)
    G = G.reshape(-1)

    N, D = TY.shape
    M = X.shape[0]

    # (N,M) pairwise squared distance
    dist2 = torch.cdist(TY, X, p=2).pow(2)

    log_a = w_src.log().view(N, 1)
    log_b = w_tgt.log().view(1, M)
    log_P = log_a + log_b + F.view(N, 1) / (blur**2) + G.view(1, M) / (blur**2) - dist2 / (2.0 * blur**2)

    # softmax-weighted target points -> mapped positions
    log_P_max = log_P.max(dim=1, keepdim=True).values
    P = (log_P - log_P_max).exp()
    row_sum = P.sum(dim=1, keepdim=True).clamp(min=1e-30)
    mapped = (P @ X) / row_sum  # (N, D)

    log_row_mass = log_P_max.squeeze(1) + row_sum.squeeze(1).log()
    row_mass = (log_row_mass - log_row_mass.max()).exp()
    return mapped, row_mass


def _weighted_procrustes(X_src, X_dst, w):
    """Weighted Kabsch, faithful to upstream `solve_rigid` (module_gradflow_prealign.py).

    Note: upstream applies `w` to both factors, so the covariance uses w² rather
    than the textbook Σ wᵢ (yᵢ-μ_y)(xᵢ-μ_x)ᵀ. We keep this quirk so numbers match
    the reference implementation; row-mass weights here are near-uniform so the
    resulting R differs from the linear-w version only marginally.
    """
    ww = w[:, None]
    sw = ww.sum().clamp(min=1e-12)
    mu_s = (X_src * ww).sum(0) / sw
    mu_d = (X_dst * ww).sum(0) / sw
    Xs = X_src - mu_s
    Xd = X_dst - mu_d
    H = (Xd * ww).T @ (Xs * ww)  # upstream w² form
    U, _, Vh = torch.linalg.svd(H)
    sign = torch.det(U @ Vh).sign()
    Dm = torch.diag(torch.tensor([1.0, 1.0, sign.item()], device=X_src.device, dtype=X_src.dtype))
    R = U @ Dm @ Vh
    t = mu_d - R @ mu_s
    return R, t


def register(source, target):
    """Returns (R, t) such that source @ R.T + t ≈ target."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Y = torch.as_tensor(source, dtype=torch.float32, device=device)
    X = torch.as_tensor(target, dtype=torch.float32, device=device)
    w_src = torch.full((Y.shape[0],), 1.0 / Y.shape[0], device=device)
    w_tgt = torch.full((X.shape[0],), 1.0 / X.shape[0], device=device)

    R = torch.eye(3, device=device, dtype=torch.float32)
    t = torch.zeros(3, device=device, dtype=torch.float32)
    A_prev = None

    for _ in range(NITER):
        TY = (Y @ R.T + t).detach()
        with torch.no_grad():
            mapped, row_mass = _barycentric_mapping(TY, X, w_src, w_tgt, BLUR, REACH, SCALING)
        w_kab = row_mass.clamp(min=1e-8)
        R_delta, t_delta = _weighted_procrustes(TY, mapped, w_kab)
        R = R_delta @ R
        t = R_delta @ t + t_delta

        A = torch.cat([R, t[:, None]], dim=1)
        if A_prev is not None and torch.norm(A - A_prev) < REL_FTOL:
            break
        A_prev = A.clone()

    return R.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64)
