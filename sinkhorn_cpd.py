"""Sinkhorn-CPD: rigid registration via dual-KL unbalanced entropic OT."""

import numpy as np
import torch

EPS = 1e-12  # log/division floor
EPS_SIGMA = 1e-8  # σ² collapse floor


@torch.no_grad()
def sinkhorn_cpd(
    source,
    target,
    tau_x=1.0,
    tau_y=1.0,
    L=20,
    max_iter=50,
    tol=1e-5,
    R_init=None,
    t_init=None,
    device=None,
):
    """Align source Y to target X.  Returns (R, t, sigma2_history).

    tau_x, tau_y are the KL marginal weights; smaller values relax the transport
    plan's marginal constraints, increasing robustness to outliers and partial
    overlap at the cost of bias on clean correspondences.
    """
    if tau_x <= 0 or tau_y <= 0:
        raise ValueError(f"tau_x, tau_y must be > 0, got {tau_x}, {tau_y}")

    dtype = torch.float64
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    Y = torch.as_tensor(source, device=device, dtype=dtype)
    X = torch.as_tensor(target, device=device, dtype=dtype)
    M, D = Y.shape
    N = X.shape[0]

    # init: identity rotation, centroid translation, σ² = mean pairwise dist / D
    R = (
        torch.eye(D, device=device, dtype=dtype)
        if R_init is None
        else torch.as_tensor(R_init, device=device, dtype=dtype)
    )
    t = (X.mean(0) - R @ Y.mean(0)) if t_init is None else torch.as_tensor(t_init, device=device, dtype=dtype)
    sigma2 = torch.clamp(_pairwise_sq(X, Y).mean() / D, min=EPS_SIGMA)

    a = torch.full((N,), 1.0 / N, device=device, dtype=dtype)
    b = torch.full((M,), 1.0 / M, device=device, dtype=dtype)
    alpha = tau_y / (tau_y + 1.0 + EPS)
    beta = tau_x / (tau_x + 1.0 + EPS)

    sigma2_history = [sigma2.item()]
    obj_prev = float("inf")

    for it in range(max_iter):
        # E-step: cost C = ‖x-Ty‖²/(2σ²) + (D/2)log(2πσ²);  K = exp(-C)
        TY = Y @ R.T + t
        C = _pairwise_sq(X, TY).T / (2.0 * sigma2) + 0.5 * D * torch.log(2 * torch.pi * sigma2)
        K = torch.exp(-C)

        # generalized Sinkhorn: u←(b/Kv)^α, v←(a/Kᵀu)^β
        u = torch.ones(M, device=device, dtype=dtype)
        v = torch.ones(N, device=device, dtype=dtype)
        for _ in range(L):
            u = (b / (K @ v + EPS)) ** alpha
            v = (a / (K.T @ u + EPS)) ** beta
        Gamma = u[:, None] * K * v[None, :]

        # convergence on objective <C,Γ> + H(Γ)
        obj = (Gamma * C).sum().item() + (Gamma * (torch.log(Gamma + EPS) - 1.0)).sum().item()
        if abs(obj_prev - obj) < tol:
            break
        obj_prev = obj

        # M-step: weighted Procrustes + closed-form σ²
        gamma = Gamma.sum() + EPS
        r = Gamma.sum(0)  # (N,) target marginal
        c = Gamma.sum(1)  # (M,) source marginal
        cx = (X * r[:, None]).sum(0) / gamma  # weighted target centroid
        cy = (Y * c[:, None]).sum(0) / gamma

        S = X.T @ Gamma.T @ Y - gamma * cx[:, None] @ cy[None, :]
        U, _, Vh = torch.linalg.svd(S)
        # ensure proper rotation (det = +1)
        if torch.det(U @ Vh) < 0:
            U[:, -1] = -U[:, -1]
        R = U @ Vh
        t = cx - R @ cy

        # σ² from weighted residual; floor prevents premature collapse
        TY = Y @ R.T + t
        residual = (Gamma * _pairwise_sq(X, TY).T).sum()
        sigma2 = torch.clamp(residual / (D * gamma), min=EPS_SIGMA)
        sigma2_history.append(sigma2.item())

    return R.cpu().numpy(), t.cpu().numpy(), np.array(sigma2_history)


def _pairwise_sq(X, Y):
    """‖x - y‖² for all pairs.  Returns (N, M) tensor."""
    return torch.clamp(
        (X * X).sum(1, keepdim=True) + (Y * Y).sum(1, keepdim=True).T - 2.0 * X @ Y.T,
        min=0.0,
    )
