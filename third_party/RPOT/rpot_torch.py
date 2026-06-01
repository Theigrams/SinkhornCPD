"""
PyTorch re-implementation of the MATLAB RPOT (partial/unbalanced OT) rigid registration.

This file mirrors the original MATLAB code in this folder:
  - ScalingAlgorithmRG_totalRG.m
  - unbalanced_OT.m
  - UnbalanceRegistration.m

Conventions:
  - Internal computations follow MATLAB's row-vector convention:
      Y_transformed = Y @ R_row + t_row
    where points are (N, D) row vectors and rotation multiplies on the right.
  - For compatibility with other python wrappers in this repo (Open3D/probreg),
    the exposed wrapper returns R_col = R_row.T so that users can apply:
      Y_transformed = Y @ R_col.T + t
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class RPOTConfig:
    # Algorithm params (same defaults as UnbalanceRegistration.m for 3D)
    epsilon: float = 0.004
    alpha: float = 0.0
    beta: float = 1.0
    alpha_totalmass: float = 0.0
    beta_totalmass: float = 0.9
    threhold: float = 1e-5  # keep original MATLAB field name
    anneal_rate: float = 0.9
    outer_iter_max: int = 50  # Maximum outer iterations (R estimation)

    # ScalingAlgorithmRG_totalRG.m
    iter_max: int = 100
    tolerance: float = 1e-6
    cost_check_every: int = 20  # MATLAB: mod(i,20)==1 (i starts at 1)

    # Torch controls
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64


def _as_tensor(x, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def pairwise_sqeuclidean(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """pdist2(X, Y, 'squaredeuclidean') with row-vector points."""
    # X: (N, D), Y: (M, D)
    x2 = (X * X).sum(dim=1, keepdim=True)  # (N, 1)
    y2 = (Y * Y).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, M)
    D = x2 + y2 - 2.0 * (X @ Y.transpose(0, 1))
    return torch.clamp(D, min=0.0)


@torch.no_grad()
def scaling_algorithm_rg_totalrg(
    p1: torch.Tensor,
    p2: torch.Tensor,
    C: torch.Tensor,
    cost_old: torch.Tensor,
    cfg: RPOTConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch version of ScalingAlgorithmRG_totalRG.m.

    Args:
        p1: (N,) point masses of target X
        p2: (M,) point masses of source Y
        C:  (N, M) cost matrix (squared euclidean)
        cost_old: scalar tensor
        cfg: RPOTConfig

    Returns:
        T: (N, M) transport plan
        cost_old: updated scalar tensor
    """
    N = p1.shape[0]
    M = p2.shape[0]

    eps_val = torch.finfo(C.dtype).eps

    b = torch.ones(M, device=C.device, dtype=C.dtype)
    K = torch.exp(-C / cfg.epsilon)

    z = torch.tensor(1.0, device=C.device, dtype=C.dtype)

    alpha = torch.tensor(cfg.alpha, device=C.device, dtype=C.dtype)
    beta = torch.tensor(cfg.beta, device=C.device, dtype=C.dtype)
    alpha_tm = torch.tensor(cfg.alpha_totalmass, device=C.device, dtype=C.dtype)
    beta_tm = torch.tensor(cfg.beta_totalmass, device=C.device, dtype=C.dtype)

    T = None
    for it in range(cfg.iter_max):
        s1 = z * (K @ b + eps_val)
        a = torch.minimum(beta * p1, torch.maximum(alpha * p1, s1)) / s1

        s2 = z * (K.transpose(0, 1) @ a + eps_val)
        b = torch.minimum(beta * p2, torch.maximum(alpha * p2, s2)) / s2

        s3 = a @ (K @ b) + eps_val
        z = torch.minimum(beta_tm, torch.maximum(alpha_tm, s3)) / s3

        # MATLAB: if mod(i,20) == 1 (i starts at 1) -> torch: it % 20 == 0
        if it % cfg.cost_check_every == 0:
            T = z * (a[:, None] * K) * b[None, :]
            cost = torch.sum(T * C)
            # Match MATLAB behavior: when converged, break *without* updating cost_old.
            # (In ScalingAlgorithmRG_totalRG.m, cost_old update happens after the break.)
            if torch.abs(cost - cost_old) < cfg.tolerance:
                break
            cost_old = cost

    # Should always be set because it=0 triggers the check
    return T, cost_old


@torch.no_grad()
def rpot_register_matlab(
    X: torch.Tensor,
    Y: torch.Tensor,
    cfg: RPOTConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Run RPOT rigid registration, matching MATLAB's UnbalanceRegistration.m behavior.

    Args:
        X: (N, D) target points
        Y: (M, D) source points
        cfg: RPOTConfig

    Returns:
        R_row: (D, D) rotation used as Y @ R_row + t
        t_row: (D,) translation used as Y @ R_row + t
        info: dict with diagnostics
    """
    device = cfg.device if cfg.device is not None else torch.device("cpu")
    dtype = cfg.dtype

    X = _as_tensor(X, device=device, dtype=dtype)
    Y = _as_tensor(Y, device=device, dtype=dtype)

    N, dim = X.shape
    M, _ = Y.shape

    # Uniform masses
    Mx = torch.full((N,), 1.0 / N, device=device, dtype=dtype)
    My = torch.full((M,), 1.0 / M, device=device, dtype=dtype)

    # === preprocess data, align mass barycenter between X and Y ===
    X_mass_bary = X.mean(dim=0)  # (D,)
    Y_mass_bary = Y.mean(dim=0)

    Xc = X - X_mass_bary
    Yc = Y - Y_mass_bary

    # === init ===
    preR = torch.zeros((dim, dim), device=device, dtype=dtype)
    R = torch.ones((dim, dim), device=device, dtype=dtype)
    t = torch.zeros((dim,), device=device, dtype=dtype)

    Ytransformed = Yc @ R + t
    # Match MATLAB call pattern:
    #   - UnbalanceRegistration.m passes D=pdist2(X,Y) into unbalanced_OT()
    #   - unbalanced_OT() uses that D for the first ScalingAlgorithmRG_totalRG() call
    # i.e. the *first* cost matrix is between (X, Y), not (X, Ytransformed).
    C = pairwise_sqeuclidean(Xc, Yc)

    cost_old = torch.tensor(9999.0, device=device, dtype=dtype)

    outer_iter = 0
    T = None
    while torch.linalg.norm(R - preR, ord="fro") > cfg.threhold and outer_iter < cfg.outer_iter_max:
        outer_iter += 1
        preR = R

        T, cost_old = scaling_algorithm_rg_totalrg(Mx, My, C, cost_old, cfg)

        sumT = torch.sum(T)
        ones_M = torch.ones((M,), device=device, dtype=dtype)
        ones_N = torch.ones((N,), device=device, dtype=dtype)

        ux = (Xc.transpose(0, 1) @ (T @ ones_M)) / sumT  # (D,)
        uy = (Yc.transpose(0, 1) @ (T.transpose(0, 1) @ ones_N)) / sumT

        X_acrid = Xc - ux[None, :]
        Y_acrid = Yc - uy[None, :]

        A = Y_acrid.transpose(0, 1) @ T.transpose(0, 1) @ X_acrid  # (D, D)
        U, _, Vh = torch.linalg.svd(A, full_matrices=False)

        C_det = torch.eye(dim, device=device, dtype=dtype)
        C_det[-1, -1] = torch.det(U @ Vh)
        R = U @ C_det @ Vh

        t = ux - (uy @ R)

        Ytransformed = Yc @ R + t
        C = pairwise_sqeuclidean(Xc, Ytransformed)

        cfg.epsilon = cfg.epsilon * cfg.anneal_rate

    # Recover translation in the original coordinate system (MATLAB):
    # t0 = t0 + XmassBarycenter - YmassBarycenter * R0;
    t = t + X_mass_bary - (Y_mass_bary @ R)

    info = {
        "outer_iter": float(outer_iter),
        "final_epsilon": float(cfg.epsilon),
        "transport_cost": float(cost_old),
        "sumT": float(torch.sum(T)),
    }
    return R, t, info
