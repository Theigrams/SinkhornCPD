"""RPOT baseline (Qin et al., Computer Graphics Forum 2022)
Paper: "Registration of Point Clouds on Partial Optimal Transport"
Code:  https://github.com/Hongxing-CQU/RPOT

This is a PyTorch port of the official MATLAB implementation
(UnbalanceRegistration.m / unbalanced_OT.m / ScalingAlgorithmRG_totalRG.m).
The core Sinkhorn loop runs on GPU in parallel, giving a large speed-up while keeping numerical results consistent with the original code.

Coordinate convention:
  MATLAB reference estimates  X ≈ Y @ R + t  (row-vectors).
  The public register(source, target) wrapper therefore calls
  _rpot(target, source) and returns R.T so that
  source @ R.T + t ≈ target.

Hyperparameters (same as the public MATLAB script):
  epsilon=0.004, alpha=0, beta=1,
  alpha_totalmass=0, beta_totalmass=0.8,
  threshold=1e-5, anneal_rate=0.9, tolerance=1e-6.
  inner_iter_max is reduced from 500 → 100 for uniform comparison.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class _Config:
    epsilon: float = 0.004
    alpha: float = 0.0
    beta: float = 1.0
    alpha_totalmass: float = 0.0
    beta_totalmass: float = 0.8
    threshold: float = 1e-5
    anneal_rate: float = 0.9
    outer_iter_max: int = 50
    inner_iter_max: int = 100
    tolerance: float = 1e-6
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64


def _pairwise_sq(X, Y):
    return torch.clamp(
        (X * X).sum(1, keepdim=True) + (Y * Y).sum(1, keepdim=True).T - 2 * X @ Y.T,
        min=0.0,
    )


@torch.no_grad()
def _sinkhorn_unbalanced(p1, p2, C, cost_old, cfg):
    eps = torch.finfo(C.dtype).eps
    K = torch.exp(-C / cfg.epsilon)
    b = torch.ones(p2.shape[0], device=C.device, dtype=C.dtype)
    z = torch.tensor(1.0, device=C.device, dtype=C.dtype)
    alpha, beta = cfg.alpha, cfg.beta
    a_tm, b_tm = cfg.alpha_totalmass, cfg.beta_totalmass

    T = None
    for it in range(cfg.inner_iter_max):
        s1 = z * (K @ b + eps)
        a = torch.minimum(torch.tensor(beta), torch.maximum(torch.tensor(alpha), s1 / p1))
        a = p1 * a / s1
        s2 = z * (K.T @ a + eps)
        b = torch.minimum(torch.tensor(beta), torch.maximum(torch.tensor(alpha), s2 / p2))
        b = p2 * b / s2
        s3 = a @ (K @ b) + eps
        z = torch.minimum(torch.tensor(b_tm), torch.maximum(torch.tensor(a_tm), s3)) / s3
        if it % 20 == 0:
            T = z * (a[:, None] * K) * b[None, :]
            cost = (T * C).sum()
            if torch.abs(cost - cost_old) < cfg.tolerance:
                break
            cost_old = cost
    return T, cost_old


@torch.no_grad()
def _rpot(X, Y, cfg):
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(X, device=device, dtype=cfg.dtype)
    Y = torch.as_tensor(Y, device=device, dtype=cfg.dtype)
    N, D = X.shape
    M = Y.shape[0]

    Mx = torch.full((N,), 1.0 / N, device=device, dtype=cfg.dtype)
    My = torch.full((M,), 1.0 / M, device=device, dtype=cfg.dtype)
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    R = torch.eye(D, device=device, dtype=cfg.dtype)
    preR = torch.zeros_like(R)
    t = torch.zeros(D, device=device, dtype=cfg.dtype)
    cost_old = torch.tensor(9999.0, device=device, dtype=cfg.dtype)
    C = _pairwise_sq(Xc, Yc)

    outer = 0
    while torch.linalg.norm(R - preR, "fro") > cfg.threshold and outer < cfg.outer_iter_max:
        outer += 1
        preR = R.clone()
        T, cost_old = _sinkhorn_unbalanced(Mx, My, C, cost_old, cfg)
        sumT = T.sum()
        ux = (Xc.T @ T.sum(1)) / sumT
        uy = (Yc.T @ T.sum(0)) / sumT
        A = (Yc - uy).T @ T.T @ (Xc - ux)
        U, _, Vh = torch.linalg.svd(A)
        D_det = torch.eye(D, device=device, dtype=cfg.dtype)
        D_det[-1, -1] = torch.det(U @ Vh)
        R = U @ D_det @ Vh
        t = ux - uy @ R
        C = _pairwise_sq(Xc, Yc @ R + t)
        cfg.epsilon *= cfg.anneal_rate

    t = t + X.mean(0) - Y.mean(0) @ R
    return R, t


def register(source, target):
    cfg = _Config()
    R_row, t = _rpot(np.asarray(target, dtype=np.float64), np.asarray(source, dtype=np.float64), cfg)
    return R_row.cpu().numpy().T, t.cpu().numpy()
