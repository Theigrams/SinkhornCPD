"""Registration evaluation metrics."""

import numpy as np
from scipy.spatial import cKDTree


def rotation_error(R_est, R_gt):
    """Geodesic angle in degrees between two rotations.

    Input sanitization: we project R_est to the nearest rotation in SO(3) via
    SVD before computing the angle.  Without this, any `R_est` that drifts
    off SO(3) (‖R·Rᵀ − I‖ > 0) produces a `trace(R_estᵀ·R_gt)` that can exceed
    the theoretical maximum 3; the subsequent `np.clip(cos, -1, 1)` then
    silently caps the argument at 1 and `arccos(1) = 0`, reporting a perfect
    RE = 0° when the real geodesic angle may be several tenths of a degree.

    This is not hypothetical:  probreg's FilterReg / CPD use a C++ Kabsch
    whose intermediate `dr` matrices drift ~1e-5 from SO(3) (float32-level
    precision, not float64).  On ModelNet 61.8% of FilterReg's RE values
    were reported as exact 0° for that reason.  SinkhornCPD is unaffected
    because we build R from `torch.linalg.svd` in float64, so the drift is
    ~1e-15.  We project here to give every method a fair metric, without
    touching the wrappers.
    """
    R_est = np.asarray(R_est, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)
    # Nearest-rotation projection: closed form via SVD of R_est.
    U, _, Vt = np.linalg.svd(R_est)
    det_sign = np.sign(np.linalg.det(U @ Vt))
    R_est = U @ np.diag([1.0, 1.0, det_sign]) @ Vt
    cos = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    # clip still needed for the valid SO(3) range where cos ∈ {−1, 1} edge
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def translation_error(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))


def rmse(source, R_est, t_est, R_gt, t_gt):
    """RMSE of source under estimated vs ground-truth transform."""
    diff = source @ (R_est - R_gt).T + (t_est - t_gt)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def chamfer(X, Y):
    """Symmetric squared chamfer distance."""
    d_xy = cKDTree(Y).query(X)[0]
    d_yx = cKDTree(X).query(Y)[0]
    return float(np.mean(d_xy ** 2) + np.mean(d_yx ** 2))
