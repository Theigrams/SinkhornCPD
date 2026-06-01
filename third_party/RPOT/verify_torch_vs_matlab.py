from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent


def run_matlab_reference(out_mat: Path) -> None:
    cmd = (
        f"addpath('{THIS_DIR.as_posix()}'); "
        f"rpot_export_reference('{out_mat.as_posix()}');"
    )
    subprocess.run(["matlab", "-batch", cmd], check=True, cwd=str(REPO_ROOT))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_mat = Path(tmpdir) / "matlab_reference_output.mat"
        run_matlab_reference(out_mat)
        ref = scipy.io.loadmat(out_mat)
    R0 = np.asarray(ref["R0"], dtype=np.float64)
    t0 = np.asarray(ref["t0"], dtype=np.float64).reshape(-1)
    Y_ref = np.asarray(ref["Yorgtransformed"], dtype=np.float64)

    data = scipy.io.loadmat(THIS_DIR / "data.mat")
    X = np.asarray(data["X"], dtype=np.float64)
    Y = np.asarray(data["Y"], dtype=np.float64)

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "compare_methods"))

    from RPOT.rpot_torch import RPOTConfig, rpot_register_matlab

    cfg = RPOTConfig(
        epsilon=0.004,
        alpha=0.0,
        beta=1.0,
        alpha_totalmass=0.0,
        beta_totalmass=0.8,
        threhold=1e-5,
        anneal_rate=0.9,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    R_row, t_row, info = rpot_register_matlab(X, Y, cfg)
    R_row = R_row.detach().cpu().numpy()
    t_row = t_row.detach().cpu().numpy().reshape(-1)

    Y_torch = Y @ R_row + t_row

    r_max = float(np.max(np.abs(R_row - R0)))
    t_max = float(np.max(np.abs(t_row - t0)))
    y_max = float(np.max(np.abs(Y_torch - Y_ref)))

    print("=== RPOT Torch vs MATLAB (strict compare) ===")
    print(f"R max_abs_err: {r_max:.3e}")
    print(f"t max_abs_err: {t_max:.3e}")
    print(f"Y_aligned max_abs_err: {y_max:.3e}")
    print(f"outer_iter: {int(info.get('outer_iter', -1))}")
    print(f"transport_cost: {info.get('transport_cost'):.6g}")

    # These tolerances are tight for float64; relax if you run on GPU/float32.
    ok = (r_max < 1e-8) and (t_max < 1e-8) and (y_max < 1e-7)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
