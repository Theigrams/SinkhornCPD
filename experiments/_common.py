"""Shared experiment runner: time + score one method on a list of pairs."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import rotation_error, translation_error, rmse

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_method(register_fn, source, pairs, *, desc=""):
    """For each (target, R_gt, t_gt) in pairs, time the registration and score it.

    Returns a list of dicts with rre, rte, rmse, time (NaN on failure).
    """
    rows = []
    for i, (target, R_gt, t_gt) in enumerate(tqdm(pairs, desc=desc, leave=False)):
        target = np.asarray(target, dtype=np.float64)
        try:
            t0 = time.perf_counter()
            R, t = register_fn(source, target)
            elapsed = time.perf_counter() - t0
            rre = rotation_error(R, R_gt)
            rte = translation_error(t, t_gt)
            rms = rmse(source, R, t, R_gt, t_gt)
        except Exception as e:
            tqdm.write(f"  pair {i}: {type(e).__name__}: {e}")
            rre = rte = rms = elapsed = float("nan")
        rows.append({"pair": i, "rre": rre, "rte": rte, "rmse": rms, "time": elapsed})
    return rows


def save_csv(path, rows, *, level_col=None, level=None):
    """Write rows to path.

    If level_col is given, replace rows matching that level (so the file accumulates
    across levels but a re-run of one level overwrites cleanly).  If level_col is
    None, overwrite the entire file — this prevents the common "append duplicates
    on re-run" bug for single-shot experiments.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if level_col is not None:
        rows = [{level_col: level, **r} for r in rows]
    df_new = pd.DataFrame(rows)
    if level_col is not None and path.exists():
        df_old = pd.read_csv(path)
        df_old = df_old[df_old[level_col] != level]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path
