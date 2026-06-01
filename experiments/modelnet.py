"""ModelNet40 cross-category benchmark (paper Table 5).

Data: datasets/modelnet/data.npz (2148 pairs from GeoTransformer pipeline)
  source[i]: (2048, 3) CAD model points
  target[i]: (2048, 3) cropped + transformed scan
  R_gt[i], t_gt[i]: source @ R_gt.T + t_gt ≈ target

Usage:
    python -m experiments.modelnet --method FGR
    python -m experiments.modelnet --method all
    python -m experiments.modelnet --method SinkhornCPD --n-pairs 100
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from baselines import METHODS
from datasets import DATASETS_DIR
from metrics import rotation_error, translation_error, rmse
from sinkhorn_cpd import sinkhorn_cpd
from experiments._common import RESULTS_DIR, save_csv

DATA_PATH = DATASETS_DIR / "modelnet" / "data.npz"


def _scpd(source, target):
    R, t, _ = sinkhorn_cpd(source, target, tau_x=1.0, tau_y=1.0)
    return R, t


METHODS_PLUS = {**METHODS, "SinkhornCPD": _scpd}


def load_modelnet():
    """Load all 2148 pairs.  Returns (sources, targets, R_gts, t_gts)."""
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["source"], d["target"], d["R_gt"], d["t_gt"]


def run_one_method(name, sources, targets, R_gts, t_gts, n):
    fn = METHODS_PLUS.get(name)
    if fn is None:
        print(f"[skip] {name}", flush=True)
        return

    rows = []
    for i in tqdm(range(n), desc=f"  {name}", leave=False):
        source = np.asarray(sources[i], dtype=np.float64)
        target = np.asarray(targets[i], dtype=np.float64)
        R_gt, t_gt = R_gts[i], t_gts[i]
        try:
            t0 = perf_counter()
            R, t = fn(source, target)
            elapsed = perf_counter() - t0
            rre = rotation_error(R, R_gt)
            rte = translation_error(t, t_gt)
            rms = rmse(source, R, t, R_gt, t_gt)
        except Exception as e:
            tqdm.write(f"  pair {i}: {type(e).__name__}: {e}")
            rre = rte = rms = elapsed = float("nan")
        rows.append({"pair": i, "rre": rre, "rte": rte, "rmse": rms, "time": elapsed})

    out = RESULTS_DIR / "modelnet" / f"{name}.csv"
    save_csv(out, rows)

    valid = [r for r in rows if r["rre"] == r["rre"]]
    mean_rre = np.nanmean([r["rre"] for r in rows])
    rec = sum(1 for r in valid if r["rre"] < 1 and r["rte"] < 0.1) / max(1, len(valid))
    mean_time = np.nanmean([r["time"] for r in rows])
    print(f"{name}: RE={mean_rre:.2f}deg  RR={rec*100:.1f}%  time={mean_time:.2f}s  -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--n-pairs", type=int, default=None)
    args = p.parse_args()

    method_names = list(METHODS_PLUS) if args.method == "all" else [args.method]
    sources, targets, R_gts, t_gts = load_modelnet()
    total = len(sources)
    n = total if args.n_pairs is None else min(args.n_pairs, total)
    print(f"Loaded {n} pairs", flush=True)

    for name in method_names:
        run_one_method(name, sources, targets, R_gts, t_gts, n)


if __name__ == "__main__":
    main()
