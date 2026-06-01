"""Shared utilities for analysis scripts.

Reads result CSVs from `results/` and provides loaders + statistics helpers.
Pure stdlib — no pandas/numpy dependency, so analysis runs anywhere Python 3.9+
is installed without needing the full experiment environment.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
BUNNY = RESULTS / "bunny"
MODELNET = RESULTS / "modelnet"

BUNNY_AXES = {
    "noise":    ("sigma",   ["0.01", "0.02", "0.03", "0.04", "0.05"]),
    "outlier":  ("ratio",   ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]),
    "overlap":  ("overlap", ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]),
    "rotation": ("angle",   ["10", "20", "30", "40", "50", "60", "70", "80", "90"]),
}

# Bunny ships two Sinkhorn-CPD variants (tau_y = 0.1 robustness mode and
# tau_y = 1.0 balanced-OT sensitivity).  ModelNet uses the single tau=1.0 run.
BUNNY_METHODS = [
    "TEASER++", "FGR", "Sparse-ICP", "CPD", "FilterReg",
    "LSG-CPD", "RPOT", "RobOT",
    "SinkhornCPD_tau0.1", "SinkhornCPD_tau1.0",
]
MODELNET_METHODS = [
    "TEASER++", "FGR", "Sparse-ICP", "CPD", "FilterReg",
    "LSG-CPD", "RPOT", "RobOT", "SinkhornCPD",
]
DISPLAY = {
    "SinkhornCPD_tau0.1": "Ours(tau=0.1)",
    "SinkhornCPD_tau1.0": "Ours(tau=1.0)",
    "SinkhornCPD": "Ours",
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_bunny(axis, method, metric="rre"):
    """Load Bunny CSV and group `metric` values by perturbation level."""
    col = BUNNY_AXES[axis][0]
    path = BUNNY / axis / f"{method}.csv"
    if not path.exists():
        return {}
    groups = defaultdict(list)
    for r in load_csv(path):
        groups[r[col].strip()].append(float(r[metric]))
    return dict(groups)


def load_flat(benchmark_dir, method, metric="rre"):
    """Load a single-file benchmark CSV (ModelNet-style) into a list of floats."""
    path = benchmark_dir / f"{method}.csv"
    if not path.exists():
        return []
    return [float(r[metric]) for r in load_csv(path) if metric in r]


def mean_std(vals):
    """Return (mean, sample_std, n).  Empty input -> (nan, nan, 0)."""
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    m = sum(vals) / n
    if n == 1:
        return (m, 0.0, 1)
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return (m, math.sqrt(var), n)
