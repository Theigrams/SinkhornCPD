"""Bunny single-factor robustness benchmark (paper §5.3).

Usage:
    python -m experiments.bunny --axis noise   --method SinkhornCPD --trials 2
    python -m experiments.bunny --axis outlier --method all
"""

import argparse

from baselines import METHODS
from sinkhorn_cpd import sinkhorn_cpd
from datasets.bunny_synth import load_synth

from experiments._common import RESULTS_DIR, run_method, save_csv

LEVELS = {
    "noise": ([0.01, 0.02, 0.03, 0.04, 0.05], "sigma"),
    "outlier": ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "ratio"),
    "overlap": ([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "overlap"),
    "rotation": ([10, 20, 30, 40, 50, 60, 70, 80, 90], "angle"),
}


TAU_Y_BY_AXIS = {"noise": 0.1, "outlier": 0.1, "overlap": 0.1, "rotation": 0.1}
# TAU_Y_BY_AXIS = {"noise": 1.0, "outlier": 1.0, "overlap": 1.0, "rotation": 1.0}


def _scpd_factory(tau_y):
    def _scpd(source, target):
        R, t, _ = sinkhorn_cpd(source, target, tau_y=tau_y)
        return R, t

    return _scpd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True, choices=list(LEVELS))
    p.add_argument("--method", required=True, help="method name or 'all'")
    p.add_argument("--trials", type=int, default=None, help="number of trials per level (default: all 20)")
    args = p.parse_args()

    methods_plus = {**METHODS, "SinkhornCPD": _scpd_factory(TAU_Y_BY_AXIS[args.axis])}
    method_names = list(methods_plus) if args.method == "all" else [args.method]
    levels, level_col = LEVELS[args.axis]

    for name in method_names:
        fn = methods_plus[name]
        if fn is None:
            print(f"[skip] {name}: backend not available")
            continue
        print(f"\n=== {args.axis} / {name} ===")
        out = RESULTS_DIR / "bunny" / args.axis / f"{name}.csv"
        for level in levels:
            source, targets, R_gts, t_gts = load_synth(args.axis, level)
            n = len(targets) if args.trials is None else min(args.trials, len(targets))
            pairs = list(zip(targets[:n], R_gts[:n], t_gts[:n]))
            rows = run_method(fn, source, pairs, desc=f"  {level_col}={level}")
            save_csv(out, rows, level_col=level_col, level=level)
            mean_rre = sum(r["rre"] for r in rows if r["rre"] == r["rre"]) / max(1, n)
            print(f"  {level_col}={level}: mean RRE={mean_rre:.3f}°")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
