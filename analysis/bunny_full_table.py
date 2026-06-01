"""Generate full Bunny table (mean +/- std) for all 4 axes x all methods.

Writes results/bunny_full_table.txt and prints to stdout.

Usage:
    python -m analysis.bunny_full_table
    python -m analysis.bunny_full_table --metric rmse
    python -m analysis.bunny_full_table --axis rotation
"""

import argparse
import sys

from analysis.common import (
    BUNNY_AXES, BUNNY_METHODS, DISPLAY, RESULTS,
    load_bunny, mean_std,
)

METRIC_LABEL = {
    "rre":  "RRE (deg)",
    "rte":  "RTE",
    "rmse": "RMSE (x10^3)",
    "time": "Time (s)",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", default="rre", choices=list(METRIC_LABEL))
    p.add_argument("--axis", default=None, choices=list(BUNNY_AXES))
    args = p.parse_args()

    scale = 1000 if args.metric == "rmse" else 1
    axes = [args.axis] if args.axis else list(BUNNY_AXES)

    out_lines = []
    def emit(s=""):
        out_lines.append(s)
        print(s)

    emit(f"Bunny full table -- {METRIC_LABEL[args.metric]} (mean +/- std over 20 trials)")

    for axis in axes:
        col, levels = BUNNY_AXES[axis]
        emit()
        emit("=" * 80)
        emit(f"{axis.upper()}  ({col} sweep)")
        emit("=" * 80)

        header = f"{col:>10}"
        for m in BUNNY_METHODS:
            header += f"  {DISPLAY.get(m, m):>14}"
        emit(header)
        emit("-" * len(header))

        for lvl in levels:
            line = f"{lvl:>10}"
            for m in BUNNY_METHODS:
                vals = load_bunny(axis, m, args.metric).get(lvl, [])
                if not vals:
                    line += f"  {'N/A':>14}"
                else:
                    mu, sd, _ = mean_std(vals)
                    line += f"  {mu * scale:>5.2f} +/- {sd * scale:>5.2f}"
            emit(line)

    out_path = RESULTS / "bunny_full_table.txt"
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nsaved {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
