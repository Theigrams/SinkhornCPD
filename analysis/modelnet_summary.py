"""Summarize ModelNet40 results (paper Table 5).

Writes results/modelnet_summary.txt and prints to stdout.

RR (Registration Recall) = fraction of pairs with RE < 1 deg AND TE < 0.1.

Usage:
    python -m analysis.modelnet_summary
"""

import sys

from analysis.common import (
    MODELNET, MODELNET_METHODS, DISPLAY, RESULTS,
    load_csv, mean_std,
)

RR_RE_THRESH = 1.0   # degrees
RR_TE_THRESH = 0.1   # unit-ball Euclidean


def main():
    out_lines = []
    def emit(s=""):
        out_lines.append(s)
        print(s)

    emit(f"ModelNet40 Summary  (RR = RE < {RR_RE_THRESH} deg AND TE < {RR_TE_THRESH})")
    emit("=" * 92)
    emit(f"{'Method':<14} {'RE (deg)':>16} {'TE (x10^3)':>16} {'RMSE (x10^3)':>16} {'RR (%)':>8} {'Time (s)':>10}")
    emit("-" * 92)

    for method in MODELNET_METHODS:
        path = MODELNET / f"{method}.csv"
        label = DISPLAY.get(method, method)
        if not path.exists():
            emit(f"{label:<14} {'N/A':>16}")
            continue

        rows = load_csv(path)
        re_vals   = [float(r["rre"])  for r in rows]
        te_vals   = [float(r["rte"])  for r in rows]
        rmse_vals = [float(r["rmse"]) for r in rows]
        time_vals = [float(r["time"]) for r in rows]

        m_re,   sd_re,   _ = mean_std(re_vals)
        m_te,   sd_te,   _ = mean_std(te_vals)
        m_rmse, sd_rmse, _ = mean_std(rmse_vals)
        m_time, _,       _ = mean_std(time_vals)
        rr = sum(1 for re, te in zip(re_vals, te_vals)
                 if re < RR_RE_THRESH and te < RR_TE_THRESH) / len(re_vals) * 100

        emit(f"{label:<14} "
             f"{m_re:>6.2f} +/- {sd_re:>5.2f} "
             f"{m_te*1000:>6.1f} +/- {sd_te*1000:>4.1f} "
             f"{m_rmse*1000:>6.1f} +/- {sd_rmse*1000:>4.1f} "
             f"{rr:>7.1f} {m_time:>10.2f}")

    out_path = RESULTS / "modelnet_summary.txt"
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nsaved {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
