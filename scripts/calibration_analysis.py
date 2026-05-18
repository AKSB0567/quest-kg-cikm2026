"""Phase 5 calibration: ECE/Brier/NLL/AURC + reliability-diagram data +
risk-coverage curve over a (p*, H*) abstention threshold grid.

Operates on existing per-cell CSVs (no LLM, no GPU). ~30 sec total.

Usage:
    .venv312/Scripts/python scripts/calibration_analysis.py
    .venv312/Scripts/python scripts/calibration_analysis.py --tag <tag>

Outputs:
    results/_calibration_<tag>.csv   per (method, dataset) row with metrics
    results/_rc_curve_<tag>.csv      risk-coverage points for plotting Fig 2
    results/_reliability_<tag>.csv   binned (mean_conf, mean_acc) for Fig 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quest_kg.eval.metrics import aurc, brier, ece, nll  # noqa: E402


def reliability_bins(conf, correct, n_bins=15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        rows.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "n": int(m.sum()),
            "mean_conf": float(conf[m].mean()),
            "mean_acc":  float(correct[m].mean()),
        })
    return rows


def rc_curve(conf, correct):
    """Risk-coverage curve: sort by confidence desc; for each coverage cutoff,
    compute risk = (1 - acc) on the retained set."""
    order = np.argsort(-conf)
    correct_sorted = correct[order]
    cum_correct = np.cumsum(correct_sorted)
    cum_n = np.arange(1, len(order) + 1)
    coverage = cum_n / len(order)
    risk = 1.0 - cum_correct / cum_n
    return coverage, risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="Filter CSVs by tag substring")
    ap.add_argument("--out_suffix", default="", help="Suffix for output filenames")
    args = ap.parse_args()

    results_dir = ROOT / "results"
    pattern = f"*{args.tag}*.csv" if args.tag else "*.csv"
    csvs = sorted(p for p in results_dir.glob(pattern)
                  if not p.name.startswith("_"))
    if not csvs:
        print(f"no CSVs matched pattern {pattern}")
        return

    suffix = args.out_suffix or (args.tag.replace("/", "_") if args.tag else "all")

    cal_rows = []
    rc_rows = []
    rel_rows = []

    for csv_path in csvs:
        parts = csv_path.stem.split("__")
        method = parts[0]
        dataset = parts[1] if len(parts) > 1 else "?"
        tag_part = "__".join(parts[2:]) if len(parts) > 2 else ""

        df = pd.read_csv(csv_path)
        if "confidence" not in df.columns or "em" not in df.columns:
            continue
        conf = df["confidence"].fillna(0.0).clip(0, 1).to_numpy(dtype=float)
        correct = df["em"].fillna(0).astype(int).to_numpy(dtype=float)
        if len(conf) == 0:
            continue

        # Core metrics
        e = ece(conf, correct, n_bins=15)
        b = brier(np.stack([1 - conf, conf], axis=1),
                  np.stack([1 - correct, correct], axis=1))
        # NLL needs proper probabilistic outputs; treat conf as P(correct)
        clipped = np.clip(conf, 1e-9, 1 - 1e-9)
        n_val = float(-np.mean(np.where(correct == 1,
                                         np.log(clipped),
                                         np.log(1 - clipped))))
        a = aurc(conf, correct)

        cal_rows.append({
            "method": method, "dataset": dataset, "tag": tag_part, "n": len(df),
            "ECE": e, "Brier": b, "NLL": n_val, "AURC": a,
            "mean_conf": float(conf.mean()), "accuracy": float(correct.mean()),
        })

        # RC curve
        cov, risk = rc_curve(conf, correct)
        for c, r in zip(cov, risk):
            rc_rows.append({"method": method, "dataset": dataset,
                            "coverage": float(c), "risk": float(r)})

        # Reliability bins
        for row in reliability_bins(conf, correct, n_bins=15):
            row.update({"method": method, "dataset": dataset})
            rel_rows.append(row)

    if not cal_rows:
        print("no rows produced; results may lack confidence/em columns")
        return

    cal_df = pd.DataFrame(cal_rows)
    rc_df = pd.DataFrame(rc_rows)
    rel_df = pd.DataFrame(rel_rows)

    cal_df.to_csv(results_dir / f"_calibration_{suffix}.csv", index=False)
    rc_df.to_csv(results_dir / f"_rc_curve_{suffix}.csv", index=False)
    rel_df.to_csv(results_dir / f"_reliability_{suffix}.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print("\n=== CALIBRATION (lower ECE/AURC = better) ===")
    print(cal_df[["method", "dataset", "n", "ECE", "Brier", "NLL", "AURC",
                  "mean_conf", "accuracy"]].sort_values(
        ["dataset", "ECE"]).to_string(index=False))
    print(f"\nWrote {len(cal_rows)} calibration rows, "
          f"{len(rc_rows)} RC-curve points, {len(rel_rows)} reliability bins")
    print(f"  results/_calibration_{suffix}.csv")
    print(f"  results/_rc_curve_{suffix}.csv")
    print(f"  results/_reliability_{suffix}.csv")


if __name__ == "__main__":
    main()
