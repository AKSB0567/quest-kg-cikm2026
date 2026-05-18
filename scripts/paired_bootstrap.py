"""Phase 7 statistical validation: paired bootstrap CIs for accuracy/MRR/EM
between QUEST-KG variants and each baseline, per dataset.

Operates on existing per-cell CSVs. Reports 95% CI on (method_A - method_B)
delta per (dataset, primary_metric).

Usage:
    .venv312/Scripts/python scripts/paired_bootstrap.py [--tag <tag>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRIMARY_METRIC_BY_DATASET = {
    "orgaccess": "balanced_accuracy",
    "icews18": "mrr",
    "webqsp": "accuracy",
    "cwq": "accuracy",
}


def per_query_signal(df: pd.DataFrame, metric: str) -> np.ndarray:
    """Return per-query metric value as a numpy array, aligned by qid."""
    df = df.sort_values("qid").reset_index(drop=True)
    if metric in ("accuracy", "exact_match", "EM"):
        return df["em"].fillna(0).astype(int).to_numpy(dtype=float)
    if metric == "balanced_accuracy":
        # bootstrap balanced acc requires per-query gold+pred -> approximate
        # via raw EM; full BAcc CI is reconstructed in the aggregator
        return df["em"].fillna(0).astype(int).to_numpy(dtype=float)
    if metric == "mrr":
        if "x_rank" not in df.columns:
            return np.zeros(len(df))
        r = df["x_rank"].fillna(0).astype(int).to_numpy(dtype=float)
        return np.where(r > 0, 1.0 / np.maximum(r, 1.0), 0.0)
    # token_f1 fallback
    if "f1" in df.columns:
        return df["f1"].fillna(0.0).astype(float).to_numpy()
    return df["em"].fillna(0).astype(int).to_numpy(dtype=float)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 5000,
                     seed: int = 0) -> tuple[float, float, float, float]:
    """Return (delta_mean, lo_95, hi_95, p_two_sided) for a - b."""
    assert len(a) == len(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    delta = float((a - b).mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    # two-sided p: 2 * min(fraction below 0, fraction above 0)
    p_below = float((diffs <= 0).mean())
    p_above = float((diffs >= 0).mean())
    p_two = 2.0 * min(p_below, p_above)
    return delta, lo, hi, p_two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="Filter CSVs by tag substring")
    ap.add_argument("--n_resamples", type=int, default=5000)
    args = ap.parse_args()

    results_dir = ROOT / "results"
    pattern = f"*{args.tag}*.csv" if args.tag else "*.csv"
    csvs = sorted(p for p in results_dir.glob(pattern)
                  if not p.name.startswith("_"))
    if not csvs:
        print("no CSVs matched")
        return

    # Group CSVs by dataset
    by_dataset: dict[str, dict[str, Path]] = {}
    for p in csvs:
        parts = p.stem.split("__")
        if len(parts) < 2:
            continue
        method, dataset = parts[0], parts[1]
        by_dataset.setdefault(dataset, {})[method] = p

    rows = []
    for dataset, method_paths in by_dataset.items():
        primary_metric = PRIMARY_METRIC_BY_DATASET.get(dataset, "accuracy")
        # All methods present in this dataset
        methods = sorted(method_paths)
        if "quest_kg" not in methods and "quest_kg_llm" not in methods:
            continue
        # Compare each QUEST-KG variant against each baseline
        qkg_methods = [m for m in methods if m.startswith("quest_kg")]
        baselines = [m for m in methods if not m.startswith("quest_kg")]
        for qkg in qkg_methods:
            df_q = pd.read_csv(method_paths[qkg])
            sig_q = per_query_signal(df_q, primary_metric)
            for base in baselines:
                df_b = pd.read_csv(method_paths[base])
                if len(df_b) != len(df_q):
                    # Different N -- skip pairing (would be unaligned)
                    continue
                sig_b = per_query_signal(df_b, primary_metric)
                delta, lo, hi, p = paired_bootstrap(
                    sig_q, sig_b, n_resamples=args.n_resamples)
                sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                rows.append({
                    "dataset": dataset, "metric": primary_metric,
                    "ours": qkg, "baseline": base,
                    "ours_mean": float(sig_q.mean()),
                    "base_mean": float(sig_b.mean()),
                    "delta": delta, "ci_lo": lo, "ci_hi": hi,
                    "p_two_sided": p, "signif": sig_str, "n": len(df_q),
                })

    if not rows:
        print("no comparable (matched-N) cells found")
        return

    df = pd.DataFrame(rows)
    suffix = args.tag.replace("/", "_") if args.tag else "all"
    df.to_csv(results_dir / f"_paired_bootstrap_{suffix}.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== PAIRED BOOTSTRAP (95% CI on ours - baseline; positive = ours wins) ===")
    print(df.sort_values(["dataset", "ours", "baseline"]).to_string(index=False))
    print(f"\nWrote {len(rows)} comparisons to results/_paired_bootstrap_{suffix}.csv")


if __name__ == "__main__":
    main()
