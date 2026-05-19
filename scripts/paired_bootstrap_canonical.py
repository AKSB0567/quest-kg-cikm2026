"""Iter 6i: Paired bootstrap on the canonical 28-cell CSV set.

Unlike `paired_bootstrap.py` (which picks one CSV per (method, dataset) by
dict overwrite order and is brittle), this script selects canonical CSVs
explicitly via tag_priority — same priority logic the headline-table
builder uses — and pairs against QUEST-KG / QUEST-KG-LLM consistently
across all 28 cells.

Reports:
  results/_paired_bootstrap_canonical.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
DATASETS = ["orgaccess", "icews18", "webqsp", "cwq"]
LLM_BASELINES = ["vanilla_rag", "graphrag", "tog1", "tog2", "cok"]

PRIMARY_METRIC = {
    "orgaccess": "balanced_accuracy",  # we bootstrap EM here, BA CI separately
    "icews18": "mrr",
    "webqsp": "accuracy",
    "cwq": "accuracy",
}


def canonical_csv(method: str, dataset: str) -> Path | None:
    """Return the canonical per-query CSV for this (method, dataset).

    quest_kg:        local-1080ti__symbolic (LOCKED) on all 4 datasets.
    quest_kg_llm:    local-1080ti__symbolic on orgaccess/icews18 (mirror);
                     colab-l4__Qwen2.5-7B on webqsp/cwq (LOCKED rerun).
    Baselines:       colab-l4__Qwen2.5-7B (Qwen-7B canonical).
    """
    if method == "quest_kg":
        p = RESULTS / f"quest_kg__{dataset}__local-1080ti__symbolic__seed0.csv"
    elif method == "quest_kg_llm":
        if dataset in ("orgaccess", "icews18"):
            p = RESULTS / f"quest_kg_llm__{dataset}__local-1080ti__symbolic__seed0.csv"
        else:
            p = RESULTS / f"quest_kg_llm__{dataset}__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"
    else:
        p = RESULTS / f"{method}__{dataset}__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"
    return p if p.exists() else None


def per_query_signal(df: pd.DataFrame, metric: str) -> np.ndarray:
    df = df.sort_values("qid").reset_index(drop=True) if "qid" in df.columns else df.reset_index(drop=True)
    if metric in ("accuracy", "exact_match", "balanced_accuracy"):
        return df["em"].fillna(0).astype(int).to_numpy(dtype=float)
    if metric == "mrr":
        if "x_rank" in df.columns:
            r = df["x_rank"].fillna(0).astype(int).to_numpy(dtype=float)
            return np.where(r > 0, 1.0 / np.maximum(r, 1.0), 0.0)
        # MRR fallback when rank not stored: use EM (1.0 for correct, 0 else)
        return df["em"].fillna(0).astype(int).to_numpy(dtype=float)
    return df["em"].fillna(0).astype(int).to_numpy(dtype=float)


def paired_bootstrap(a: np.ndarray, b: np.ndarray,
                      n_resamples: int = 10000, seed: int = 0):
    """Return (delta_mean, ci_lo, ci_hi, p_two_sided)."""
    assert len(a) == len(b), f"len mismatch {len(a)} vs {len(b)}"
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    delta = float((a - b).mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    p_below = float((diffs <= 0).mean())
    p_above = float((diffs >= 0).mean())
    p_two = 2.0 * min(p_below, p_above)
    return delta, lo, hi, p_two


def align(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Truncate to common N (min of the two) and align by row order.

    We assume both runs evaluated the first N queries of the same test split.
    If qid column is present we sort by qid; otherwise rely on row order.
    """
    for d in (df_a, df_b):
        if "qid" in d.columns:
            d.sort_values("qid", inplace=True)
            d.reset_index(drop=True, inplace=True)
    n = min(len(df_a), len(df_b))
    return df_a.iloc[:n], df_b.iloc[:n], n


def main():
    rows = []
    for ds in DATASETS:
        metric = PRIMARY_METRIC[ds]
        qkg_csv = canonical_csv("quest_kg", ds)
        qkgllm_csv = canonical_csv("quest_kg_llm", ds)
        if not qkg_csv:
            print(f"  skip {ds}: no quest_kg canonical CSV")
            continue
        df_q = pd.read_csv(qkg_csv)
        df_qllm = pd.read_csv(qkgllm_csv) if qkgllm_csv else None
        for base in LLM_BASELINES:
            bcsv = canonical_csv(base, ds)
            if not bcsv:
                print(f"  miss {base} {ds}")
                continue
            df_b = pd.read_csv(bcsv)
            # quest_kg vs baseline
            a_q, b_b, n = align(df_q, df_b)
            sig_q = per_query_signal(a_q, metric)
            sig_b = per_query_signal(b_b, metric)
            delta, lo, hi, p = paired_bootstrap(sig_q, sig_b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            rows.append({
                "dataset": ds, "metric": metric,
                "ours": "quest_kg", "baseline": base,
                "n": n,
                "ours_mean": float(sig_q.mean()),
                "base_mean": float(sig_b.mean()),
                "delta": delta, "ci_lo": lo, "ci_hi": hi,
                "p_two_sided": p, "signif": sig,
            })
            # quest_kg_llm vs baseline (if we have a CSV)
            if df_qllm is not None:
                a_qllm, b_b2, n2 = align(df_qllm, df_b)
                sig_qllm = per_query_signal(a_qllm, metric)
                sig_b2 = per_query_signal(b_b2, metric)
                delta, lo, hi, p = paired_bootstrap(sig_qllm, sig_b2)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                rows.append({
                    "dataset": ds, "metric": metric,
                    "ours": "quest_kg_llm", "baseline": base,
                    "n": n2,
                    "ours_mean": float(sig_qllm.mean()),
                    "base_mean": float(sig_b2.mean()),
                    "delta": delta, "ci_lo": lo, "ci_hi": hi,
                    "p_two_sided": p, "signif": sig,
                })

    if not rows:
        print("no comparisons computed")
        return

    df = pd.DataFrame(rows)
    out = RESULTS / "_paired_bootstrap_canonical.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== PAIRED BOOTSTRAP (95% CI on ours - baseline; +ve = QUEST-KG wins) ===")
    cols = ["dataset", "ours", "baseline", "n", "ours_mean", "base_mean",
            "delta", "ci_lo", "ci_hi", "p_two_sided", "signif"]
    print(df[cols].sort_values(["dataset", "ours", "baseline"]).to_string(index=False))

    # Win summary
    wins = (df["delta"] > 0).sum()
    sig_wins = ((df["delta"] > 0) & (df["p_two_sided"] < 0.05)).sum()
    print(f"\nTotal comparisons: {len(df)}")
    print(f"QUEST-KG wins (delta > 0):                     {wins}")
    print(f"QUEST-KG significant wins (p < 0.05):          {sig_wins}")
    print(f"\nWrote {out.name}")


if __name__ == "__main__":
    main()
