"""Re-aggregate finished result CSVs into task-aware summaries.

After Iter 1a we added balanced_accuracy + pos-F1 for OrgAccess, Hits@k + MRR for
ICEWS18, and kept EM/F1 for WebQSP/CWQ. The overnight run wrote per-cell summaries
under the old (EM-only) aggregator. Rather than re-running 50 queries × 24 cells,
this script reads every results/*.csv and rewrites the matching results/*.json
with the new task-aware fields.

Idempotent: running it again is fine; it only writes the JSON.

Usage:
    .venv312/Scripts/python scripts/reaggregate_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from quest_kg.eval.harness import QueryResult, aggregate  # noqa: E402


TASK_TYPE = {
    "orgaccess": "access_control",
    "icews18": "link_prediction",
    "webqsp": "qa",
    "cwq": "qa",
}


def parse_tag(csv_path: Path) -> tuple[str, str, str]:
    # filename: <method>__<dataset>__<tag>.csv
    stem = csv_path.stem
    parts = stem.split("__", 2)
    method, dataset = parts[0], parts[1]
    tag = parts[2] if len(parts) > 2 else ""
    return method, dataset, tag


def csv_to_results(df: pd.DataFrame) -> list[QueryResult]:
    out = []
    for _, row in df.iterrows():
        extra = {}
        for col in df.columns:
            if col.startswith("x_"):
                extra[col[2:]] = row[col]
        out.append(QueryResult(
            qid=str(row["qid"]),
            question=str(row["question"]),
            gold=str(row["gold"]),
            prediction=(None if pd.isna(row["prediction"]) else str(row["prediction"])),
            em=int(row["em"]),
            f1=float(row["f1"]),
            confidence=float(row["confidence"]),
            latency_ms=float(row["latency_ms"]),
            abstained=bool(row["abstained"]),
            extra=extra,
        ))
    return out


def main():
    results_dir = ROOT / "results"
    csvs = sorted(p for p in results_dir.glob("*.csv") if not p.name.startswith("_"))
    if not csvs:
        print(f"[reagg] no CSVs in {results_dir}")
        return
    print(f"[reagg] found {len(csvs)} CSVs")
    written = 0
    for csv_path in csvs:
        method, dataset, tag = parse_tag(csv_path)
        task_type = TASK_TYPE.get(dataset, "qa")
        df = pd.read_csv(csv_path)
        results = csv_to_results(df)
        summary = aggregate(results, task_type=task_type)
        # Preserve any prior method/dataset/llm/seed/wallclock fields
        json_path = csv_path.with_suffix(".json")
        prior = {}
        if json_path.exists():
            try:
                prior = json.loads(json_path.read_text())
            except Exception:
                prior = {}
        merged = {**prior, **summary, "method": method, "dataset": dataset, "tag": tag}
        json_path.write_text(json.dumps(merged, indent=2))
        written += 1
        pv = summary.get("primary_value", 0.0)
        pm = summary.get("primary_metric", "?")
        print(f"  {method:>12s} / {dataset:<10s}  EM={summary['exact_match']:.3f}  "
              f"{pm}={pv:.3f}")
    print(f"[reagg] wrote {written} summaries")


if __name__ == "__main__":
    main()
