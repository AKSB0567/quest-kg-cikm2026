"""Snapshot current task-aware metrics so iteration-over-iteration deltas
become a one-command diff.

Each invocation reads every `results/*.json`, pulls task_type-appropriate
primary metric, and writes a single row per (method, dataset, primary_metric,
value, em, f1) into `results/_snapshots/<timestamp>.json`. A pointer file
`results/_snapshots/LATEST` keeps the most recent snapshot ID.

Pair with `scripts/compare_snapshots.py` to view a before/after diff.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SNAPS = RESULTS / "_snapshots"


def snapshot(label: str | None = None) -> Path:
    SNAPS.mkdir(parents=True, exist_ok=True)
    rows = []
    for jp in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(jp.read_text())
        except Exception:
            continue
        method = d.get("method") or jp.stem.split("__")[0]
        dataset = d.get("dataset") or jp.stem.split("__")[1]
        rows.append({
            "method": method,
            "dataset": dataset,
            "task_type": d.get("task_type", "qa"),
            "primary_metric": d.get("primary_metric"),
            "primary_value": d.get("primary_value"),
            "exact_match": d.get("exact_match"),
            "token_f1": d.get("token_f1"),
            "balanced_accuracy": d.get("balanced_accuracy"),
            "pos_f1": d.get("pos_f1"),
            "hits_at_1": d.get("hits_at_1"),
            "hits_at_10": d.get("hits_at_10"),
            "mrr": d.get("mrr"),
            "mean_latency_ms": d.get("mean_latency_ms"),
            "abstention_rate": d.get("abstention_rate"),
        })
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_{label or 'snap'}.json"
    out = SNAPS / name
    out.write_text(json.dumps({"label": label, "timestamp": ts, "rows": rows}, indent=2))
    (SNAPS / "LATEST").write_text(name)
    print(f"[snap] wrote {out} ({len(rows)} cells)")
    return out


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else None
    snapshot(label)
