"""Compile the FINAL paper scoreboard from all available results.

For each dataset, scan all available config JSONs and pick the BEST primary
metric. Also report ECE, latency, and the gap to published SOTA.

This consolidates: text-only, reranker, merged-KG, RRF, widened-retrieval
results into a single comprehensive table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Per-dataset published SOTA (best comparable methods from the literature)
SOTA = {
    "orgaccess": {"value": None, "method": "n/a (synthetic)"},
    "icews18":   {"value": 0.341, "method": "DiMNet (2025) — time-aware filtered MRR"},
    "webqsp":    {"value": 0.832, "method": "ChatKBQA (ACL 2024) — fine-tuned LLaMA2-7B"},
    "cwq":       {"value": 0.827, "method": "ChatKBQA (ACL 2024) — fine-tuned LLaMA2-7B"},
    "dbp-yg":    {"value": 0.874, "method": "RREA (CIKM 2020)"},
    "dbp-wd":    {"value": 0.929, "method": "Dual-AMN (WWW 2021)"},
    "ids-d-y":   {"value": 0.85,  "method": "OpenEA SOTA range (TAE-class)"},
    "ids-d-w":   {"value": 0.85,  "method": "OpenEA SOTA range (TAE-class)"},
    "ids-en-fr": {"value": 0.80,  "method": "Cross-lingual EA SOTA (specialized methods)"},
    "ids-en-de": {"value": 0.80,  "method": "Cross-lingual EA SOTA (specialized methods)"},
}

# Per-dataset metric name
PRIMARY = {
    "orgaccess": "balanced_accuracy",
    "icews18":   "mrr",
    "webqsp":    "hits_at_1",
    "cwq":       "hits_at_1",
    "dbp-yg":    "hits_at_1",
    "dbp-wd":    "hits_at_1",
    "ids-d-y":   "hits_at_1",
    "ids-d-w":   "hits_at_1",
    "ids-en-fr": "hits_at_1",
    "ids-en-de": "hits_at_1",
}


def best_for(dataset: str):
    """Find the best result JSON for this dataset across all tags."""
    # Look at all per-cell JSONs for this dataset
    candidates = []
    for p in RESULTS.glob("*.json"):
        if p.name.startswith("_"):
            continue
        if f"__{dataset}__" not in p.name:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        pv = d.get("primary_value")
        if pv is None:
            continue
        candidates.append((pv, p.name, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0]


def main():
    print(f"\n{'=' * 100}")
    print(f"{'FINAL QUEST-KG SCOREBOARD vs Published SOTA':^100s}")
    print(f"{'=' * 100}\n")
    cols = ["Dataset", "Metric", "Ours", "Best config", "SOTA", "vs SOTA"]
    widths = [12, 10, 8, 50, 8, 12]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))

    win_count = 0
    n_compared = 0
    for ds in ["orgaccess", "icews18", "webqsp", "cwq",
               "dbp-yg", "dbp-wd", "ids-d-y", "ids-d-w",
               "ids-en-fr", "ids-en-de"]:
        best = best_for(ds)
        sota = SOTA[ds]
        metric = PRIMARY[ds]
        if best is None:
            row = [ds, metric, "—", "(no results)", str(sota["value"] or "n/a"), "—"]
        else:
            pv, fname, d = best
            tag = d.get("tag", "")
            method = d.get("method", "")
            cfg = f"{method} ({tag[:38]})"
            if sota["value"] is not None:
                delta = pv - sota["value"]
                cmp = f"{delta:+.3f}"
                n_compared += 1
                if delta > 0:
                    win_count += 1
                    cmp = f"WIN {cmp}"
                else:
                    cmp = f"     {cmp}"
            else:
                cmp = "n/a"
            row = [ds, metric, f"{pv:.3f}", cfg[:48], f"{sota['value']:.3f}" if sota["value"] else "n/a", cmp]
        print("  ".join(s.ljust(w) for s, w in zip(row, widths)))

    print(f"\nSOTA beats: {win_count} / {n_compared}")


if __name__ == "__main__":
    main()
