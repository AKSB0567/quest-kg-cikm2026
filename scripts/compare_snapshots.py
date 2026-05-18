"""Compare two snapshots written by `snapshot_results.py`.

Usage:
    .venv312/Scripts/python scripts/compare_snapshots.py <before>.json <after>.json
    .venv312/Scripts/python scripts/compare_snapshots.py LATEST          # before = LATEST, after = LATEST
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPS = ROOT / "results" / "_snapshots"


def load(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = SNAPS / name_or_path
    if p.is_dir() or not p.exists():
        raise FileNotFoundError(name_or_path)
    return json.loads(p.read_text())


def key(row: dict) -> tuple[str, str]:
    return (row.get("method", ""), row.get("dataset", ""))


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    before_arg = sys.argv[1]
    after_arg = sys.argv[2] if len(sys.argv) > 2 else "LATEST"
    if before_arg == "LATEST":
        latest = (SNAPS / "LATEST").read_text().strip()
        before = load(latest)
        after = before
    else:
        before = load(before_arg)
        after = load(after_arg)
    b_rows = {key(r): r for r in before["rows"]}
    a_rows = {key(r): r for r in after["rows"]}
    all_keys = sorted(set(b_rows) | set(a_rows))

    print(f"{'method':<14s} {'dataset':<11s}  {'metric':<22s}  {'before':>8s}  {'after':>8s}  {'delta':>8s}")
    print("-" * 78)
    for k in all_keys:
        b, a = b_rows.get(k), a_rows.get(k)
        method, dataset = k
        m_name = (a or b).get("primary_metric") or "primary"
        b_val = (b or {}).get("primary_value")
        a_val = (a or {}).get("primary_value")
        delta = (a_val - b_val) if (a_val is not None and b_val is not None) else None
        print(f"{method:<14s} {dataset:<11s}  {m_name:<22s}  {fmt(b_val):>8s}  {fmt(a_val):>8s}  {fmt(delta):>8s}")


if __name__ == "__main__":
    main()
