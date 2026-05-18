"""Print the list of results/*.json files that need to be deleted to force
a rerun of cells affected by recent code changes.

After Iteration 1b/1d/1e we changed:
  - QUEST-KG inference (MID-aware QA tail extraction + candidate ranking)
  - OrgAccess loader (class-balanced interleave)

This means:
  - All orgaccess cells need rerun: balanced split changes which queries
    are evaluated.
  - quest_kg cells on webqsp / icews18 need rerun: new prediction path +
    ranks.

Cells that DO NOT need rerun:
  - Baselines on webqsp/icews18: code unchanged.
  - quest_kg/cwq: if it runs after this script's commit time, it gets the
    new code automatically (resumable runner spawns a fresh Python).

Usage:
    .venv312/Scripts/python scripts/select_cells_to_rerun.py [--delete]

Without --delete the script prints the file list. With --delete, files are
removed (CSV + JSON together).
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def candidate_files() -> list[Path]:
    paths: list[Path] = []
    # All orgaccess cells (balanced interleave changes the eval queries)
    paths += list(RESULTS.glob("*__orgaccess__*.json"))
    # quest_kg on webqsp (MID-aware prediction)
    paths += list(RESULTS.glob("quest_kg__webqsp__*.json"))
    # All icews18 cells (need MRR + rank emission)
    paths += list(RESULTS.glob("*__icews18__*.json"))
    return sorted(set(paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="actually delete the files")
    args = ap.parse_args()

    files = candidate_files()
    if not files:
        print("[select] no candidate files found")
        return
    print(f"[select] {len(files)} cells will be re-run on next run_all_local.py:")
    for f in files:
        print(f"  {f.name}")
        csv = f.with_suffix(".csv")
        if csv.exists():
            print(f"  {csv.name}")
    if args.delete:
        for f in files:
            f.unlink(missing_ok=True)
            f.with_suffix(".csv").unlink(missing_ok=True)
        print(f"[select] deleted {len(files)} JSON + matching CSV files")
    else:
        print("\n[select] dry-run: pass --delete to actually remove these.")


if __name__ == "__main__":
    main()
