"""Wait for local quest_kg-only CSVs to land, then run ablations.
Convenience wrapper so we can leave the 1080 Ti alone overnight.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

EXPECTED = [
    "quest_kg__orgaccess__local-1080ti__symbolic__seed0.json",
    "quest_kg__icews18__local-1080ti__symbolic__seed0.json",
    "quest_kg__webqsp__local-1080ti__symbolic__seed0.json",
    "quest_kg__cwq__local-1080ti__symbolic__seed0.json",
]


def main():
    print("[wait] watching for symbolic quest_kg JSONs ...")
    last_count = -1
    for _ in range(7200):  # max ~2 hours
        existing = sum(1 for f in EXPECTED if (RESULTS / f).exists())
        if existing != last_count:
            print(f"[wait] {existing}/{len(EXPECTED)} JSONs present")
            last_count = existing
        if existing == len(EXPECTED):
            break
        time.sleep(5)
    else:
        print("[wait] timeout, proceeding with what's available")

    print("\n[wait] launching ablations ...")
    r = subprocess.run([sys.executable, "scripts/local_ablations.py"],
                       cwd=str(ROOT))
    print(f"[wait] ablations exit code {r.returncode}")

    print("\n[wait] launching calibration analysis ...")
    r = subprocess.run([sys.executable, "scripts/calibration_analysis.py",
                        "--tag", "local-1080ti__symbolic"], cwd=str(ROOT))
    print(f"[wait] calibration exit code {r.returncode}")


if __name__ == "__main__":
    main()
