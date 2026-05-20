"""Run all 6 EA datasets with merged-KG QUEST-KG (hybrid cosine + structural).

Per-dataset alpha tuned by what we know about the dataset:
  - Strong text labels on both sides  -> alpha=0.8 (cosine dominates)
  - Wikidata K2 (no labels)            -> alpha=0.2 (structural dominates)
  - Cross-lingual                      -> alpha=0.5 (balanced)
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv312" / "Scripts" / "python.exe"
LOG_PATH = ROOT / "results" / "_ea_merged_run_log.txt"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


DATASETS = [
    # (display, root, format, encoder, alpha, extra)
    ("dbp-yg",
     "data/raw/dwy100k/BootEA/dataset/DWY100K/dbp_yg",
     "dwy100k",
     "sentence-transformers/all-MiniLM-L6-v2",
     0.8, []),
    ("dbp-wd",
     "data/raw/dwy100k/BootEA/dataset/DWY100K/dbp_wd",
     "dwy100k",
     "sentence-transformers/all-MiniLM-L6-v2",
     0.2, ["--propagate"]),
    ("ids-d-y",
     "data/raw/openea/OpenEA_dataset_v1.1/D_Y_100K_V2",
     "openea",
     "sentence-transformers/all-MiniLM-L6-v2",
     0.5, []),
    ("ids-d-w",
     "data/raw/openea/OpenEA_dataset_v1.1/D_W_100K_V2",
     "openea",
     "sentence-transformers/all-MiniLM-L6-v2",
     0.2, ["--propagate"]),
    ("ids-en-fr",
     "data/raw/openea/OpenEA_dataset_v1.1/EN_FR_100K_V2",
     "openea",
     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     0.5, []),
    ("ids-en-de",
     "data/raw/openea/OpenEA_dataset_v1.1/EN_DE_100K_V2",
     "openea",
     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     0.5, []),
]


def main():
    t_global = time.perf_counter()
    with open(LOG_PATH, "w", encoding="utf-8", buffering=1) as log:
        def write(msg: str):
            print(msg, flush=True)
            log.write(msg + "\n"); log.flush()
        write("#" * 60)
        write("# Merged-KG EA orchestrator (full test set)")
        write("#" * 60)

        results: dict[str, dict] = {}
        for i, (name, path, fmt, encoder, alpha, extra) in enumerate(DATASETS, 1):
            write(f"\n{'='*60}")
            write(f"[{i}/{len(DATASETS)}] {name}  alpha={alpha}")
            write(f"  encoder: {encoder}")
            write(f"{'='*60}")
            cmd = [str(PYTHON), "-u", "scripts/ea_merged_kg.py",
                   "--dataset", name, "--root", path, "--format", fmt,
                   "--encoder", encoder, "--alpha", str(alpha),
                   "--tag", "local-1080ti__merged-kg__seed0"] + extra
            t0 = time.perf_counter()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, universal_newlines=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            proc.wait()
            elapsed = time.perf_counter() - t0
            results[name] = {"exit": proc.returncode, "wall_s": round(elapsed, 1)}
            write(f"\n  -> exit={proc.returncode}, wall={elapsed/60:.1f} min")

        write(f"\n{'#'*60}")
        write(f"# ALL MERGED-KG EA DONE — total wall: {(time.perf_counter()-t_global)/60:.1f} min")
        write(f"{'#'*60}")
        for name, info in results.items():
            write(f"  {name:<15s} exit={info['exit']:<3d} wall={info['wall_s']/60:>5.1f} min")


if __name__ == "__main__":
    main()
