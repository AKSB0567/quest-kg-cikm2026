"""Run all 6 EA datasets locally on the 1080 Ti, sequentially.

Writes live output (including tqdm bars) to results/_ea_run_log.txt so
the user can monitor progress with `tail -f`.

Per-dataset config (informed by prior smoke tests):
  - dbp-yg, dbp-wd : DWY100K (BootEA). DBP-YG cosine already 0.919, reranker
                     hurt -> keep cosine only. DBP-WD is structural-bottlenecked,
                     propagate Wikidata labels via training pairs.
  - ids-d-y, ids-d-w : OpenEA monolingual. Try reranker (untested).
  - ids-en-fr, ids-en-de : OpenEA cross-lingual. Use multilingual encoder.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv312" / "Scripts" / "python.exe"
LOG_PATH = ROOT / "results" / "_ea_run_log.txt"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# (display, root, format, encoder, extra_flags)
DATASETS = [
    ("dbp-yg",
     "data/raw/dwy100k/BootEA/dataset/DWY100K/dbp_yg",
     "dwy100k",
     "sentence-transformers/all-MiniLM-L6-v2",
     []),
    ("ids-d-y",
     "data/raw/openea/OpenEA_dataset_v1.1/D_Y_100K_V2",
     "openea",
     "sentence-transformers/all-MiniLM-L6-v2",
     ["--reranker"]),
    ("ids-d-w",
     "data/raw/openea/OpenEA_dataset_v1.1/D_W_100K_V2",
     "openea",
     "sentence-transformers/all-MiniLM-L6-v2",
     ["--reranker"]),
    ("dbp-wd",
     "data/raw/dwy100k/BootEA/dataset/DWY100K/dbp_wd",
     "dwy100k",
     "sentence-transformers/all-MiniLM-L6-v2",
     ["--propagate"]),
    ("ids-en-fr",
     "data/raw/openea/OpenEA_dataset_v1.1/EN_FR_100K_V2",
     "openea",
     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     ["--reranker"]),
    ("ids-en-de",
     "data/raw/openea/OpenEA_dataset_v1.1/EN_DE_100K_V2",
     "openea",
     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     ["--reranker"]),
]


def main():
    t_global = time.perf_counter()
    with open(LOG_PATH, "w", encoding="utf-8", buffering=1) as log:
        def write(msg: str):
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

        write(f"\n{'#' * 60}")
        write(f"# Local EA orchestrator — 6 datasets, sequential")
        write(f"# Log: {LOG_PATH}")
        write(f"#" * 60)

        results: dict[str, dict] = {}
        for i, (name, path, fmt, encoder, extra) in enumerate(DATASETS, 1):
            write(f"\n{'='*60}")
            write(f"[{i}/{len(DATASETS)}] {name}")
            write(f"  root:    {path}")
            write(f"  format:  {fmt}")
            write(f"  encoder: {encoder}")
            write(f"  extra:   {' '.join(extra) if extra else '(cosine only)'}")
            write(f"{'='*60}")

            cmd = [
                str(PYTHON), "-u", "scripts/ea_runner.py",
                "--dataset", name,
                "--root", path,
                "--format", fmt,
                "--encoder", encoder,
                "--tag", "local-1080ti__sequence__seed0",
            ] + extra

            t0 = time.perf_counter()
            # Stream output to BOTH stdout (so live tqdm visible) AND log file
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

        # Summary
        write(f"\n{'#'*60}")
        write(f"# ALL DONE — total wall: {(time.perf_counter()-t_global)/60:.1f} min")
        write(f"{'#'*60}")
        for name, info in results.items():
            write(f"  {name:<15s} exit={info['exit']:<3d} wall={info['wall_s']/60:>5.1f} min")
        write("\nResults JSONs in results/quest_kg_ea__<dataset>__local-1080ti__sequence__seed0.json")


if __name__ == "__main__":
    main()
