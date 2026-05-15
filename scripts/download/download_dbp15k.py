"""Download DBP15K cross-lingual entity alignment dataset.

DBP15K is the canonical cross-lingual EA benchmark, with three subsets:
  - zh_en (Chinese-English)
  - ja_en (Japanese-English)
  - fr_en (French-English)

Sources (tried in order):
  1. HuggingFace datasets mirrors (multiple guesses; may 404)
  2. Direct file downloads from ZJU-DAILY/RREA or other paper repos
  3. ClusterEA repo (joker-xii/ClusterEA) — proven to mirror DBP15K files

Each subset is treated independently — if zh_en succeeds but ja_en fails, we
mark zh_en as done so the EA story can proceed with whatever subsets are available.

Usage:
    python -m scripts.download.download_dbp15k --root /content/drive/MyDrive/quest_kg/data
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.download.common import safe_target, have_marker


SUBSETS = ["zh_en", "ja_en", "fr_en"]


# Files per subset, common across all DBP15K mirrors
SUBSET_FILES = [
    "ent_ids_1", "ent_ids_2", "ref_ent_ids",
    "rel_ids_1", "rel_ids_2",
    "training_attrs_1", "training_attrs_2",
    "triples_1", "triples_2",
]


# Public GitHub mirrors that host DBP15K with the standard file structure
GITHUB_MIRRORS = [
    "https://raw.githubusercontent.com/joker-xii/ClusterEA/main/data/DBP15K/{subset}/{file}",
    "https://raw.githubusercontent.com/cambridgeltl/sapbert/main/training/data/DBP15K/{subset}/{file}",
    "https://raw.githubusercontent.com/MaybeBian/DivEA/main/data/DBP15K/{subset}/{file}",
]


def subset_complete(subset_dir: Path) -> bool:
    """All required files present and non-empty."""
    for f in SUBSET_FILES:
        p = subset_dir / f
        if not p.exists() or p.stat().st_size == 0:
            return False
    return True


def download_subset_from_github(subset: str, subset_dir: Path) -> bool:
    """Try each GitHub mirror in turn until one succeeds for all files."""
    import requests
    subset_dir.mkdir(parents=True, exist_ok=True)

    for tmpl in GITHUB_MIRRORS:
        success = True
        for fname in SUBSET_FILES:
            url = tmpl.format(subset=subset, file=fname)
            dest = subset_dir / fname
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 0:
                    dest.write_bytes(r.content)
                else:
                    print(f"  [{subset}] {fname}: HTTP {r.status_code} from {tmpl.split('/')[3]}")
                    success = False
                    break
            except Exception as e:
                print(f"  [{subset}] {fname} from {tmpl.split('/')[3]}: {e}")
                success = False
                break
        if success and subset_complete(subset_dir):
            print(f"  [{subset}] downloaded from {tmpl.split('/')[3]}")
            return True
        # else cleanup partial and try next mirror
        for fname in SUBSET_FILES:
            try:
                (subset_dir / fname).unlink(missing_ok=True)
            except Exception:
                pass
    return False


def main(root: str) -> None:
    name = "dbp15k"
    out = safe_target(root, name)

    # Clean up any prior partial state (e.g. half-cached HF datasets)
    cache = out / "_cache"
    if cache.exists():
        print(f"  clearing stale cache {cache}")
        shutil.rmtree(cache, ignore_errors=True)

    results = {}
    for subset in SUBSETS:
        subset_dir = out / subset
        marker = subset_dir / ".done"
        if marker.exists() and subset_complete(subset_dir):
            print(f"[dbp15k/{subset}] already present; skipping.")
            results[subset] = True
            continue

        print(f"[dbp15k/{subset}] downloading...")
        ok = download_subset_from_github(subset, subset_dir)
        if ok:
            marker.write_text("ok\n")
            results[subset] = True
        else:
            results[subset] = False

    n_ok = sum(results.values())
    print(f"\n[dbp15k] SUMMARY: {n_ok}/{len(SUBSETS)} subsets OK -- {results}")
    # Top-level marker if all 3 succeeded
    if n_ok == len(SUBSETS):
        (out / ".done").write_text("ok\n")

    if n_ok == 0:
        raise RuntimeError(
            "DBP15K download failed via all known mirrors.\n"
            "Manual fallback: download from https://github.com/nju-websoft/JAPE/tree/master/data\n"
            f"and place files under {out}/<zh_en|ja_en|fr_en>/"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)
