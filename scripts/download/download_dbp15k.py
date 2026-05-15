"""Download DBP15K cross-lingual entity alignment dataset (replaces DWY100K).

DBP15K is the canonical cross-lingual EA benchmark, with three subsets:
  - DBP15K_zh_en (Chinese-English)
  - DBP15K_ja_en (Japanese-English)
  - DBP15K_fr_en (French-English)

Each subset has ~15K entities aligned across two DBpedia language editions.
Used by GCN-Align, RREA, Dual-AMN, ELsEA, ClusterEA, DivEA, LargeEA -- giving
us strong matched-condition baselines for §4.

Source: HuggingFace mirror via the datasets library.

Usage:
    python -m scripts.download.download_dbp15k --root /content/drive/MyDrive/quest_kg/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.download.common import have_marker, safe_target, write_marker


# HF datasets that mirror DBP15K (one of these should work; we try in order)
HF_CANDIDATES = [
    # Each: (HF dataset id, list of subsets/configs we want)
    ("Bryanzhouzh/dbp15k", ["zh_en", "ja_en", "fr_en"]),
    ("yashbonde/DBP15K", ["zh_en", "ja_en", "fr_en"]),
]


def try_hf_datasets(out_dir: Path) -> bool:
    """Try to fetch DBP15K via the HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  installing datasets...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
        from datasets import load_dataset

    for repo_id, subsets in HF_CANDIDATES:
        try:
            for subset in subsets:
                ds = load_dataset(repo_id, subset, cache_dir=str(out_dir / "_cache"))
                ds.save_to_disk(str(out_dir / subset))
                print(f"  saved {subset}")
            return True
        except Exception as e:
            print(f"  HF {repo_id} failed: {e}")
            continue
    return False


def try_github_raw(out_dir: Path) -> bool:
    """Fallback: pull raw splits from a community GitHub mirror."""
    from scripts.download.common import download_file
    # ClusterEA paper hosts the processed splits
    base = "https://raw.githubusercontent.com/joker-xii/ClusterEA/main/data/DBP15K"
    files_per_subset = {
        "zh_en": ["ent_ids_1", "ent_ids_2", "ref_ent_ids", "rel_ids_1", "rel_ids_2",
                  "training_attrs_1", "training_attrs_2", "triples_1", "triples_2"],
        "ja_en": ["ent_ids_1", "ent_ids_2", "ref_ent_ids", "rel_ids_1", "rel_ids_2",
                  "training_attrs_1", "training_attrs_2", "triples_1", "triples_2"],
        "fr_en": ["ent_ids_1", "ent_ids_2", "ref_ent_ids", "rel_ids_1", "rel_ids_2",
                  "training_attrs_1", "training_attrs_2", "triples_1", "triples_2"],
    }
    all_ok = True
    for subset, files in files_per_subset.items():
        (out_dir / subset).mkdir(parents=True, exist_ok=True)
        for fname in files:
            try:
                download_file(f"{base}/{subset}/{fname}", out_dir / subset / fname)
            except Exception as e:
                print(f"  GH raw failed {subset}/{fname}: {e}")
                all_ok = False
    return all_ok


def main(root: str) -> None:
    name = "dbp15k"
    if have_marker(root, name):
        print(f"[dbp15k] already present; skipping.")
        return
    out = safe_target(root, name)

    if try_hf_datasets(out):
        write_marker(root, name)
        print(f"[dbp15k] OK via HF datasets -> {out}")
        return

    print(f"[dbp15k] HF fallback failed, trying GitHub raw...")
    if try_github_raw(out):
        write_marker(root, name)
        print(f"[dbp15k] OK via GitHub raw -> {out}")
        return

    raise RuntimeError(
        "DBP15K download failed via all known mirrors.\n"
        "Manual fallback: download from https://github.com/nju-websoft/JAPE\n"
        f"and place under {out}/<zh_en|ja_en|fr_en>/"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)
