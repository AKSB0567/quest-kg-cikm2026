"""Orchestrator: download all CIKM datasets in sequence.

Usage:
    python -m scripts.download.download_all --root /content/drive/MyDrive/quest_kg/data
"""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DATASETS = [
    ("download_dwy100k", {}),
    ("download_webqsp", {}),
    ("download_cwq", {}),
    ("download_icews18", {}),
    # IDS100K shares packaging with DWY100K in OpenEA bundle; handled by dwy100k script.
]


def main(root: str) -> None:
    failed = []
    for mod_name, kw in DATASETS:
        print("=" * 60)
        print(f"[orchestrator] running {mod_name}")
        try:
            mod = importlib.import_module(f"scripts.download.{mod_name}")
            mod.main(root=root, **kw)
        except Exception as e:
            print(f"[orchestrator] {mod_name} failed: {e}")
            traceback.print_exc()
            failed.append(mod_name)

    # OrgAccess (synthetic)
    print("=" * 60)
    print("[orchestrator] generating orgaccess (synthetic)")
    try:
        mod = importlib.import_module("scripts.download.generate_orgaccess")
        mod.main(root=root, n_users=500, n_resources=200, n_policies=50,
                 n_contexts=8, n_queries=5000, T_horizon=200, seed=0)
    except Exception as e:
        print(f"[orchestrator] orgaccess failed: {e}")
        traceback.print_exc()
        failed.append("orgaccess")

    print("=" * 60)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("All datasets present.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)
