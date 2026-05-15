"""Orchestrator: download all CIKM datasets.

Continues even if a single dataset fails -- prints a final summary so you can
re-run individual ones with manual fallbacks.

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


# (module_name, kwargs, required?)
DATASETS = [
    ("download_dbp15k",   {}, True),   # replaces DWY100K
    ("download_webqsp",   {}, True),
    ("download_cwq",      {}, True),
    ("download_icews18",  {}, False),  # optional in main paper, useful in appendix
]


def main(root: str) -> None:
    ok, failed = [], []
    for mod_name, kw, _required in DATASETS:
        print("=" * 60)
        print(f"[orchestrator] running {mod_name}")
        try:
            mod = importlib.import_module(f"scripts.download.{mod_name}")
            mod.main(root=root, **kw)
            ok.append(mod_name)
        except Exception as e:
            print(f"[orchestrator] {mod_name} failed: {e}")
            traceback.print_exc()
            failed.append(mod_name)

    # OrgAccess (synthetic, always works)
    print("=" * 60)
    print("[orchestrator] generating orgaccess (synthetic)")
    try:
        mod = importlib.import_module("scripts.download.generate_orgaccess")
        mod.main(root=root, n_users=500, n_resources=200, n_policies=50,
                 n_contexts=8, n_queries=5000, T_horizon=200, seed=0)
        ok.append("orgaccess")
    except Exception as e:
        print(f"[orchestrator] orgaccess failed: {e}")
        traceback.print_exc()
        failed.append("orgaccess")

    print("=" * 60)
    print("SUMMARY")
    print(f"  OK    : {ok}")
    print(f"  FAILED: {failed}")
    print("=" * 60)
    if failed:
        print(f"\nNote: orchestrator continues past individual failures so re-runs only fetch what's missing.")
        print(f"      For each failed dataset, try the script directly to see detailed errors.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)
