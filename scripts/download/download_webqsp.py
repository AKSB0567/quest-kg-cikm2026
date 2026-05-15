"""Download WebQSP (Microsoft) via HuggingFace datasets library.

The `rmanluo/RoG-webqsp` HF dataset packages WebQSP with Freebase subgraphs
in the format used by RoG/ToG/GraphRAG papers.

Usage:
    python -m scripts.download.download_webqsp --root /content/drive/MyDrive/quest_kg/data
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.download.common import have_marker, safe_target, write_marker


HF_CANDIDATES = [
    "rmanluo/RoG-webqsp",
    "rmanluo/WebQuestionsSP",
]


def main(root: str) -> None:
    name = "webqsp"
    if have_marker(root, name):
        print(f"[webqsp] already present; skipping.")
        return
    out = safe_target(root, name)

    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
        from datasets import load_dataset

    last_err = None
    for repo_id in HF_CANDIDATES:
        try:
            ds = load_dataset(repo_id, cache_dir=str(out / "_cache"))
            ds.save_to_disk(str(out / "dataset"))
            write_marker(root, name)
            print(f"[webqsp] OK via {repo_id} -> {out}")
            return
        except Exception as e:
            print(f"[webqsp] {repo_id} failed: {e}")
            last_err = e

    raise RuntimeError(
        f"WebQSP download failed via HF: {last_err}\n"
        "Manual fallback: https://www.microsoft.com/en-us/download/details.aspx?id=52763"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)
