"""Download ComplexWebQuestions (CWQ) v1.1.

Sources:
  - HuggingFace mirror via rmanluo (used in RoG/ToG)
  - TAU-NLP: https://www.tau-nlp.org/compwebq

Usage:
    python -m scripts.download.download_cwq --root /content/drive/MyDrive/quest_kg/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.download.common import download_file, extract, have_marker, safe_target, write_marker


URLS = [
    "https://huggingface.co/datasets/rmanluo/RoG-cwq/resolve/main/data.zip",
]


def main(root: str) -> None:
    name = "cwq"
    if have_marker(root, name):
        print(f"[cwq] already present at {root}/raw/{name}; skipping.")
        return
    out = safe_target(root, name)
    archive = out / "cwq.zip"
    last_err = None
    for url in URLS:
        try:
            download_file(url, archive)
            extract(archive, out)
            write_marker(root, name)
            print(f"[cwq] OK -> {out}")
            return
        except Exception as e:
            print(f"[cwq] failed from {url}: {e}")
            last_err = e
    raise RuntimeError(
        f"CWQ download failed: {last_err}\n"
        "Manual fallback: download from https://www.tau-nlp.org/compwebq and unpack to data/raw/cwq/."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)
