"""Download WebQSP (WebQuestionsSP, Microsoft).

Public mirrors of WebQSP are widely available via huggingface datasets and
direct mirrors. We try the rmanluo mirror that packages the full v1 release
with Freebase subgraph dumps as used by RoG/ToG papers.

Usage:
    python -m scripts.download.download_webqsp --root /content/drive/MyDrive/quest_kg/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.download.common import download_file, extract, have_marker, safe_target, write_marker


URLS = [
    # rmanluo's mirror used by recent KGQA papers (RoG, ToG)
    "https://huggingface.co/datasets/rmanluo/RoG-webqsp/resolve/main/data.zip",
    # raw v1 from Microsoft (license-gated; manual fallback)
]


def main(root: str) -> None:
    name = "webqsp"
    if have_marker(root, name):
        print(f"[webqsp] already present at {root}/raw/{name}; skipping.")
        return
    out = safe_target(root, name)
    archive = out / "webqsp.zip"
    last_err = None
    for url in URLS:
        try:
            download_file(url, archive)
            extract(archive, out)
            write_marker(root, name)
            print(f"[webqsp] OK -> {out}")
            return
        except Exception as e:
            print(f"[webqsp] failed from {url}: {e}")
            last_err = e
    raise RuntimeError(
        f"WebQSP download failed: {last_err}\n"
        "Manual fallback: download https://www.microsoft.com/en-us/download/details.aspx?id=52763 "
        "and place the unpacked files under data/raw/webqsp/."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)
