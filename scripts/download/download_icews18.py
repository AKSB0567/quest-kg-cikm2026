"""Download ICEWS18 temporal knowledge graph dataset.

Standard splits from RE-Net (Jin et al., 2020) are widely mirrored.

Usage:
    python -m scripts.download.download_icews18 --root /content/drive/MyDrive/quest_kg/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.download.common import download_file, extract, have_marker, safe_target, write_marker


# Standard RE-Net split, mirrored in several public repos
FILE_URLS = {
    "train.txt": "https://raw.githubusercontent.com/INK-USC/RE-Net/master/data/ICEWS18/train.txt",
    "valid.txt": "https://raw.githubusercontent.com/INK-USC/RE-Net/master/data/ICEWS18/valid.txt",
    "test.txt":  "https://raw.githubusercontent.com/INK-USC/RE-Net/master/data/ICEWS18/test.txt",
    "stat.txt":  "https://raw.githubusercontent.com/INK-USC/RE-Net/master/data/ICEWS18/stat.txt",
}


def main(root: str) -> None:
    name = "icews18"
    if have_marker(root, name):
        print(f"[icews18] already present; skipping.")
        return
    out = safe_target(root, name)
    for fname, url in FILE_URLS.items():
        try:
            download_file(url, out / fname)
        except Exception as e:
            print(f"[icews18] failed {fname}: {e}")
            raise
    Path(out / ".done").write_text("ok\n")
    print(f"[icews18] OK -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)
