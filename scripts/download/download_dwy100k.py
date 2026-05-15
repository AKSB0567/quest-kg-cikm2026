"""Download DWY100K (DBpedia/Wikidata/YAGO) entity alignment dataset.

Source: OpenEA benchmark, https://github.com/nju-websoft/OpenEA
Mirror: the v1.1 release packages DBP15K, DWY100K, etc. as a single archive.

Usage:
    python -m scripts.download.download_dwy100k --root /content/drive/MyDrive/quest_kg/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.download.common import download_file, extract, have_marker, safe_target, write_marker


# OpenEA v1.1 dataset bundle (mirror; check upstream for latest)
URLS = [
    # Primary
    "https://figshare.com/ndownloader/files/27514099",
    # Fallback (HF mirror, if available)
    # add other mirrors as discovered
]


def main(root: str) -> None:
    name = "dwy100k"
    if have_marker(root, name):
        print(f"[dwy100k] already present at {root}/raw/{name}; skipping.")
        return
    out = safe_target(root, name)
    archive = out / "openea_v1.1.zip"
    last_err = None
    for url in URLS:
        try:
            download_file(url, archive)
            extract(archive, out)
            write_marker(root, name)
            print(f"[dwy100k] OK -> {out}")
            return
        except Exception as e:
            print(f"[dwy100k] failed from {url}: {e}")
            last_err = e
    raise RuntimeError(f"DWY100K download failed: {last_err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data root (e.g. /content/drive/MyDrive/quest_kg/data)")
    args = ap.parse_args()
    main(args.root)
