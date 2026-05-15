"""Shared helpers for dataset download scripts.

Designed to run in Colab Pro: idempotent (skip if already present), atomic writes,
resumable downloads.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def download_file(url: str, dest: str | os.PathLike, chunk: int = 1 << 20, resume: bool = True) -> Path:
    """Stream a URL to disk with progress bar; resume if partial file exists."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    existing = dest.stat().st_size if (resume and dest.exists()) else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code in (200, 206):
            total = int(r.headers.get("Content-Length", 0)) + existing
            mode = "ab" if (existing and r.status_code == 206) else "wb"
            with open(dest, mode) as f, tqdm(
                total=total, initial=existing, unit="B", unit_scale=True, desc=dest.name
            ) as bar:
                for blk in r.iter_content(chunk_size=chunk):
                    if blk:
                        f.write(blk)
                        bar.update(len(blk))
        else:
            r.raise_for_status()
    return dest


def sha256_of(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def extract(archive: str | os.PathLike, out_dir: str | os.PathLike) -> Path:
    """Extract a .zip / .tar / .tar.gz to `out_dir`."""
    archive = Path(archive)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(out_dir)
    elif name.endswith(".tar"):
        with tarfile.open(archive, "r") as t:
            t.extractall(out_dir)
    else:
        raise ValueError(f"unsupported archive type: {archive}")
    return out_dir


def safe_target(root: str | os.PathLike, dataset_name: str) -> Path:
    p = Path(root) / "raw" / dataset_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def have_marker(root: str | os.PathLike, dataset_name: str) -> bool:
    """Check the .done marker file written after successful extraction."""
    return (Path(root) / "raw" / dataset_name / ".done").exists()


def write_marker(root: str | os.PathLike, dataset_name: str) -> None:
    (Path(root) / "raw" / dataset_name / ".done").write_text("ok\n")
