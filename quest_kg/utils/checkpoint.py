"""Colab Pro–friendly checkpoint/resume utilities.

Colab Pro sessions disconnect without warning and don't have background execution.
Every long-running experiment must:
  1. Persist intermediate state every N items to Drive (atomic write).
  2. On restart, detect existing checkpoint and resume from where it left off.

Usage:
    with ResumableRun("/content/drive/MyDrive/quest_kg/ckpts/headline_webqsp_seed0.pkl",
                     total=len(queries), every=50) as run:
        for i, q in run.iter(queries):
            result = process(q)
            run.append(result)
"""
from __future__ import annotations

import os
import pickle
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


def atomic_save(obj: Any, path: str | os.PathLike) -> None:
    """Write `obj` to `path` atomically (temp file + rename).

    Atomic rename guarantees we never end up with a half-written checkpoint
    if Colab kills the session mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.move(tmp_path, str(path))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


@dataclass
class _State:
    next_idx: int = 0
    results: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class ResumableRun:
    """Context manager for checkpoint-resumable iteration.

    Args:
        ckpt_path: path on Drive where state is persisted.
        total: total number of items expected (for tqdm and bookkeeping).
        every: write checkpoint every N processed items.
    """

    def __init__(self, ckpt_path: str | os.PathLike, total: int, every: int = 50):
        self.ckpt_path = Path(ckpt_path)
        self.total = total
        self.every = max(1, every)
        self.state: _State

    def __enter__(self) -> "ResumableRun":
        if self.ckpt_path.exists():
            with open(self.ckpt_path, "rb") as f:
                self.state = pickle.load(f)
            print(f"[ResumableRun] resumed from {self.ckpt_path} at idx={self.state.next_idx}/{self.total}")
        else:
            self.state = _State()
            print(f"[ResumableRun] fresh run at {self.ckpt_path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Flush final state
        atomic_save(self.state, self.ckpt_path)
        if exc is None:
            print(f"[ResumableRun] completed {self.state.next_idx}/{self.total}")

    def iter(self, items: Iterable[Any]) -> Iterator[tuple[int, Any]]:
        items = list(items)
        for i, item in enumerate(items):
            if i < self.state.next_idx:
                continue
            yield i, item
            self.state.next_idx = i + 1
            if (i + 1) % self.every == 0:
                atomic_save(self.state, self.ckpt_path)

    def append(self, result: Any) -> None:
        self.state.results.append(result)

    @property
    def results(self) -> list:
        return self.state.results


@contextmanager
def anti_idle():
    """No-op placeholder; the real anti-idle hack lives in the Colab JS console.

    See notebooks/00_setup.ipynb for the browser-side script that prevents
    Colab Pro sessions from idling out.
    """
    yield
