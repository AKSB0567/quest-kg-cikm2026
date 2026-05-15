"""Sentence encoder wrapper used by retrieval.

Default backend: `sentence-transformers/intfloat-e5-large-v2`.
Falls back to deterministic hash encoder for unit tests / CPU smoke runs.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Union

import numpy as np


class HashEncoder:
    """Deterministic hash-based encoder (testing / fallback).

    Maps any text to a unit-norm vector in R^d using token hashing.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def __call__(self, text: Union[str, list[str]]) -> np.ndarray:
        if isinstance(text, str):
            return self._one(text)
        return np.stack([self._one(t) for t in text])

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in text.lower().split():
            h = hashlib.md5(tok.encode("utf-8")).digest()
            for i, b in enumerate(h[: self.dim]):
                v[i] += b
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class SentenceTransformerEncoder:
    """Wraps sentence-transformers with caching and GPU support.

    Default model: intfloat/e5-large-v2 (1024d). Always uses CUDA if available.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: str | None = None,
        batch_size: int = 64,
        cache_size: int = 100_000,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. `pip install sentence-transformers`"
            ) from e
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}
        self._cache_max = cache_size

    def __call__(self, text: Union[str, list[str]]) -> np.ndarray:
        if isinstance(text, str):
            if text in self._cache:
                return self._cache[text]
            v = self.model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            if len(self._cache) < self._cache_max:
                self._cache[text] = v
            return v
        # Batch case — strip cached, encode the rest
        rest: list[str] = [t for t in text if t not in self._cache]
        if rest:
            embs = self.model.encode(
                rest,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for t, v in zip(rest, embs):
                if len(self._cache) < self._cache_max:
                    self._cache[t] = v
        return np.stack([self._cache[t] for t in text])


def get_encoder(name: str = "e5-large-v2", **kwargs) -> object:
    """Factory: 'hash' for tests, 'e5-large-v2' / 'minilm' for real runs."""
    if name == "hash":
        return HashEncoder(**kwargs)
    if name in ("e5-large-v2", "intfloat/e5-large-v2"):
        return SentenceTransformerEncoder("intfloat/e5-large-v2", **kwargs)
    if name in ("minilm", "all-MiniLM-L12-v2"):
        return SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L12-v2", **kwargs)
    # Pass through any SentenceTransformer-compatible string
    return SentenceTransformerEncoder(name, **kwargs)
