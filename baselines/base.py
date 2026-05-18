"""Common interface for baselines.

Every baseline takes a query, optionally a query-specific KG slice / retrieval
budget, and returns a BaselinePrediction including a confidence (used for
matched-condition calibration comparison).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

from quest_kg.core.types import Triple


@dataclass
class BaselinePrediction:
    query: str
    prediction: str | None
    confidence: float = 1.0       # for calibration metrics
    abstained: bool = False
    evidence: list = field(default_factory=list)  # supporting triples / spans
    extra: dict = field(default_factory=dict)


class Baseline(Protocol):
    """All baselines implement this minimal interface."""

    name: str

    def predict(self, query: str, **kwargs) -> BaselinePrediction: ...


def ranking_from_triples(triples: Iterable[Triple], max_k: int = 64) -> list[str]:
    """Build a per-baseline candidate ranking from retrieved triples.

    The ranking is the deduplicated tail (`o`) order of the retrieved triples.
    Used by the eval harness to compute Hits@k / MRR for link-prediction tasks
    (e.g. ICEWS18 — gold is a tail entity index given head+rel+ts). Not as
    informative as QUEST-KG's path-scored ranking, but consistent across all
    baselines that retrieve a candidate set.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in triples:
        tail = str(t.o)
        if tail in seen:
            continue
        seen.add(tail)
        out.append(tail)
        if len(out) >= max_k:
            break
    return out
