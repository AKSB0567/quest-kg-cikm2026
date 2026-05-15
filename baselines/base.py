"""Common interface for baselines.

Every baseline takes a query, optionally a query-specific KG slice / retrieval
budget, and returns a BaselinePrediction including a confidence (used for
matched-condition calibration comparison).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

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
