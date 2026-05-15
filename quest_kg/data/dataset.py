"""Unified Dataset dataclass returned by all loaders.

Every dataset (WebQSP, CWQ, ICEWS18, OrgAccess) ends up in the same shape so
the eval harness and baselines can iterate them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quest_kg.core.types import Triple


@dataclass
class Dataset:
    name: str
    task: str                     # 'qa' | 'temporal_link_pred' | 'access_control'
    triples: list[Triple]         # full KG / fact set
    queries: list[dict]           # each has at least: qid, question, answer
    metadata: dict = field(default_factory=dict)

    @property
    def n_triples(self) -> int:
        return len(self.triples)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "task": self.task,
            "n_triples": self.n_triples,
            "n_queries": self.n_queries,
            "metadata_keys": list(self.metadata.keys()),
        }
