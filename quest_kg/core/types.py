"""Shared dataclasses used across QUEST-KG modules.

These types are deliberately lightweight (frozen dataclasses + plain dicts/sets)
so they pickle cleanly for checkpointing and serialize well to JSON for logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Provenance:
    """Provenance record attached to each triple. See paper/method.md §3.2."""
    source_id: str = "unknown"
    extraction_conf: float = 1.0
    edit_age_days: float = 0.0
    source_trust: float = 1.0


@dataclass(frozen=True)
class Triple:
    """A KG triple (subject, relation, object) with provenance + types."""
    s: str
    r: str
    o: str
    s_type: str = "Entity"
    o_type: str = "Entity"
    timestamp: int | None = None
    prov: Provenance = field(default_factory=Provenance)


@dataclass
class NodeState:
    """Per-node state during evidential message passing.

    Dirichlet pseudo-counts (alpha_pos, alpha_neg) give belief b and uncertainty u
    via b = a_pos / (a_pos + a_neg + 2), u = 2 / (a_pos + a_neg + 2).
    """
    alpha_pos: float = 1.0
    alpha_neg: float = 1.0

    @property
    def belief(self) -> float:
        return self.alpha_pos / (self.alpha_pos + self.alpha_neg + 2.0)

    @property
    def uncertainty(self) -> float:
        return 2.0 / (self.alpha_pos + self.alpha_neg + 2.0)


@dataclass
class Subgraph:
    """Query-specific retrieved subgraph Gq.

    Attributes:
        nodes:        set of entity ids in Vq
        triples:      list of Triple in Eq
        anchors:      ids of entities matched directly to the query
        edge_scores:  {(s, r, o): EdgeScore} per the retrieval-stage scoring
        node_states:  {entity_id: NodeState} after MP (filled by Stage 2)
        attention:    {(j, i): alpha_ji} attention weights from MP (filled by Stage 2)
    """
    nodes: set = field(default_factory=set)
    triples: list = field(default_factory=list)
    anchors: set = field(default_factory=set)
    edge_scores: dict = field(default_factory=dict)
    node_states: dict = field(default_factory=dict)
    attention: dict = field(default_factory=dict)

    def neighbors_of(self, node: str) -> list:
        """Outgoing neighbors (objects) of a node in this subgraph."""
        return [t.o for t in self.triples if t.s == node]

    def incoming_of(self, node: str) -> list:
        """Incoming subjects of a node in this subgraph."""
        return [(t.s, t.r) for t in self.triples if t.o == node]


@dataclass
class Path:
    """A reasoning path through Gq. Sequence of triples; nodes derived."""
    triples: list  # list[Triple]
    score: float = 0.0
    violates_symbolic: bool = False

    @property
    def head(self) -> str:
        return self.triples[0].s if self.triples else ""

    @property
    def tail(self) -> str:
        return self.triples[-1].o if self.triples else ""

    @property
    def length(self) -> int:
        return len(self.triples)

    def node_seq(self) -> list:
        if not self.triples:
            return []
        return [self.triples[0].s] + [t.o for t in self.triples]


@dataclass
class PredictResult:
    """Output of QUEST-KG.predict()."""
    query: str
    prediction: str | None  # None if abstained
    abstained: bool
    confidence: float  # retained posterior mass M_q
    entropy: float     # H_q (bits)
    subgraph: Subgraph
    top_path: Path | None
    candidate_paths: list = field(default_factory=list)  # all valid paths with scores
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "prediction": self.prediction,
            "abstained": self.abstained,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "n_nodes": len(self.subgraph.nodes),
            "n_triples": len(self.subgraph.triples),
            "n_anchors": len(self.subgraph.anchors),
            "top_path": [(t.s, t.r, t.o) for t in self.top_path.triples] if self.top_path else None,
            "n_candidate_paths": len(self.candidate_paths),
            "extra": self.extra,
        }
