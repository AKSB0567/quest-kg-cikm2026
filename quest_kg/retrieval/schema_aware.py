"""Schema-aware Graph-RAG retrieval (Stage 1 of QUEST-KG).

Implements §3.3 of paper/method.md:
  - Anchor linking via cosine similarity + ontology-type filter
  - Bounded multi-hop expansion with top-K per hop
  - Edge scoring: gamma_1 * s_sem + gamma_2 * s_prov + gamma_3 * s_schema

Designed to be encoder-agnostic: pass any callable that maps strings to vectors.
Defaults to sentence-transformers e5-large-v2 in production; tests use a
deterministic dummy encoder.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

import numpy as np

from quest_kg.core.types import Provenance, Subgraph, Triple


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def provenance_score(prov: Provenance, alpha: float = 1.0, beta: float = 0.7,
                     gamma: float = 0.3, tau_decay: float = 365.0) -> float:
    """Provenance reliability score s_prov in [0, 1].

    Sigmoid of weighted sum of extraction confidence, source trust, and
    a recency decay term. See paper/method.md §3.3 step 3.
    """
    decay = 1.0 - math.exp(-prov.edit_age_days / tau_decay)
    logit = alpha * prov.extraction_conf + beta * prov.source_trust - gamma * decay
    return _sigmoid(logit)


class SchemaAwareRetriever:
    """Retrieve a query-specific subgraph Gq from a global KG.

    Args:
        triples:        list of Triple making up the global KG
        encoder:        callable str -> np.ndarray (shape (d,))
        type_compat:    callable (s_type, r, o_type, q_type_set) -> bool
                        Returns True iff (s_type, r, o_type) is admissible under
                        ontology. Per-task implementations live in quest_kg.symbolic.
        k:              hop budget (default 2)
        top_k:          top-K candidates per hop / anchor pool (default 32)
        eps_keep:       minimum EdgeScore to keep an edge (default 0.05)
        gammas:         (gamma_1, gamma_2, gamma_3) weights for (sem, prov, schema)
    """

    def __init__(
        self,
        triples: Iterable[Triple],
        encoder: Callable[[str], np.ndarray],
        type_compat: Optional[Callable] = None,
        k: int = 2,
        top_k: int = 32,
        eps_keep: float = 0.05,
        gammas: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        self.triples = list(triples)
        self.encoder = encoder
        self.type_compat = type_compat or (lambda s_t, r, o_t, q: True)
        self.k = k
        self.top_k = top_k
        self.eps_keep = eps_keep
        assert abs(sum(gammas) - 1.0) < 1e-6, "gammas must sum to 1"
        self.gamma_sem, self.gamma_prov, self.gamma_schema = gammas

        # Build entity -> outgoing-triples index for fast expansion
        self._out: dict[str, list[Triple]] = {}
        self._entities: set[str] = set()
        for t in self.triples:
            self._out.setdefault(t.s, []).append(t)
            self._entities.add(t.s)
            self._entities.add(t.o)

        # Cache entity embeddings (lazily filled)
        self._entity_emb: dict[str, np.ndarray] = {}

    def _embed_entity(self, e: str, e_label: Optional[str] = None) -> np.ndarray:
        if e not in self._entity_emb:
            self._entity_emb[e] = self.encoder(e_label or e)
        return self._entity_emb[e]

    def _edge_text(self, t: Triple) -> str:
        return f"{t.s} {t.r} {t.o}"

    def _edge_score(self, t: Triple, q_emb: np.ndarray, q_types: set[str]) -> float:
        # Schema indicator
        s_schema = 1.0 if self.type_compat(t.s_type, t.r, t.o_type, q_types) else 0.0
        # Semantic similarity
        edge_emb = self.encoder(self._edge_text(t))
        s_sem = max(0.0, _cosine(q_emb, edge_emb))
        # Provenance reliability
        s_prov = provenance_score(t.prov)
        return (
            self.gamma_sem * s_sem
            + self.gamma_prov * s_prov
            + self.gamma_schema * s_schema
        )

    def retrieve(self, query: str, expected_types: Optional[set[str]] = None) -> Subgraph:
        """Run Stage 1 retrieval and return Subgraph Gq.

        Args:
            query:            natural-language query
            expected_types:   set of admissible answer types under the query's
                              intent (used by the type-compat function). If None,
                              pass through (no type filter).
        """
        expected_types = expected_types or set()
        q_emb = self.encoder(query)

        # Anchor linking: cosine similarity to all entities, type-filter, top-K
        anchor_scores: list[tuple[str, float]] = []
        for e in self._entities:
            sim = max(0.0, _cosine(q_emb, self._embed_entity(e)))
            if sim > 0.0:
                anchor_scores.append((e, sim))
        anchor_scores.sort(key=lambda x: -x[1])
        anchors: set[str] = set(e for e, _ in anchor_scores[: self.top_k])

        sg = Subgraph(nodes=set(anchors), anchors=set(anchors))

        # Bounded multi-hop expansion
        frontier: set[str] = set(anchors)
        for _hop in range(self.k):
            next_frontier: set[str] = set()
            candidates: list[tuple[Triple, float]] = []
            for v in frontier:
                for t in self._out.get(v, []):
                    score = self._edge_score(t, q_emb, expected_types)
                    if score < self.eps_keep:
                        continue
                    candidates.append((t, score))
            # top-K across this hop's candidates
            candidates.sort(key=lambda x: -x[1])
            for t, score in candidates[: self.top_k]:
                if t.o not in sg.nodes:
                    next_frontier.add(t.o)
                sg.nodes.add(t.o)
                sg.triples.append(t)
                sg.edge_scores[(t.s, t.r, t.o)] = score
            frontier = next_frontier
            if not frontier:
                break

        return sg
