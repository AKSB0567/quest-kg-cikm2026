"""Schema-aware Graph-RAG retrieval (Stage 1 of QUEST-KG).

Implements §3.3 of paper/method.md:
  - Anchor linking via cosine similarity + ontology-type filter
  - Bounded multi-hop expansion with top-K per hop
  - Edge scoring: gamma_1 * s_sem + gamma_2 * s_prov + gamma_3 * s_schema

Performance: all entity surface forms and all edge texts are batch-embedded
**once** at construction time, then per-query retrieval is pure NumPy matmul.
Per-query latency on a KG of 100K triples is ~5-50ms on CPU, ~1-5ms on GPU encoder.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from typing import Optional

import numpy as np

from quest_kg.core.types import Provenance, Subgraph, Triple


# Stop words for relation-aware retrieval token overlap. Question words and
# common syntax should not match relation names; content tokens should.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into",
    "and", "or", "but", "not", "no", "if", "then", "than", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose", "where",
    "when", "why", "how", "can", "could", "should", "would", "will",
    "i", "you", "he", "she", "it", "we", "they", "them", "his", "her",
    "us", "our", "your", "their", "my", "me",
})


_TOK_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _content_tokens(s: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, drop stopwords + 1-char tokens."""
    if not s:
        return set()
    raw = _TOK_SPLIT_RE.split(s.lower())
    return {t for t in raw if t and len(t) > 1 and t not in _STOPWORDS}


def _relation_tokens(r: str) -> set[str]:
    """Split a relation surface form like 'location.country.languages_spoken'
    or 'place_of_birth' into content tokens."""
    if not r:
        return set()
    # Normalise: dots and underscores are token separators in Freebase + ICEWS schemas.
    r = r.replace(".", " ").replace("_", " ").replace("~", " ")
    return _content_tokens(r)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def provenance_score(prov: Provenance, alpha: float = 1.0, beta: float = 0.7,
                     gamma: float = 0.3, tau_decay: float = 365.0) -> float:
    """Provenance reliability score s_prov in [0, 1]. See paper/method.md §3.3 step 3."""
    decay = 1.0 - math.exp(-prov.edit_age_days / tau_decay)
    logit = alpha * prov.extraction_conf + beta * prov.source_trust - gamma * decay
    return _sigmoid(logit)


class SchemaAwareRetriever:
    """Retrieve a query-specific subgraph Gq from a global KG.

    All entity and edge embeddings are precomputed at construction time, so
    `retrieve(query)` is O(|V| d) for anchor scoring + O(|expanded edges| d) for
    edge scoring, dominated by matmul ops -- not by encoder forward passes.

    Args:
        triples:        list of Triple making up the global KG
        encoder:        callable str -> np.ndarray (shape (d,) or batch (B,d))
        type_compat:    callable (s_type, r, o_type, q_type_set) -> bool
        k:              hop budget (default 2)
        top_k:          top-K candidates per hop / anchor pool (default 32)
        eps_keep:       minimum EdgeScore to keep an edge (default 0.05)
        gammas:         (gamma_1, gamma_2, gamma_3) weights for (sem, prov, schema)
        precompute:     if True (default), embed all entities + edges once.
                        If False, lazy-embed (legacy slow path).
    """

    def __init__(
        self,
        triples: Iterable[Triple],
        encoder: Callable,
        type_compat: Optional[Callable] = None,
        k: int = 2,
        top_k: int = 32,
        eps_keep: float = 0.05,
        gammas: tuple[float, float, float] = (0.4, 0.3, 0.3),
        precompute: bool = True,
        relation_bias: float = 0.4,
        bidirectional: bool = True,
    ):
        self.triples = list(triples)
        self.encoder = encoder
        self.type_compat = type_compat or (lambda s_t, r, o_t, q: True)
        self.k = k
        self.top_k = top_k
        self.eps_keep = eps_keep
        if abs(sum(gammas) - 1.0) > 1e-6:
            # Renormalize defensively
            tot = sum(gammas) or 1.0
            gammas = tuple(g / tot for g in gammas)
        self.gamma_sem, self.gamma_prov, self.gamma_schema = gammas
        # Additive boost for each edge whose relation tokens overlap with
        # query content tokens. Default 0.4 -> a full-overlap edge gets
        # +0.4 added to its (sem+prov+schema) score before top-k pruning.
        self.relation_bias = float(relation_bias)
        # Bidirectional retrieval: True for QA (Freebase has reverse-direction
        # answers like `(Lang | main_country | Jamaica)` where the anchor is
        # the object). False for clean directional graphs like ICEWS18 where
        # adding incoming edges only adds noise.
        self.bidirectional = bool(bidirectional)
        # Precompute relation token sets per triple (only depends on KG)
        self._rel_tokens: list[set[str]] = [_relation_tokens(t.r) for t in self.triples]

        # Build entity -> outgoing AND incoming edge-index indices.
        # Incoming-edge expansion lets us reach answers where the anchor is
        # the object of a triple (e.g. Freebase `(Lang | main_country | Jamaica)`
        # when the query anchors on Jamaica). Without `_in_idx`, the relevant
        # triple is never scored and the answer is unreachable regardless of
        # downstream path enumeration.
        self._out_idx: dict[str, list[int]] = {}
        self._in_idx: dict[str, list[int]] = {}
        self._entities: list[str] = []
        seen: set[str] = set()
        for i, t in enumerate(self.triples):
            self._out_idx.setdefault(t.s, []).append(i)
            self._in_idx.setdefault(t.o, []).append(i)
            for e in (t.s, t.o):
                if e not in seen:
                    seen.add(e)
                    self._entities.append(e)
        self._entity_idx = {e: i for i, e in enumerate(self._entities)}

        # Provenance scores per triple (constant — depends only on Provenance, not query)
        self._prov_scores = np.array(
            [provenance_score(t.prov) for t in self.triples], dtype=np.float32
        )

        # Embeddings
        self._entity_emb: Optional[np.ndarray] = None      # (|V|, d)
        self._edge_emb: Optional[np.ndarray] = None        # (|E|, d)

        if precompute:
            self._precompute_embeddings()

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def _precompute_embeddings(self) -> None:
        """Batch-embed all entities and edge texts once."""
        if self._entities:
            ent_embs = self.encoder(self._entities)
            self._entity_emb = self._as_2d(ent_embs)
        if self.triples:
            edge_texts = [f"{t.s} {t.r} {t.o}" for t in self.triples]
            ed_embs = self.encoder(edge_texts)
            self._edge_emb = self._as_2d(ed_embs)

    @staticmethod
    def _as_2d(x) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        return a

    @staticmethod
    def _l2norm(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            n = float(np.linalg.norm(a))
            return a / n if n > 0 else a
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return a / norms

    # -----------------------------------------------------------------------
    # Lazy fallback for non-precomputed mode
    # -----------------------------------------------------------------------

    def _embed_entity(self, e: str) -> np.ndarray:
        if self._entity_emb is not None and e in self._entity_idx:
            return self._entity_emb[self._entity_idx[e]]
        return np.asarray(self.encoder(e), dtype=np.float32)

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        expected_types: Optional[set[str]] = None,
        explicit_anchors: Optional[Iterable[str]] = None,
    ) -> Subgraph:
        """Run Stage 1 retrieval and return Subgraph Gq.

        Args:
            query:            natural-language query string (used for semantic edge scoring)
            expected_types:   set of admissible answer types (passed to type_compat)
            explicit_anchors: if provided, use these as the anchor set directly,
                              skipping semantic-similarity anchor linking. Use this
                              when the query metadata names specific KG entities
                              (e.g. OrgAccess: user + resource; WebQSP: q_entity).
                              Unknown anchor names are silently dropped.
        """
        expected_types = expected_types or set()
        q_emb = self._l2norm(np.asarray(self.encoder(query), dtype=np.float32))
        q_tokens = _content_tokens(query)

        # ---- Anchor linking ------------------------------------------------
        if explicit_anchors is not None:
            anchors = {a for a in explicit_anchors if a in self._entity_idx}
            # If no explicit anchors resolved, fall back to top-K semantic
            if not anchors and self._entity_emb is not None:
                ent_norm = self._l2norm(self._entity_emb)
                sims = ent_norm @ q_emb
                top_idx = np.argpartition(-sims, min(self.top_k, len(sims) - 1))[: self.top_k]
                top_idx = top_idx[np.argsort(-sims[top_idx])]
                anchors = {self._entities[i] for i in top_idx if sims[i] > 0}
        elif self._entity_emb is None:
            # Fallback path (slow; tests only)
            anchor_scores = []
            for e in self._entities:
                sim = float(np.dot(q_emb, self._l2norm(self._embed_entity(e))))
                if sim > 0:
                    anchor_scores.append((e, sim))
            anchor_scores.sort(key=lambda x: -x[1])
            anchors = set(e for e, _ in anchor_scores[: self.top_k])
        else:
            ent_norm = self._l2norm(self._entity_emb)
            sims = ent_norm @ q_emb  # (|V|,)
            top_idx = np.argpartition(-sims, min(self.top_k, len(sims) - 1))[: self.top_k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            anchors = {self._entities[i] for i in top_idx if sims[i] > 0}

        sg = Subgraph(nodes=set(anchors), anchors=set(anchors))

        # ---- Bounded multi-hop expansion -----------------------------------
        frontier: set[str] = set(anchors)
        for _hop in range(self.k):
            # Gather candidate-edge indices touching the current frontier in
            # EITHER direction. De-dup with a set so an edge whose subject and
            # object are both frontier nodes isn't counted twice.
            cand_set: set[int] = set()
            for v in frontier:
                cand_set.update(self._out_idx.get(v, []))
                if self.bidirectional:
                    cand_set.update(self._in_idx.get(v, []))
            if not cand_set:
                break
            # Sort by triple index for deterministic candidate order. Set
            # iteration order is hash-based so without this two identical
            # inputs can produce different rankings when path scores tie.
            cand_edge_idx = np.array(sorted(cand_set), dtype=np.int64)

            # Vectorized scoring
            if self._edge_emb is not None:
                edge_norm = self._l2norm(self._edge_emb[cand_edge_idx])
                s_sem = np.maximum(edge_norm @ q_emb, 0.0)  # (E,)
            else:
                s_sem = np.array(
                    [max(0.0, float(np.dot(q_emb, self._l2norm(np.asarray(self.encoder(f"{self.triples[i].s} {self.triples[i].r} {self.triples[i].o}"), dtype=np.float32)))))
                     for i in cand_edge_idx],
                    dtype=np.float32,
                )
            s_prov = self._prov_scores[cand_edge_idx]

            # Schema indicator (per-edge, by triple type) — vectorize via list comp
            s_schema = np.array(
                [1.0 if self.type_compat(self.triples[i].s_type, self.triples[i].r,
                                         self.triples[i].o_type, expected_types) else 0.0
                 for i in cand_edge_idx],
                dtype=np.float32,
            )

            # Relation-aware boost: fraction of query content tokens that
            # appear in the edge's relation surface form. Edges with high
            # overlap (e.g. "where ... from" matching "place_of_birth") get
            # an additive `relation_bias` lift, putting them above noisy
            # but semantically-similar distractors.
            if q_tokens and self.relation_bias > 0:
                s_rel = np.array(
                    [
                        (len(q_tokens & self._rel_tokens[int(i)]) / len(q_tokens))
                        if self._rel_tokens[int(i)] else 0.0
                        for i in cand_edge_idx
                    ],
                    dtype=np.float32,
                )
            else:
                s_rel = np.zeros(len(cand_edge_idx), dtype=np.float32)

            scores = (
                self.gamma_sem * s_sem
                + self.gamma_prov * s_prov
                + self.gamma_schema * s_schema
                + self.relation_bias * s_rel
            )

            # Keep top-K with score >= eps_keep
            keep_mask = scores >= self.eps_keep
            kept_idx_local = np.where(keep_mask)[0]
            if len(kept_idx_local) == 0:
                break
            kept_scores = scores[kept_idx_local]
            order = np.argsort(-kept_scores)
            top_local = kept_idx_local[order[: self.top_k]]
            top_global = cand_edge_idx[top_local]
            top_scores = scores[top_local]

            next_frontier: set[str] = set()
            for ei, sc in zip(top_global, top_scores):
                t = self.triples[int(ei)]
                # Frontier grows on the *other* endpoint relative to anchors.
                # Add both endpoints to nodes; whichever wasn't already in the
                # subgraph extends the frontier for the next hop.
                for endpoint in (t.s, t.o):
                    if endpoint not in sg.nodes:
                        next_frontier.add(endpoint)
                    sg.nodes.add(endpoint)
                sg.triples.append(t)
                sg.edge_scores[(t.s, t.r, t.o)] = float(sc)
            frontier = next_frontier
            if not frontier:
                break

        return sg
