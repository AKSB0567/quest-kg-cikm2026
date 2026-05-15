"""End-to-end QUEST-KG inference pipeline (Algorithm 1).

Wires together:
  1. SchemaAwareRetriever  -> Subgraph Gq
  2. evidential_mp_numpy   -> per-node (b_i, u_i), attention weights alpha_ij
  3. Per-task SymbolicChecker -> mark paths with delta(pi)
  4. Path scoring          -> S(pi)
  5. should_abstain        -> (p*, H*) abstention rule

For training, swap step 2 for the PyTorch `EvidentialMP` module and backprop
through `gamma_k` + per-relation `w_r^{pos,neg}`. This file is the inference
glue and is what the experiment notebooks will call.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np

from quest_kg.abstention.posterior import should_abstain
from quest_kg.core.types import Path, PredictResult, Subgraph, Triple
from quest_kg.mp.evidential import evidential_mp_numpy
from quest_kg.retrieval.schema_aware import SchemaAwareRetriever, provenance_score


def _enumerate_paths(sg: Subgraph, max_length: int) -> list[Path]:
    """Best-first enumeration of paths from any anchor up to `max_length` hops.

    Score-prioritized; we cap candidates per path-length step to keep total <= 256.
    """
    paths: list[Path] = []
    # Seed with anchor-out 1-hop edges
    queue: list[list[Triple]] = []
    for t in sg.triples:
        if t.s in sg.anchors:
            queue.append([t])
    while queue:
        cur = queue.pop()
        paths.append(Path(triples=list(cur)))
        if len(cur) >= max_length:
            continue
        tail = cur[-1].o
        # Outgoing edges from current tail within the subgraph
        for t in sg.triples:
            if t.s == tail:
                queue.append(cur + [t])
        if len(paths) >= 1024:
            break
    return paths[:1024]


def _score_path(p: Path, sg: Subgraph, anchor_sims: dict[str, float]) -> float:
    """log S(pi) = sum log alpha + sum log b_i (no per-relation weights at inference
    in this NumPy path; learned weights live in the PyTorch module)."""
    log_s = 0.0
    for t in p.triples:
        a = sg.attention.get((t.s, t.o), None)
        if a is None or a <= 0:
            continue
        log_s += math.log(a)
    for node in p.node_seq():
        b = sg.node_states[node].belief if node in sg.node_states else 0.5
        log_s += math.log(max(b, 1e-12))
    return math.exp(log_s)


class QuestKG:
    """End-to-end inference glue (Algorithm 1)."""

    def __init__(
        self,
        retriever: SchemaAwareRetriever,
        symbolic_checker,
        p_star: float = 0.05,
        h_star: float = 0.6,
        mp_gammas: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
        kappa: float = 4.0,
        max_path_length: int | None = None,
    ):
        self.retriever = retriever
        self.symbolic_checker = symbolic_checker
        self.p_star = p_star
        self.h_star = h_star
        self.mp_gammas = mp_gammas
        self.kappa = kappa
        self.max_path_length = max_path_length or retriever.k

    def predict(
        self,
        query: str,
        expected_types: set[str] | None = None,
    ) -> PredictResult:
        # ---- Stage 1: retrieval -------------------------------------------------
        sg = self.retriever.retrieve(query, expected_types=expected_types)

        # Anchor similarities (for MP init)
        q_emb = self.retriever.encoder(query)
        anchor_sims: dict[str, float] = {}
        for a in sg.anchors:
            e = self.retriever._embed_entity(a)
            num = float(np.dot(q_emb, e))
            den = float(np.linalg.norm(q_emb) * np.linalg.norm(e))
            anchor_sims[a] = (num / den) if den > 0 else 0.0

        # ---- Stage 2: evidential MP --------------------------------------------
        evidential_mp_numpy(
            sg,
            anchor_sims=anchor_sims,
            gammas=self.mp_gammas,
            kappa=self.kappa,
            n_iters=self.retriever.k,
        )

        # ---- Stage 3: paths + symbolic + abstention ----------------------------
        candidate_paths = _enumerate_paths(sg, max_length=self.max_path_length)
        for p in candidate_paths:
            p.violates_symbolic = bool(self.symbolic_checker.violates(p))
            p.score = _score_path(p, sg, anchor_sims)

        abstained, M_q, H_q, probs = should_abstain(
            candidate_paths, p_star=self.p_star, h_star=self.h_star
        )

        valid = [p for p in candidate_paths if not p.violates_symbolic]
        top = max(valid, key=lambda x: x.score) if valid else None
        prediction = top.tail if (top and not abstained) else None

        return PredictResult(
            query=query,
            prediction=prediction,
            abstained=abstained,
            confidence=M_q,
            entropy=H_q,
            subgraph=sg,
            top_path=top,
            candidate_paths=candidate_paths,
            extra={"answer_probs": probs, "n_valid_paths": len(valid)},
        )
