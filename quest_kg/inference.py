"""End-to-end QUEST-KG inference pipeline (Algorithm 1).

Wires together:
  1. SchemaAwareRetriever  -> Subgraph Gq
  2. evidential_mp_numpy   -> per-node (b_i, u_i), attention weights alpha_ij
  3. Per-task SymbolicChecker -> mark paths with delta(pi)
  4. Path scoring          -> S(pi)
  5. should_abstain        -> (p*, H*) abstention rule

Defaults are tuned to be **permissive** at inference (low p*, high H*) so we
get predictions out and let calibration analysis decide where to draw the line.
"""
from __future__ import annotations

import math

import numpy as np

from quest_kg.abstention.posterior import should_abstain
from quest_kg.core.types import Path, PredictResult, Subgraph, Triple
from quest_kg.mp.evidential import evidential_mp_numpy
from quest_kg.retrieval.schema_aware import SchemaAwareRetriever


def _enumerate_paths(sg: Subgraph, max_length: int, cap: int = 256) -> list[Path]:
    """Enumerate paths from any anchor, up to `max_length` hops; cap total at `cap`."""
    paths: list[Path] = []
    out_idx: dict[str, list[Triple]] = {}
    for t in sg.triples:
        out_idx.setdefault(t.s, []).append(t)
    queue: list[list[Triple]] = []
    for t in sg.triples:
        if t.s in sg.anchors:
            queue.append([t])
    while queue:
        cur = queue.pop()
        paths.append(Path(triples=list(cur)))
        if len(paths) >= cap:
            break
        if len(cur) >= max_length:
            continue
        for t in out_idx.get(cur[-1].o, []):
            queue.append(cur + [t])
    return paths


def _score_path(p: Path, sg: Subgraph) -> float:
    """Path score S(pi) = exp(sum log alpha + sum log belief).

    If attention or belief is missing for some node/edge, fall back to 0.5
    (neutral) rather than dropping the term entirely — keeps short paths
    competitive with longer ones.
    """
    log_s = 0.0
    for t in p.triples:
        a = sg.attention.get((t.s, t.o))
        if a is None or a <= 0:
            a = 1.0 / max(len(sg.triples), 1)
        log_s += math.log(max(a, 1e-12))
    for node in p.node_seq():
        b = sg.node_states[node].belief if node in sg.node_states else 0.5
        log_s += math.log(max(b, 1e-12))
    # Normalize by path length so 1-hop and 2-hop paths are comparable
    log_s /= max(p.length, 1)
    return math.exp(log_s)


class QuestKG:
    """End-to-end inference glue (Algorithm 1).

    Permissive defaults so we don't abstain on everything by accident:
      p_star = 0.0       -> never abstain on low retained mass
      h_star = 999.0     -> never abstain on entropy
    Calibration analysis (§4.6) will sweep these for the R-C curve.
    """

    def __init__(
        self,
        retriever: SchemaAwareRetriever,
        symbolic_checker,
        p_star: float = 0.0,
        h_star: float = 999.0,
        mp_gammas: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
        kappa: float = 4.0,
        max_path_length: int | None = None,
        path_cap: int = 256,
    ):
        self.retriever = retriever
        self.symbolic_checker = symbolic_checker
        self.p_star = p_star
        self.h_star = h_star
        self.mp_gammas = mp_gammas
        self.kappa = kappa
        self.max_path_length = max_path_length or retriever.k
        self.path_cap = path_cap

    def predict(
        self,
        query: str,
        expected_types: set[str] | None = None,
    ) -> PredictResult:
        # ---- Stage 1: retrieval -------------------------------------------------
        sg = self.retriever.retrieve(query, expected_types=expected_types)

        # Anchor similarities for MP init (use cached entity embeddings)
        anchor_sims: dict[str, float] = {}
        if sg.anchors:
            try:
                q_emb = np.asarray(self.retriever.encoder(query), dtype=np.float32)
                q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
                for a in sg.anchors:
                    if (self.retriever._entity_emb is not None
                            and a in self.retriever._entity_idx):
                        e = self.retriever._entity_emb[self.retriever._entity_idx[a]]
                        e_norm = e / (np.linalg.norm(e) + 1e-12)
                        anchor_sims[a] = float(np.dot(q_norm, e_norm))
                    else:
                        anchor_sims[a] = 1.0
            except Exception:
                anchor_sims = {a: 1.0 for a in sg.anchors}

        # ---- Stage 2: evidential MP --------------------------------------------
        if sg.triples:
            evidential_mp_numpy(
                sg,
                anchor_sims=anchor_sims,
                gammas=self.mp_gammas,
                kappa=self.kappa,
                n_iters=self.retriever.k,
            )
        else:
            # No edges retrieved -> assign neutral belief to anchors
            from quest_kg.core.types import NodeState
            for n in sg.nodes:
                sg.node_states[n] = NodeState(alpha_pos=1.0, alpha_neg=1.0)

        # ---- Stage 3: paths + symbolic + abstention ----------------------------
        candidate_paths = _enumerate_paths(sg, max_length=self.max_path_length, cap=self.path_cap)
        for p in candidate_paths:
            try:
                p.violates_symbolic = bool(self.symbolic_checker.violates(p))
            except Exception:
                p.violates_symbolic = False
            p.score = _score_path(p, sg)

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
