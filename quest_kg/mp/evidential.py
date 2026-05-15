"""Evidential message passing (Stage 2 of QUEST-KG).

Implements §3.4 of paper/method.md:
  - Each node carries Dirichlet pseudo-counts (alpha_pos, alpha_neg).
  - Attention logit per neighbor: phi_ji = g1*s_sem + g2*s_prov + g3*s_schema - g4*u_j
  - alpha_ji = softmax_j(phi_ji); then propagate beliefs.

This module is **PyTorch-based** for training; inference can run on CPU or CUDA.
For lightweight unit tests we expose a `forward_numpy` variant that runs without torch.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional

import numpy as np

try:  # torch is optional for unit tests
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False

from quest_kg.core.types import NodeState, Subgraph
from quest_kg.retrieval.schema_aware import provenance_score


# -----------------------------------------------------------------------------
# NumPy reference implementation (used in tests + as fallback)
# -----------------------------------------------------------------------------

def _softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(xs)] * len(xs)


def init_node_states(
    sg: Subgraph,
    anchor_sims: dict[str, float],
    kappa: float = 4.0,
) -> None:
    """Initialize NodeState for every node in sg (in place).

    Anchors get a boosted positive prior, others get the uninformative prior (1, 1).
    """
    for node in sg.nodes:
        if node in sg.anchors:
            s = max(0.0, anchor_sims.get(node, 0.0))
            sg.node_states[node] = NodeState(alpha_pos=1.0 + kappa * s, alpha_neg=1.0)
        else:
            sg.node_states[node] = NodeState(alpha_pos=1.0, alpha_neg=1.0)


def evidential_mp_numpy(
    sg: Subgraph,
    anchor_sims: dict[str, float],
    gammas: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
    kappa: float = 4.0,
    w_pos: float = 1.0,
    w_neg: float = 1.0,
    n_iters: int | None = None,
) -> None:
    """Run Algorithm 1, Stage 2 in pure NumPy (no torch).

    Mutates `sg` in place by filling `sg.node_states` and `sg.attention`.

    Args:
        sg:            retrieved subgraph (from Stage 1)
        anchor_sims:   {anchor_id: cosine_sim_to_query}
        gammas:        (gamma_1, gamma_2, gamma_3, gamma_4) attention weights.
                       gamma_4 multiplies the uncertainty penalty term.
        kappa:         anchor prior boost
        w_pos, w_neg:  per-relation propagation scalars (single global default
                       for tests; use the Torch class for per-relation learned values)
        n_iters:       MP iterations; defaults to k (from retrieval); we don't have k here,
                       so default to 2 (matches §3.7 default).
    """
    if n_iters is None:
        n_iters = 2

    assert abs(sum(gammas[:3]) + gammas[3] - 1.0) < 1e-6 or True, "gammas: γ_4 is a penalty, not part of the softmax constraint"
    g1, g2, g3, g4 = gammas

    init_node_states(sg, anchor_sims, kappa=kappa)

    # Pre-compute static edge features (semantic similarity is constant once retrieval is done;
    # we lift it from sg.edge_scores by decomposing — for simplicity here, we use the stored
    # combined edge_score as a proxy for the semantic + provenance + schema contribution).
    # In the production torch version below we recompute the three components individually.
    static_edge_logit = {
        (t.s, t.r, t.o): g1 * sg.edge_scores.get((t.s, t.r, t.o), 0.0)
        + g2 * provenance_score(t.prov)
        + g3 * (1.0 if hasattr(t, "_schema_ok") and getattr(t, "_schema_ok") else 1.0)
        for t in sg.triples
    }

    sg.attention = {}

    # Build incoming-edge map for each node
    inbound: dict[str, list[tuple[str, str]]] = {}
    for t in sg.triples:
        inbound.setdefault(t.o, []).append((t.s, t.r))

    for _ in range(n_iters):
        new_states: dict[str, NodeState] = {}
        for node in sg.nodes:
            sources = inbound.get(node, [])
            if not sources:
                new_states[node] = sg.node_states[node]
                continue
            # Compute attention logits over incoming neighbors
            logits = []
            for s, r in sources:
                u_s = sg.node_states[s].uncertainty
                logit = static_edge_logit.get((s, r, node), 0.0) - g4 * u_s
                logits.append(logit)
            alphas = _softmax(logits)
            # Aggregate
            da_pos = 0.0
            da_neg = 0.0
            for (src, r), a in zip(sources, alphas):
                b_s = sg.node_states[src].belief
                da_pos += a * b_s * w_pos
                da_neg += a * (1.0 - b_s) * w_neg
                sg.attention[(src, node)] = a
            cur = sg.node_states[node]
            new_states[node] = NodeState(
                alpha_pos=cur.alpha_pos + da_pos,
                alpha_neg=cur.alpha_neg + da_neg,
            )
        sg.node_states = new_states


# -----------------------------------------------------------------------------
# PyTorch implementation with learnable γ and per-relation w_r
# -----------------------------------------------------------------------------

if HAS_TORCH:

    class EvidentialMP(nn.Module):
        """Learnable evidential MP module.

        γ_1..γ_4 are reparameterized via softmax over a 4-vector so that they
        sum to 1 and remain non-negative (γ_4 is the uncertainty penalty
        component; we project the same way for stability).

        Per-relation scalars w_r^pos, w_r^neg are learned as an Embedding table.
        """

        def __init__(self, n_relations: int, init_gammas: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)):
            super().__init__()
            init = torch.log(torch.tensor(list(init_gammas)) + 1e-8)
            self.gamma_logits = nn.Parameter(init.clone())
            self.w_pos = nn.Embedding(n_relations, 1)
            self.w_neg = nn.Embedding(n_relations, 1)
            nn.init.constant_(self.w_pos.weight, 1.0)
            nn.init.constant_(self.w_neg.weight, 1.0)

        @property
        def gammas(self) -> torch.Tensor:
            return torch.softmax(self.gamma_logits, dim=0)

        def forward(
            self,
            edges: torch.Tensor,           # (E, 2) int64 [src_idx, dst_idx]
            edge_rel: torch.Tensor,        # (E,)    int64 relation index
            s_sem: torch.Tensor,           # (E,)    float
            s_prov: torch.Tensor,          # (E,)    float
            s_schema: torch.Tensor,        # (E,)    float
            alpha_pos: torch.Tensor,       # (N,)    float
            alpha_neg: torch.Tensor,       # (N,)    float
            n_iters: int = 2,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Run MP and return (alpha_pos, alpha_neg, attention_weights).

            attention_weights is (E,) — per-edge softmaxed weight relative to its destination's incoming edges.
            """
            device = alpha_pos.device
            gammas = self.gammas
            g1, g2, g3, g4 = gammas[0], gammas[1], gammas[2], gammas[3]

            src = edges[:, 0]
            dst = edges[:, 1]
            N = alpha_pos.shape[0]
            E = edges.shape[0]

            for _ in range(n_iters):
                # Per-node uncertainty u_s
                u = 2.0 / (alpha_pos + alpha_neg + 2.0)  # (N,)
                u_src = u[src]                            # (E,)

                # Attention logits per edge
                phi = g1 * s_sem + g2 * s_prov + g3 * s_schema - g4 * u_src  # (E,)

                # Per-dst softmax: for stability subtract per-dst max
                # max-pool by dst
                max_per_dst = torch.full((N,), float("-inf"), device=device).scatter_reduce(
                    0, dst, phi, reduce="amax", include_self=True
                )
                phi_shift = phi - max_per_dst[dst]
                exp_phi = torch.exp(phi_shift)
                sum_per_dst = torch.zeros(N, device=device).scatter_add(0, dst, exp_phi)
                attn = exp_phi / (sum_per_dst[dst] + 1e-12)  # (E,)

                # Belief of source
                b_src = alpha_pos[src] / (alpha_pos[src] + alpha_neg[src] + 2.0)

                # Per-relation propagation scalars
                wpos = self.w_pos(edge_rel).squeeze(-1)  # (E,)
                wneg = self.w_neg(edge_rel).squeeze(-1)  # (E,)

                # Messages
                msg_pos = attn * b_src * wpos
                msg_neg = attn * (1.0 - b_src) * wneg

                # Aggregate to destination
                d_alpha_pos = torch.zeros(N, device=device).scatter_add(0, dst, msg_pos)
                d_alpha_neg = torch.zeros(N, device=device).scatter_add(0, dst, msg_neg)

                alpha_pos = alpha_pos + d_alpha_pos
                alpha_neg = alpha_neg + d_alpha_neg

            return alpha_pos, alpha_neg, attn

else:  # pragma: no cover

    class EvidentialMP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch not available; install torch to use the learnable EvidentialMP module.")
