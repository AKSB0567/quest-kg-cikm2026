"""Posterior-mass + entropy abstention head (Stage 3 of QUEST-KG).

Implements §3.5 of paper/method.md:
  Abstain iff M_q < p* OR H_q > H*
where
  M_q = sum of valid path scores / sum of all path scores
  H_q = entropy over candidate-answer distribution P(a | Gq).
"""
from __future__ import annotations

import math
from collections.abc import Iterable

from quest_kg.core.types import Path


def _safe_log(p: float, eps: float = 1e-12) -> float:
    return math.log(max(p, eps))


def aggregate_answer_probs(valid_paths: Iterable[Path]) -> dict[str, float]:
    """Build P(answer | Gq) by summing path scores per terminal entity, normalized."""
    sums: dict[str, float] = {}
    z = 0.0
    for p in valid_paths:
        sums[p.tail] = sums.get(p.tail, 0.0) + p.score
        z += p.score
    if z <= 0.0:
        return {}
    return {a: s / z for a, s in sums.items()}


def retained_posterior_mass(all_paths: Iterable[Path]) -> float:
    """M_q = (sum of valid path scores) / (sum of all path scores)."""
    all_paths = list(all_paths)
    total = sum(p.score for p in all_paths)
    if total <= 0.0:
        return 0.0
    kept = sum(p.score for p in all_paths if not p.violates_symbolic)
    return kept / total


def answer_entropy(probs: dict[str, float]) -> float:
    """Shannon entropy over the answer distribution, in bits."""
    h = 0.0
    for p in probs.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def should_abstain(
    all_paths: Iterable[Path],
    p_star: float = 0.05,
    h_star: float = 0.6,
) -> tuple[bool, float, float, dict[str, float]]:
    """Decide whether to abstain on this query.

    Returns:
        abstained:     True if should abstain
        M_q:           retained posterior mass
        H_q:           entropy in bits
        answer_probs:  P(answer | Gq) over candidate tails
    """
    paths = list(all_paths)
    M_q = retained_posterior_mass(paths)
    valid = [p for p in paths if not p.violates_symbolic]
    probs = aggregate_answer_probs(valid)
    H_q = answer_entropy(probs)
    abstained = (M_q < p_star) or (H_q > h_star)
    return abstained, M_q, H_q, probs
