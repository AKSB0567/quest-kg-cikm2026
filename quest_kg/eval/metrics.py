"""Calibration + reasoning metrics. All numpy/torch, GPU-friendly where it matters."""
from __future__ import annotations

import numpy as np


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Args:
        confidences: predicted probability of the chosen class, shape (N,).
        correct:     1 if prediction was correct else 0, shape (N,).
        n_bins:      number of equal-width bins.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece_val += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece_val)


def brier(probs: np.ndarray, one_hot_targets: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between probabilities and one-hot truth."""
    probs = np.asarray(probs, dtype=np.float64)
    one_hot_targets = np.asarray(one_hot_targets, dtype=np.float64)
    return float(np.mean(np.sum((probs - one_hot_targets) ** 2, axis=-1)))


def nll(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-12) -> float:
    """Negative log-likelihood. `targets` are class indices."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    p_true = np.clip(probs[np.arange(len(targets)), targets], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def aurc(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Area Under Risk-Coverage curve. Lower is better.

    Sort by confidence descending; cumulatively compute risk vs coverage.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-confidences)
    risks = (1 - correct[order]).cumsum() / (np.arange(len(order)) + 1)
    coverages = (np.arange(len(order)) + 1) / len(order)
    return float(np.trapz(risks, coverages))


def hits_at_k(scores: np.ndarray, gold_idx: np.ndarray, k: int) -> float:
    """Hits@k for entity alignment / link prediction.

    Args:
        scores:    shape (N_queries, N_candidates) — higher = more likely match.
        gold_idx:  shape (N_queries,) — index of correct candidate.
    """
    scores = np.asarray(scores)
    gold_idx = np.asarray(gold_idx)
    top_k = np.argsort(-scores, axis=1)[:, :k]
    hits = (top_k == gold_idx[:, None]).any(axis=1).mean()
    return float(hits)


def mrr(scores: np.ndarray, gold_idx: np.ndarray) -> float:
    """Mean Reciprocal Rank."""
    scores = np.asarray(scores)
    gold_idx = np.asarray(gold_idx)
    ranks = (-scores).argsort(axis=1).argsort(axis=1)
    gold_ranks = ranks[np.arange(len(gold_idx)), gold_idx] + 1
    return float(np.mean(1.0 / gold_ranks))
