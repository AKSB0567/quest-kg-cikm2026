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
    # np.trapezoid in NumPy >= 2.0, np.trapz in older versions
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(trapezoid(risks, coverages))


def balanced_accuracy(preds: list[str], golds: list[str], positive: str = "1") -> float:
    """Balanced accuracy = (TPR + TNR) / 2 for binary tasks.

    Robust to class imbalance. preds/golds already normalized to "1"/"0" (or
    equivalent) by harness._normalize. Returns 0.5 if either class is missing.
    """
    p = np.asarray([1 if str(x) == positive else 0 for x in preds], dtype=np.int64)
    g = np.asarray([1 if str(x) == positive else 0 for x in golds], dtype=np.int64)
    tp = int(((p == 1) & (g == 1)).sum())
    tn = int(((p == 0) & (g == 0)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    if (tp + fn) == 0 or (tn + fp) == 0:
        return 0.5
    return float((tpr + tnr) / 2.0)


def positive_class_prf(preds: list[str], golds: list[str], positive: str = "1") -> dict:
    """Precision/Recall/F1 for the *positive* class on a binary task.

    On OrgAccess this is the metric that actually matters — the negative class
    is ~91% prevalence, so a "guess no" baseline scores EM ≈ 0.91 but pos-F1 = 0.
    """
    p = np.asarray([1 if str(x) == positive else 0 for x in preds], dtype=np.int64)
    g = np.asarray([1 if str(x) == positive else 0 for x in golds], dtype=np.int64)
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn, "support_pos": int(g.sum()),
            "support_neg": int((1 - g).sum())}


def macro_f1_binary(preds: list[str], golds: list[str], positive: str = "1") -> float:
    """Macro-F1 across both classes (avg of pos-F1 and neg-F1)."""
    pos = positive_class_prf(preds, golds, positive=positive)
    neg = positive_class_prf(preds, golds, positive=("0" if positive == "1" else "1"))
    return float((pos["f1"] + neg["f1"]) / 2.0)


def hits_at_k_from_ranks(ranks: list[int], k: int) -> float:
    """Hits@k from a list of per-query ranks (1-indexed). rank=0 means not in candidates."""
    r = np.asarray(ranks, dtype=np.int64)
    valid = r > 0
    if not valid.any():
        return 0.0
    return float(((r > 0) & (r <= k)).sum() / len(r))


def mrr_from_ranks(ranks: list[int]) -> float:
    """MRR from a list of per-query ranks (1-indexed). rank=0 contributes 0."""
    r = np.asarray(ranks, dtype=np.float64)
    rr = np.where(r > 0, 1.0 / np.maximum(r, 1.0), 0.0)
    return float(rr.mean()) if len(rr) else 0.0


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
