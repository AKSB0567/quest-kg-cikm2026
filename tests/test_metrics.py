"""Sanity tests for evaluation metrics."""
import numpy as np
from quest_kg.eval.metrics import (
    aurc, balanced_accuracy, brier, ece, hits_at_k, hits_at_k_from_ranks,
    macro_f1_binary, mrr, mrr_from_ranks, nll, positive_class_prf,
)


def test_ece_perfect_calibration():
    # If confidence == accuracy exactly, ECE should be ~0.
    rng = np.random.default_rng(0)
    n = 1000
    conf = rng.uniform(0, 1, n)
    correct = rng.binomial(1, conf)
    assert ece(conf, correct, n_bins=20) < 0.05


def test_ece_overconfident():
    # Always predict confidence 0.99 but only 50% correct -> high ECE.
    conf = np.full(1000, 0.99)
    correct = np.zeros(1000); correct[:500] = 1
    assert ece(conf, correct, n_bins=20) > 0.4


def test_brier_perfect():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert brier(probs, targets) == 0.0


def test_hits_at_k_basic():
    # 3 queries, 4 candidates each, gold at known position
    scores = np.array([
        [0.9, 0.1, 0.1, 0.1],
        [0.1, 0.9, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.9],
    ])
    gold = np.array([0, 1, 3])
    assert hits_at_k(scores, gold, 1) == 1.0


def test_mrr_basic():
    scores = np.array([
        [0.9, 0.1, 0.1],     # gold at 0 -> rank 1
        [0.1, 0.9, 0.1],     # gold at 1 -> rank 1
    ])
    gold = np.array([0, 1])
    assert abs(mrr(scores, gold) - 1.0) < 1e-9


def test_aurc_ranges():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 100)
    correct = (conf > 0.5).astype(float)
    val = aurc(conf, correct)
    assert 0.0 <= val <= 1.0


def test_balanced_accuracy_majority_class_is_chance():
    # 91 negatives, 9 positives, predictor always says "0" -> EM=0.91 but BAcc=0.5
    preds = ["0"] * 100
    golds = ["0"] * 91 + ["1"] * 9
    assert balanced_accuracy(preds, golds) == 0.5


def test_balanced_accuracy_perfect():
    preds = ["1", "0", "1", "0"]
    golds = ["1", "0", "1", "0"]
    assert balanced_accuracy(preds, golds) == 1.0


def test_positive_class_prf_zero_when_majority():
    preds = ["0"] * 100
    golds = ["0"] * 91 + ["1"] * 9
    prf = positive_class_prf(preds, golds)
    assert prf["f1"] == 0.0
    assert prf["recall"] == 0.0


def test_macro_f1_binary():
    preds = ["1", "1", "0", "0"]
    golds = ["1", "0", "0", "0"]
    val = macro_f1_binary(preds, golds)
    # pos: P=0.5, R=1.0, F1=2/3 ; neg: P=1.0, R=2/3, F1=0.8 -> avg ~0.733
    assert abs(val - 0.7333333) < 1e-3


def test_hits_at_k_from_ranks():
    # 5 queries: ranks 1, 2, 3, 11, 0(not in candidates)
    assert hits_at_k_from_ranks([1, 2, 3, 11, 0], 1) == 0.2
    assert hits_at_k_from_ranks([1, 2, 3, 11, 0], 3) == 0.6
    assert hits_at_k_from_ranks([1, 2, 3, 11, 0], 10) == 0.6


def test_mrr_from_ranks():
    ranks = [1, 2, 0, 4]  # 1.0 + 0.5 + 0 + 0.25 = 1.75 / 4 = 0.4375
    assert abs(mrr_from_ranks(ranks) - 0.4375) < 1e-9


if __name__ == "__main__":
    test_ece_perfect_calibration()
    test_ece_overconfident()
    test_brier_perfect()
    test_hits_at_k_basic()
    test_mrr_basic()
    test_aurc_ranges()
    test_balanced_accuracy_majority_class_is_chance()
    test_balanced_accuracy_perfect()
    test_positive_class_prf_zero_when_majority()
    test_macro_f1_binary()
    test_hits_at_k_from_ranks()
    test_mrr_from_ranks()
    print("all metric tests OK")
