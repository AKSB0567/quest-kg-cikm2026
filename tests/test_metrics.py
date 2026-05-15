"""Sanity tests for evaluation metrics."""
import numpy as np
from quest_kg.eval.metrics import ece, brier, nll, aurc, hits_at_k, mrr


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


if __name__ == "__main__":
    test_ece_perfect_calibration()
    test_ece_overconfident()
    test_brier_perfect()
    test_hits_at_k_basic()
    test_mrr_basic()
    test_aurc_ranges()
    print("all metric tests OK")
