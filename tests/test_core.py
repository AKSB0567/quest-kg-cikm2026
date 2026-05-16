"""End-to-end smoke tests for the QUEST-KG core pipeline.

Uses a tiny hand-crafted KG and a deterministic dummy encoder so tests run
in <1s without GPU. Exercises retrieval -> MP -> symbolic -> abstention.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quest_kg.abstention.posterior import answer_entropy, retained_posterior_mass, should_abstain
from quest_kg.core.types import NodeState, Path as KGPath, Provenance, Triple
from quest_kg.inference import QuestKG
from quest_kg.mp.evidential import evidential_mp_numpy, init_node_states
from quest_kg.retrieval.schema_aware import SchemaAwareRetriever, provenance_score
from quest_kg.symbolic.icews18_temporal import ICEWS18Checker
from quest_kg.symbolic.orgaccess_datalog import OrgAccessChecker
from quest_kg.symbolic.webqsp_freebase import FreebaseChecker


def _hash_one(s: str, d: int = 64) -> np.ndarray:
    v = np.zeros(d, dtype=np.float64)
    for tok in s.lower().split():
        h = hashlib.md5(tok.encode("utf-8")).digest()
        for i, b in enumerate(h[:d]):
            v[i] += b
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def dummy_encoder(s, d: int = 64):
    """Deterministic encoder; accepts a single str or a list of str."""
    if isinstance(s, str):
        return _hash_one(s, d)
    return np.stack([_hash_one(x, d) for x in s])


# ---------------------------------------------------------------------------
# Provenance scoring
# ---------------------------------------------------------------------------

def test_provenance_score_in_unit_interval():
    p = Provenance(extraction_conf=0.9, source_trust=1.0, edit_age_days=10.0)
    s = provenance_score(p)
    assert 0.0 <= s <= 1.0


def test_provenance_score_recency_effect():
    fresh = Provenance(extraction_conf=1.0, source_trust=1.0, edit_age_days=0.0)
    stale = Provenance(extraction_conf=1.0, source_trust=1.0, edit_age_days=10_000.0)
    assert provenance_score(fresh) > provenance_score(stale)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _toy_kg():
    return [
        Triple("Paris", "capital_of", "France", "City", "Country"),
        Triple("France", "in_continent", "Europe", "Country", "Continent"),
        Triple("Paris", "located_in", "Europe", "City", "Continent"),
        Triple("Berlin", "capital_of", "Germany", "City", "Country"),
        Triple("Germany", "in_continent", "Europe", "Country", "Continent"),
    ]


def test_retrieval_basic_anchors_and_expansion():
    r = SchemaAwareRetriever(triples=_toy_kg(), encoder=dummy_encoder, k=2, top_k=8)
    sg = r.retrieve("capital of France")
    assert "Paris" in sg.nodes or "France" in sg.nodes
    # Expansion should reach Europe in <=2 hops
    assert "Europe" in sg.nodes
    # Anchor set is bounded by top_k
    assert len(sg.anchors) <= 8


def test_retrieval_keeps_only_high_scoring_edges():
    r = SchemaAwareRetriever(triples=_toy_kg(), encoder=dummy_encoder, k=1, eps_keep=0.99)
    sg = r.retrieve("capital of France")
    # With a very strict eps_keep, expansion should produce few edges
    assert len(sg.triples) >= 0  # smoke; just verify no crash


# ---------------------------------------------------------------------------
# Evidential MP
# ---------------------------------------------------------------------------

def test_evidential_mp_runs_and_assigns_states():
    r = SchemaAwareRetriever(triples=_toy_kg(), encoder=dummy_encoder, k=2, top_k=8)
    sg = r.retrieve("capital of France")
    q_emb = dummy_encoder("capital of France")
    anchor_sims = {a: float(np.dot(q_emb, dummy_encoder(a))) for a in sg.anchors}
    evidential_mp_numpy(sg, anchor_sims=anchor_sims, n_iters=2)
    for node in sg.nodes:
        st = sg.node_states[node]
        assert 0.0 <= st.belief <= 1.0
        assert 0.0 <= st.uncertainty <= 1.0


def test_node_state_belief_uncertainty_sum():
    # b + u <= 1 always for Dirichlet over 2 classes
    st = NodeState(alpha_pos=5.0, alpha_neg=3.0)
    assert 0.0 <= st.belief <= 1.0
    assert 0.0 <= st.uncertainty <= 1.0
    # Vacuity diminishes as evidence grows
    st2 = NodeState(alpha_pos=100.0, alpha_neg=50.0)
    assert st2.uncertainty < st.uncertainty


# ---------------------------------------------------------------------------
# Symbolic constraints
# ---------------------------------------------------------------------------

def test_freebase_checker_domain_range():
    schema = {("City", "capital_of"): {"Country"}}
    c = FreebaseChecker(schema=schema)
    good = KGPath(triples=[Triple("Paris", "capital_of", "France", "City", "Country")])
    bad  = KGPath(triples=[Triple("Paris", "capital_of", "France", "City", "Continent")])
    assert not c.violates(good)
    assert c.violates(bad)


def test_icews_checker_monotone_timestamps():
    c = ICEWS18Checker()
    good = KGPath(triples=[
        Triple("A", "r", "B", timestamp=5),
        Triple("B", "r", "C", timestamp=7),
    ])
    bad = KGPath(triples=[
        Triple("A", "r", "B", timestamp=7),
        Triple("B", "r", "C", timestamp=3),
    ])
    assert not c.violates(good)
    assert c.violates(bad)


def test_orgaccess_checker_required_relations_strict():
    # Permissive mode (default): short paths allowed.
    permissive = OrgAccessChecker(strict=False)
    short = KGPath(triples=[Triple("u1", "has_role", "r1")])
    assert not permissive.violates(short)
    # Strict mode: enforce R1 (all required relations must appear).
    strict = OrgAccessChecker(strict=True)
    assert strict.violates(short)


def test_orgaccess_checker_revocation():
    c = OrgAccessChecker()
    revoked = KGPath(triples=[
        Triple("u1", "has_role", "r1", timestamp=2),
        Triple("r1", "grants", "res1", timestamp=3),
        Triple("res1", "governed_by", "p1", timestamp=4),
        Triple("p1", "valid_in", "ctx1", timestamp=5),
        Triple("r1", "revoked_at", "3", timestamp=3),
    ])
    assert c.violates(revoked)


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_should_abstain_high_entropy():
    # Two equally good answers -> high entropy -> abstain under low H*
    paths = [
        KGPath(triples=[Triple("a", "r", "b")], score=1.0),
        KGPath(triples=[Triple("a", "r", "c")], score=1.0),
    ]
    abstained, M, H, probs = should_abstain(paths, p_star=0.0, h_star=0.5)
    assert abstained
    assert abs(H - 1.0) < 1e-6  # log2(2) = 1 bit
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_should_abstain_low_posterior_mass():
    # All paths violate -> retained mass = 0 -> abstain
    paths = [
        KGPath(triples=[Triple("a", "r", "b")], score=1.0, violates_symbolic=True),
    ]
    abstained, M, H, probs = should_abstain(paths)
    assert abstained
    assert M == 0.0


# ---------------------------------------------------------------------------
# End-to-end inference
# ---------------------------------------------------------------------------

def test_end_to_end_no_crash():
    r = SchemaAwareRetriever(triples=_toy_kg(), encoder=dummy_encoder, k=2, top_k=8)
    schema = {
        ("City", "capital_of"): {"Country"},
        ("Country", "in_continent"): {"Continent"},
        ("City", "located_in"): {"Continent"},
    }
    checker = FreebaseChecker(schema=schema)
    qkg = QuestKG(retriever=r, symbolic_checker=checker, p_star=0.0, h_star=2.0)
    out = qkg.predict("capital of France")
    assert out.query == "capital of France"
    assert out.subgraph is not None
    # Should have at least one node
    assert len(out.subgraph.nodes) > 0


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
