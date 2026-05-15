"""Phase 2 smoke tests: data loaders + eval harness + baselines on a tiny synthetic dataset.

LLM-dependent tests use a mock that returns a canned string.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quest_kg.core.types import Triple
from quest_kg.data.dataset import Dataset
from quest_kg.data.encoders import HashEncoder
from quest_kg.eval.harness import aggregate, exact_match, run_method, token_f1


class MockLLM:
    """LLM stub that returns a fixed answer regardless of prompt."""

    def __init__(self, answer: str = "Paris"):
        self.answer = answer

    def generate(self, prompts, batch_size=4, max_new_tokens=None):
        return [self.answer for _ in prompts]


def _toy_kg():
    return [
        Triple("Paris", "capital_of", "France", "City", "Country"),
        Triple("France", "in_continent", "Europe", "Country", "Continent"),
        Triple("Paris", "located_in", "Europe", "City", "Continent"),
        Triple("Berlin", "capital_of", "Germany", "City", "Country"),
    ]


def _toy_queries():
    return [
        {"qid": "q1", "question": "capital of France", "answer": "Paris",
         "all_answers": ["Paris"]},
        {"qid": "q2", "question": "what is France in",  "answer": "Europe",
         "all_answers": ["Europe", "EU"]},
    ]


# --- harness ---------------------------------------------------------------

def test_exact_match():
    assert exact_match("Paris", "paris") == 1
    assert exact_match("paris.", "paris") == 1
    assert exact_match("Paris", ["Paris", "paris"]) == 1
    assert exact_match("London", "Paris") == 0


def test_token_f1():
    assert token_f1("Paris France", "Paris") > 0.5
    assert token_f1("Paris France", ["Berlin", "Paris"]) > 0.5
    assert token_f1("", "Paris") == 0.0


# --- baselines -------------------------------------------------------------

def test_vanilla_rag_runs():
    from baselines.vanilla_rag import VanillaRAG
    enc = HashEncoder(dim=64)
    llm = MockLLM("Paris")
    method = VanillaRAG(triples=_toy_kg(), encoder=enc, llm=llm, top_k=3)
    out = method.predict("capital of France")
    assert out.prediction == "Paris"
    assert 0.0 <= out.confidence <= 1.0
    assert len(out.evidence) > 0


def test_graphrag_runs():
    from baselines.graphrag import GraphRAG
    enc = HashEncoder(dim=64)
    llm = MockLLM("Paris")
    method = GraphRAG(triples=_toy_kg(), encoder=enc, llm=llm, k=2, top_k=4)
    out = method.predict("capital of France")
    assert out.prediction == "Paris"


def test_tog1_runs():
    from baselines.tog import ToG1
    enc = HashEncoder(dim=64)
    llm = MockLLM("Paris")
    method = ToG1(triples=_toy_kg(), encoder=enc, llm=llm, max_depth=2, beam_width=2)
    out = method.predict("capital of France")
    assert out.prediction == "Paris"


def test_tog2_runs():
    from baselines.tog import ToG2
    enc = HashEncoder(dim=64)
    llm = MockLLM("Paris")
    method = ToG2(triples=_toy_kg(), encoder=enc, llm=llm, max_depth=2, beam_width=2)
    out = method.predict("capital of France")
    assert out.prediction == "Paris"


def test_cok_runs():
    from baselines.chain_of_knowledge import ChainOfKnowledge
    enc = HashEncoder(dim=64)
    llm = MockLLM("Paris")
    method = ChainOfKnowledge(triples=_toy_kg(), encoder=enc, llm=llm, max_steps=2, top_k=3)
    out = method.predict("capital of France")
    assert out.prediction == "Paris"


# --- run-method harness ----------------------------------------------------

def test_run_method_and_aggregate():
    from baselines.vanilla_rag import VanillaRAG
    enc = HashEncoder(dim=64)
    llm = MockLLM("Paris")
    method = VanillaRAG(triples=_toy_kg(), encoder=enc, llm=llm, top_k=3)
    results = run_method(method, _toy_queries(), method_name="vanilla_rag")
    assert len(results) == 2
    assert results[0].em == 1  # "Paris"
    # Second query expects "Europe" but mock returns "Paris" -> EM=0
    assert results[1].em == 0
    agg = aggregate(results)
    assert agg["n"] == 2
    assert 0.0 <= agg["exact_match"] <= 1.0


# --- dataset shim ----------------------------------------------------------

def test_dataset_summary():
    ds = Dataset(name="toy", task="qa", triples=_toy_kg(), queries=_toy_queries())
    s = ds.summary()
    assert s["n_triples"] == 4
    assert s["n_queries"] == 2


# --- end-to-end with QUEST-KG on dataset shim ------------------------------

def test_questkg_on_dataset():
    from quest_kg.inference import QuestKG
    from quest_kg.retrieval.schema_aware import SchemaAwareRetriever
    from quest_kg.symbolic.webqsp_freebase import FreebaseChecker

    enc = HashEncoder(dim=64)
    retriever = SchemaAwareRetriever(triples=_toy_kg(), encoder=enc, k=2, top_k=4)
    qkg = QuestKG(retriever=retriever, symbolic_checker=FreebaseChecker(),
                  p_star=0.0, h_star=10.0)
    results = run_method(qkg, _toy_queries(), method_name="quest_kg", is_questkg=True)
    assert len(results) == 2


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} Phase 2 tests passed.")
