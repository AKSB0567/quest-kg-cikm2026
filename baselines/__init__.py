"""Baseline reproductions for matched-condition comparison.

All baselines implement the `Baseline` interface so the eval harness can
swap them in/out of the same retrieval+LLM setup as QUEST-KG.
"""
from .base import Baseline, BaselinePrediction
from .vanilla_rag import VanillaRAG
from .graphrag import GraphRAG
from .tog import ToG1, ToG2
from .chain_of_knowledge import ChainOfKnowledge

__all__ = [
    "Baseline",
    "BaselinePrediction",
    "VanillaRAG",
    "GraphRAG",
    "ToG1",
    "ToG2",
    "ChainOfKnowledge",
]
