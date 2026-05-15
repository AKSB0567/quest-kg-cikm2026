"""Vanilla retrieval-augmented generation baseline.

Dense retrieval over the KG (linearized as text) + LLM answer generation.
No graph structure — just top-K triples concatenated into a prompt.

Faithful to the standard RAG protocol (Lewis et al., NeurIPS 2020) when
applied to a KG corpus: each triple becomes a "document" `"<s> <r> <o>"`,
retrieval scores by cosine similarity, top-K is passed to the LM.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from baselines.base import Baseline, BaselinePrediction
from quest_kg.core.types import Triple


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


class VanillaRAG:
    name = "vanilla_rag"

    def __init__(self, triples: Iterable[Triple], encoder, llm, top_k: int = 8):
        self.triples = list(triples)
        self.encoder = encoder
        self.llm = llm
        self.top_k = top_k
        # Pre-embed triples once
        triple_texts = [self._fmt(t) for t in self.triples]
        self._triple_embs = encoder(triple_texts)

    @staticmethod
    def _fmt(t: Triple) -> str:
        return f"({t.s}, {t.r}, {t.o})"

    def _retrieve(self, query: str) -> list[Triple]:
        q_emb = self.encoder(query)
        sims = self._triple_embs @ q_emb  # (N,)
        order = np.argsort(-sims)[: self.top_k]
        return [self.triples[i] for i in order]

    def predict(self, query: str, **kwargs) -> BaselinePrediction:
        retrieved = self._retrieve(query)
        context = "\n".join(self._fmt(t) for t in retrieved)
        prompt = (
            "You are a knowledge graph QA system. Use the facts below to answer "
            "the question. Be concise — output just the answer entity or value.\n\n"
            f"Facts:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        ans = self.llm.generate([prompt])[0]
        # Confidence proxy: top-K mean retrieval similarity (∈ [0,1] for normalized embeddings)
        q_emb = self.encoder(query)
        sims = sorted(self._triple_embs @ q_emb, reverse=True)[: self.top_k]
        conf = float(np.clip(np.mean(sims), 0.0, 1.0))
        return BaselinePrediction(
            query=query, prediction=ans.split("\n")[0].strip(),
            confidence=conf, evidence=retrieved,
        )
