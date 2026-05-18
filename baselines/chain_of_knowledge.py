"""Chain-of-Knowledge (CoK) baseline.

Faithful to Li et al. (ICLR 2024) at a high level: multi-step LLM prompting
where each step retrieves grounding evidence (KG triples) and generates an
intermediate reasoning step. We cap at `max_steps` and concatenate the
intermediate justifications into the final answer prompt.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from baselines.base import Baseline, BaselinePrediction, ranking_from_triples
from quest_kg.core.types import Triple


class ChainOfKnowledge:
    name = "cok"

    def __init__(
        self,
        triples: Iterable[Triple],
        encoder,
        llm,
        max_steps: int = 3,
        top_k: int = 6,
    ):
        self.triples = list(triples)
        self.encoder = encoder
        self.llm = llm
        self.max_steps = max_steps
        self.top_k = top_k
        triple_texts = [f"({t.s}, {t.r}, {t.o})" for t in self.triples]
        self._triple_embs = encoder(triple_texts)

    def _retrieve(self, query: str, exclude_idx: set[int]) -> list[tuple[int, Triple]]:
        q_emb = self.encoder(query)
        sims = self._triple_embs @ q_emb
        order = np.argsort(-sims)
        out = []
        for i in order:
            if int(i) in exclude_idx:
                continue
            out.append((int(i), self.triples[int(i)]))
            if len(out) >= self.top_k:
                break
        return out

    def predict(self, query: str, **kwargs) -> BaselinePrediction:
        chain: list[Triple] = []
        used: set[int] = set()
        cur_query = query
        for _step in range(self.max_steps):
            retrieved = self._retrieve(cur_query, used)
            if not retrieved:
                break
            for idx, _ in retrieved:
                used.add(idx)
            ctx = "\n".join(f"({t.s}, {t.r}, {t.o})" for _, t in retrieved)
            prompt = (
                "Continue the reasoning chain toward answering the question. "
                "Output the next intermediate sub-question or fact, on one line.\n\n"
                f"Original question: {query}\n"
                f"Current query: {cur_query}\n"
                f"Evidence:\n{ctx}\n\n"
                "Next step:"
            )
            try:
                step = self.llm.generate([prompt], max_new_tokens=64)[0]
                cur_query = step.split("\n")[0].strip()[:160] or query
                chain.extend(t for _, t in retrieved)
            except Exception:
                break

        ev_str = "\n".join(f"({t.s}, {t.r}, {t.o})" for t in chain)
        final_prompt = (
            "Using the reasoning chain below, give the final answer to the original question.\n\n"
            f"Chain:\n{ev_str}\n\n"
            f"Original question: {query}\n"
            "Final answer:"
        )
        ans = self.llm.generate([final_prompt])[0]
        return BaselinePrediction(
            query=query,
            prediction=ans.split("\n")[0].strip(),
            confidence=0.75,
            evidence=chain,
            extra={"candidate_ranking": ranking_from_triples(chain)},
        )
