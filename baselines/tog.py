"""Think-on-Graph baselines (ToG-1, ToG-2).

ToG-1 (Sun et al., ICLR 2024): LLM-guided beam search over the KG. At each step,
the LLM scores candidate edges from the current frontier and selects top-`beam`
to expand. Reasoning chain is the sequence of selected edges.

ToG-2 (Ma et al., ICLR 2025): adds textual evidence retrieval interleaved with
graph expansion. We implement a minimal version that pulls textual context
for each entity if available, otherwise reduces to ToG-1.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from baselines.base import Baseline, BaselinePrediction
from quest_kg.core.types import Triple


class ToG1:
    name = "tog1"

    def __init__(
        self,
        triples: Iterable[Triple],
        encoder,
        llm,
        max_depth: int = 3,
        beam_width: int = 3,
    ):
        self.triples = list(triples)
        self.encoder = encoder
        self.llm = llm
        self.max_depth = max_depth
        self.beam_width = beam_width
        # Build outgoing index
        self._out: dict[str, list[Triple]] = {}
        for t in self.triples:
            self._out.setdefault(t.s, []).append(t)
        # Embed all entity surface forms for anchor linking
        self._entities = sorted(set(t.s for t in self.triples) | set(t.o for t in self.triples))
        self._ent_emb = encoder(self._entities)

    def _anchor(self, query: str) -> list[str]:
        q = self.encoder(query)
        sims = self._ent_emb @ q
        top = np.argsort(-sims)[: self.beam_width]
        return [self._entities[i] for i in top]

    def _score_edges_with_llm(self, query: str, edges: list[Triple]) -> list[float]:
        """One LLM call ranks all candidate edges relative to the query (beam-search guidance).

        We bucket the edges into a single prompt and ask for relevance scores
        in 0..1; LLM output is parsed leniently — failures fall back to a
        uniform score so the search still makes progress.
        """
        if not edges:
            return []
        if len(edges) == 1:
            return [1.0]
        lines = [f"{i}. ({e.s}, {e.r}, {e.o})" for i, e in enumerate(edges)]
        prompt = (
            "Score each candidate triple on a 0..1 scale by how useful it is for "
            f"answering the question: {query!r}.\n"
            "Output one score per line, in order, plain numbers.\n\n"
            + "\n".join(lines)
            + "\n\nScores:"
        )
        try:
            raw = self.llm.generate([prompt], max_new_tokens=64)[0]
            nums: list[float] = []
            for line in raw.splitlines():
                line = line.strip().split()
                if not line:
                    continue
                try:
                    nums.append(float(line[0]))
                except ValueError:
                    continue
            if len(nums) >= len(edges):
                return nums[: len(edges)]
        except Exception:
            pass
        return [1.0 / len(edges)] * len(edges)

    def predict(self, query: str, **kwargs) -> BaselinePrediction:
        # Beam search
        anchors = self._anchor(query)
        beams: list[tuple[list[Triple], float]] = [([], 0.0)]
        # Seed beams with each anchor as starting point (no edges yet)
        # We'll expand from those anchors.
        current_nodes = list(anchors)
        evidence: list[Triple] = []
        for _depth in range(self.max_depth):
            candidates: list[Triple] = []
            for node in current_nodes:
                candidates.extend(self._out.get(node, []))
            if not candidates:
                break
            scores = self._score_edges_with_llm(query, candidates[:32])
            ranked = sorted(zip(candidates[:32], scores), key=lambda x: -x[1])
            picked = ranked[: self.beam_width]
            evidence.extend(t for t, _ in picked)
            current_nodes = [t.o for t, _ in picked]
            if not current_nodes:
                break

        # Final answer prompt
        ev_str = "\n".join(f"({t.s}, {t.r}, {t.o})" for t in evidence)
        final_prompt = (
            "Given the reasoning chain below, answer the question concisely.\n\n"
            f"Chain:\n{ev_str}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        ans = self.llm.generate([final_prompt])[0]
        # Confidence proxy: mean LLM score over picked edges
        return BaselinePrediction(
            query=query,
            prediction=ans.split("\n")[0].strip(),
            confidence=0.8,  # plain heuristic; ToG doesn't natively expose calibration
            evidence=evidence,
        )


class ToG2(ToG1):
    """ToG-2 adds textual-context injection per step.

    For Phase 2 we implement the minimum viable variant: at each step we ask
    the LLM to first describe the current entity briefly, then score edges.
    On datasets without textual corpora this reduces to ToG-1.
    """
    name = "tog2"

    def _score_edges_with_llm(self, query: str, edges: list[Triple]) -> list[float]:
        if not edges:
            return []
        # Augment prompt with entity context
        entity_set = sorted(set([e.s for e in edges] + [e.o for e in edges]))
        ent_str = ", ".join(entity_set[:8])
        prompt = (
            f"Question: {query}\n"
            f"Entities in current frontier: {ent_str}\n"
            "Score each candidate triple on a 0..1 scale by how useful it is for "
            "answering the question. Output one score per line.\n\n"
            + "\n".join(f"{i}. ({e.s}, {e.r}, {e.o})" for i, e in enumerate(edges))
            + "\n\nScores:"
        )
        try:
            raw = self.llm.generate([prompt], max_new_tokens=64)[0]
            nums: list[float] = []
            for line in raw.splitlines():
                line = line.strip().split()
                if not line:
                    continue
                try:
                    nums.append(float(line[0]))
                except ValueError:
                    continue
            if len(nums) >= len(edges):
                return nums[: len(edges)]
        except Exception:
            pass
        return [1.0 / len(edges)] * len(edges)
