"""GraphRAG-style baseline.

Retrieve a small subgraph around query-mentioned entities (bounded k-hop
expansion) and feed the linearized subgraph to the LM. This is QUEST-KG's
Stage 1 retriever *without* the provenance and schema-consistency terms —
isolating the contribution of those signals.

Faithful to Edge et al. (Microsoft GraphRAG) at the local-search level;
the global-summary path is out of scope here (we focus on KG-RAG).
"""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np

from baselines.base import Baseline, BaselinePrediction
from quest_kg.core.types import Triple
from quest_kg.retrieval.schema_aware import SchemaAwareRetriever


class GraphRAG:
    name = "graphrag"

    def __init__(self, triples: Iterable[Triple], encoder, llm, k: int = 2, top_k: int = 16):
        self.triples = list(triples)
        self.encoder = encoder
        self.llm = llm
        # Pure-semantic retrieval: gammas = (1.0, 0.0, 0.0) gives weight only to s_sem
        self.retriever = SchemaAwareRetriever(
            triples=self.triples,
            encoder=encoder,
            k=k,
            top_k=top_k,
            eps_keep=0.0,
            gammas=(1.0, 0.0, 0.0),
        )

    def predict(self, query: str, **kwargs) -> BaselinePrediction:
        sg = self.retriever.retrieve(query)
        context = "\n".join(f"({t.s}, {t.r}, {t.o})" for t in sg.triples)
        prompt = (
            "Use the knowledge-graph subgraph below to answer the question. "
            "Output just the answer.\n\n"
            f"Subgraph (triples):\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        ans = self.llm.generate([prompt])[0]
        # Confidence proxy: mean retrieved edge score
        scores = list(sg.edge_scores.values())
        conf = float(np.mean(scores)) if scores else 0.5
        return BaselinePrediction(
            query=query, prediction=ans.split("\n")[0].strip(),
            confidence=conf, evidence=sg.triples,
        )
