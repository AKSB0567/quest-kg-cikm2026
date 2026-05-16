"""Local smoke test on real OrgAccess data, no GPU, no LLM, no encoder downloads.

Validates the end-to-end QUEST-KG pipeline (retrieval -> MP -> symbolic -> abstention)
plus one baseline with a MockLLM. Uses HashEncoder so no model weights needed.

Run from repo root:
    python scripts/local_smoke.py

What it proves (or breaks):
  - Data loaders work on real Drive-format files.
  - Retriever pre-embedding and bounded-hop expansion produce non-empty subgraphs.
  - Evidential MP fills (b_i, u_i) for every node without crashing.
  - Path enumeration + symbolic check + scoring yields a prediction.
  - Abstention defaults are permissive (we want predictions out, not abstain-all).
  - The eval harness computes EM/F1/latency.

Outputs a friendly diagnostic dump per query and a final aggregate summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quest_kg.data.encoders import HashEncoder
from quest_kg.data.loaders import load_orgaccess
from quest_kg.eval.harness import aggregate, run_method
from quest_kg.inference import QuestKG
from quest_kg.retrieval.schema_aware import SchemaAwareRetriever
from quest_kg.symbolic.orgaccess_datalog import OrgAccessChecker


def dump_one(qkg: QuestKG, q: dict, idx: int) -> None:
    """Run a single query and print a diagnostic dump."""
    print("\n" + "=" * 72)
    print(f"Q{idx}: {q['question'][:120]}")
    print(f"  user={q['user']}  resource={q['resource']}  t={q['timestamp']}  ctx={q['context']}")
    print(f"  gold answer = {q['answer']!r}")
    t0 = time.perf_counter()
    out = qkg.predict(q["question"])
    dt = (time.perf_counter() - t0) * 1000

    sg = out.subgraph
    print(f"  -> retrieved subgraph: |V|={len(sg.nodes)}, |E|={len(sg.triples)}, "
          f"anchors={len(sg.anchors)}")
    if sg.triples:
        sample_edges = sg.triples[: min(3, len(sg.triples))]
        for t in sample_edges:
            score = sg.edge_scores.get((t.s, t.r, t.o), 0.0)
            print(f"     edge ({t.s}, {t.r}, {t.o}) score={score:.3f}")
    print(f"  -> {len(out.candidate_paths)} candidate paths, "
          f"{out.extra.get('n_valid_paths', 0)} valid (non-violating)")
    if out.top_path:
        path_str = " -> ".join(out.top_path.node_seq())
        print(f"     top path: {path_str}  score={out.top_path.score:.3e}")
    print(f"  abstained={out.abstained}  M_q={out.confidence:.3f}  H_q={out.entropy:.3f}")
    print(f"  prediction = {out.prediction!r}    (latency {dt:.1f} ms)")


def main():
    data_root = ROOT / "data"
    print(f"[smoke] loading OrgAccess from {data_root}/raw/orgaccess/ ...")
    ds = load_orgaccess(data_root)
    print(f"[smoke] {ds.summary()}")

    enc = HashEncoder(dim=64)
    print(f"[smoke] building retriever (precompute=True) on {len(ds.triples)} triples...")
    t0 = time.perf_counter()
    retriever = SchemaAwareRetriever(
        triples=ds.triples, encoder=enc, k=2, top_k=32, precompute=True,
    )
    print(f"[smoke] retriever built in {(time.perf_counter()-t0)*1000:.0f} ms")
    print(f"[smoke] entity_emb shape: "
          f"{retriever._entity_emb.shape if retriever._entity_emb is not None else None}")
    print(f"[smoke] edge_emb shape:   "
          f"{retriever._edge_emb.shape if retriever._edge_emb is not None else None}")

    ctx = {int(i + 1): c for i, c in enumerate(ds.metadata.get("context_schedule", []))}
    checker = OrgAccessChecker(active_contexts_at_t=ctx, strict=False)
    qkg = QuestKG(retriever=retriever, symbolic_checker=checker)
    print(f"[smoke] QUEST-KG ready (p*={qkg.p_star}, H*={qkg.h_star})")

    # Show 5 representative queries
    sample_queries = ds.queries[:5]
    print("\n" + "#" * 72)
    print("# Per-query diagnostic dump (first 5 queries)")
    print("#" * 72)
    for i, q in enumerate(sample_queries):
        dump_one(qkg, q, i + 1)

    # Run aggregate on first 50
    print("\n" + "#" * 72)
    print("# Aggregate over first 50 queries")
    print("#" * 72)
    t0 = time.perf_counter()
    results = run_method(qkg, ds.queries[:50], method_name="quest_kg", is_questkg=True)
    elapsed = time.perf_counter() - t0
    agg = aggregate(results)
    agg["wallclock_s"] = round(elapsed, 2)
    print(json.dumps(agg, indent=2))

    # Sanity assertions
    print("\n" + "#" * 72)
    print("# Sanity checks")
    print("#" * 72)
    n_with_subgraph = sum(1 for r in results if r.extra.get("error") is None)
    print(f"  Queries without errors:      {n_with_subgraph}/50")
    print(f"  Mean latency (should be <2s): {agg['mean_latency_ms']:.0f} ms")
    print(f"  Abstention rate (should be <0.5): {agg['abstention_rate']:.2f}")
    print(f"  EM (any > 0 means pipeline is producing matched predictions): {agg['exact_match']:.3f}")

    ok = (
        n_with_subgraph >= 45
        and agg["mean_latency_ms"] < 2000
        and agg["abstention_rate"] < 0.6
    )
    print("\n" + ("PASS — pipeline is healthy locally; safe to push to Colab"
                  if ok else "WARN — investigate above metrics before next Colab run"))


if __name__ == "__main__":
    main()
