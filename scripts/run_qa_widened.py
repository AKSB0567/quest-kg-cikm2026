"""WebQSP/CWQ with widened retrieval (top_k 4/8 -> 32/64) — testing whether
retrieval recall is the bottleneck for the QA SOTA gap.

Locked config has top_k=4 (WebQSP) / top_k=8 (CWQ). Recall at these tight
budgets is ~36% (per the finetune smoke). Widening should lift recall,
which is the upper bound on Hits@1 for any candidate-selector approach.

Methodology stays unified — only the retrieval top_k is the variable. Same
schema-aware retrieval, same evidential MP, same neuro-symbolic check.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import torch  # noqa
    from quest_kg.data.encoders import SentenceTransformerEncoder
    from quest_kg.data.loaders import load_dataset
    from quest_kg.eval.harness import aggregate, run_method, to_dataframe
    from quest_kg.inference import QuestKG
    from quest_kg.retrieval.schema_aware import SchemaAwareRetriever
    from quest_kg.symbolic.webqsp_freebase import FreebaseChecker

    OUT_DIR = ROOT / "results"

    DATA_ROOT = ROOT / "data"
    encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2",
                                          batch_size=256)
    print(f"[qa-widened] encoder ready on {encoder.device}")

    # SWEEP: try multiple top_k values
    CONFIGS = [
        # (dataset, n, top_k, hops, agg, smoke_n)
        ("webqsp",  500, 32, 1, "max", 100),
        ("webqsp",  500, 64, 1, "max", 100),
        ("cwq",     500, 32, 3, "sum", 100),
        ("cwq",     500, 64, 3, "sum", 100),
    ]

    print("Running smoke tests first (N=100), then full N=500 if smoke improves.")
    smoke_results = {}
    full_results = {}

    for ds_name, n_full, top_k, hops, agg, n_smoke in CONFIGS:
        cfg_id = f"{ds_name}_topk{top_k}"
        print(f"\n{'='*60}\n[smoke] {cfg_id} N={n_smoke}\n{'='*60}")

        # Load + restrict
        ds = load_dataset(ds_name, str(DATA_ROOT))
        queries_eval = ds.queries[: n_full]  # for stable per-query graph union
        if ds.queries and ds.queries[0].get("graph_triple_idx"):
            all_idx = set()
            for q in queries_eval:
                all_idx.update(q.get("graph_triple_idx", []))
            sorted_idx = sorted(all_idx)
            remap = {old: new for new, old in enumerate(sorted_idx)}
            ds.triples = [ds.triples[i] for i in sorted_idx]
            for q in queries_eval:
                q["graph_triple_idx"] = [remap[i] for i in q.get("graph_triple_idx", []) if i in remap]
            print(f"  per-query KG union: {len(ds.triples)} triples")

        t0 = time.perf_counter()
        retriever = SchemaAwareRetriever(
            triples=ds.triples, encoder=encoder, k=hops,
            top_k=top_k, precompute=True, bidirectional=True,
        )
        print(f"  retriever ready in {time.perf_counter()-t0:.1f}s (top_k={top_k}, hops={hops})")

        method = QuestKG(
            retriever=retriever, symbolic_checker=FreebaseChecker(),
            task_type="entity", answer_rescoring=True,
            max_path_length=hops, answer_aggregation=agg,
        )

        # Smoke
        smoke_queries = ds.queries[:n_smoke]
        t1 = time.perf_counter()
        results = run_method(method, smoke_queries,
                              method_name="quest_kg_widened", is_questkg=True)
        smoke_agg = aggregate(results, task_type="qa")
        smoke_h1 = smoke_agg["primary_value"]
        smoke_results[cfg_id] = smoke_h1
        print(f"  smoke H@1 = {smoke_h1:.3f} (N={n_smoke}, "
              f"{time.perf_counter()-t1:.1f}s)")

        # Decide: only run full if smoke is promising
        # WebQSP locked baseline at N=500 = 0.366. CWQ = 0.204.
        # Run full if smoke is at least within 0.02 of the locked baseline.
        BASELINE = {"webqsp": 0.366, "cwq": 0.204}
        if smoke_h1 < BASELINE[ds_name] - 0.05:
            print(f"  SKIP full: smoke {smoke_h1:.3f} too far below baseline {BASELINE[ds_name]}")
            continue

        # Full
        full_queries = ds.queries[:n_full]
        t1 = time.perf_counter()
        results = run_method(method, full_queries,
                              method_name="quest_kg_widened", is_questkg=True)
        full_agg = aggregate(results, task_type="qa")
        full_h1 = full_agg["primary_value"]
        full_results[cfg_id] = full_h1
        elapsed = time.perf_counter() - t1
        print(f"  full H@1 = {full_h1:.3f} (N={n_full}, {elapsed:.1f}s)")

        # Save
        full_agg.update({
            "method": "quest_kg_widened", "dataset": ds_name,
            "llm": "none", "encoder": encoder.model_name,
            "seed": 0, "wallclock_s": round(elapsed, 1),
            "limit": n_full, "kg_subset": None,
            "top_k": top_k, "hops": hops, "agg": agg,
            "bidirectional": True,
        })
        tag = f"local-1080ti__widened-topk{top_k}__seed0"
        json_p = OUT_DIR / f"quest_kg__{ds_name}__{tag}.json"
        csv_p  = OUT_DIR / f"quest_kg__{ds_name}__{tag}.csv"
        df = to_dataframe(results)
        df.to_csv(csv_p, index=False)
        json_p.write_text(json.dumps(full_agg, indent=2))
        print(f"  wrote {json_p.name}")

    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)
    print("\nSmoke (N=100):")
    for k, v in smoke_results.items():
        print(f"  {k}: {v:.3f}")
    print("\nFull (N=500):")
    for k, v in full_results.items():
        print(f"  {k}: {v:.3f}")
    print(f"\nBaseline locked: WebQSP top_k=4 -> 0.366, CWQ top_k=8 -> 0.204")


if __name__ == "__main__":
    main()
