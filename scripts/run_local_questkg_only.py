"""LOCAL-only: run pure-symbolic QUEST-KG on all 4 datasets at full N.

While the COLAB L4 notebook runs the 5 LLM baselines + quest_kg_llm with 7B,
the 1080 Ti runs the cheap symbolic-only QUEST-KG cells. ~30 min total.

Output: results/quest_kg__<dataset>__local-1080ti__symbolic__seedX.json/.csv
Different tag so it doesn't collide with the 3B cells already on disk.
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
    from quest_kg.symbolic.icews18_temporal import ICEWS18Checker
    from quest_kg.symbolic.orgaccess_datalog import OrgAccessChecker
    from quest_kg.symbolic.webqsp_freebase import FreebaseChecker

    OUT_DIR = ROOT / "results"
    TAG = "local-1080ti__symbolic__seed0"

    DATA_ROOT = ROOT / "data"
    encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2", batch_size=256)
    print(f"[qkg] encoder ready on {encoder.device}")

    # Per-dataset config (Iter 5):
    #   hops: WebQSP/ICEWS18 1, OrgAccess/CWQ 2 (ablation Table evidence)
    #   answer_aggregation: "max" for k=1 datasets, "sum" (path-vote) for k=2 QA
    DATASETS = {
        "orgaccess": dict(kg_subset=20000,  top_k=32,  bidir=True,  hops=2, agg="max", n=750,  task_type="yes_no", checker_kind="orgaccess"),
        # Iter 6b: bumped ICEWS18 N=150->200, WebQSP N=500->1000 to match
        # the N's used by the Colab L4 baselines (paper headline matched-N).
        "icews18":   dict(kg_subset=20000,  top_k=32,  bidir=False, hops=1, agg="max", n=200,  task_type="entity", checker_kind="icews18"),
        # WebQSP/CWQ: kg_subset=None to keep ALL triples from each query's
        # local HF graph. Inference uses query['graph_triple_idx'] to restrict
        # retrieval per query, matching graphrag's per-query-graph setup.
        "webqsp":    dict(kg_subset=None,   top_k=4,   bidir=True,  hops=1, agg="max", n=1000, task_type="entity", checker_kind="freebase"),
        "cwq":       dict(kg_subset=None,   top_k=8,   bidir=True,  hops=3, agg="sum", n=500,  task_type="entity", checker_kind="freebase"),
    }

    TASK_TYPE_MAP = {
        "orgaccess": "access_control",
        "icews18": "link_prediction",
        "webqsp": "qa",
        "cwq": "qa",
    }

    for ds_name, cfg in DATASETS.items():
        tag = f"quest_kg__{ds_name}__{TAG}"
        json_path = OUT_DIR / f"{tag}.json"
        csv_path  = OUT_DIR / f"{tag}.csv"
        if json_path.exists():
            print(f"[qkg]  SKIP {tag} (already done)")
            continue
        t_all = time.perf_counter()
        print(f"[qkg] === {ds_name} (n={cfg['n']}, kg={cfg['kg_subset']}, top_k={cfg['top_k']}, bidir={cfg['bidir']}) ===")
        ds = load_dataset(ds_name, str(DATA_ROOT))
        # WebQSP/CWQ: when queries carry per-query graphs (graph_triple_idx),
        # build the KG from the UNION of the first N per-query graphs only.
        # Avoids embedding 2M+ irrelevant triples; ensures every query's
        # local graph survives intact (kg_subset truncation would drop them).
        if cfg["kg_subset"] is None and ds.queries and ds.queries[0].get("graph_triple_idx"):
            n_eval = cfg["n"] or len(ds.queries)
            queries_eval = ds.queries[: n_eval]
            all_idx_set: set[int] = set()
            for q in queries_eval:
                all_idx_set.update(q.get("graph_triple_idx", []))
            sorted_idx = sorted(all_idx_set)
            remap = {old: new for new, old in enumerate(sorted_idx)}
            ds.triples = [ds.triples[i] for i in sorted_idx]
            for q in queries_eval:
                q["graph_triple_idx"] = [remap[i] for i in q.get("graph_triple_idx", []) if i in remap]
            print(f"[qkg]   per-query KG: union of first {n_eval} query graphs = {len(ds.triples)} triples")
        elif cfg["kg_subset"] and len(ds.triples) > cfg["kg_subset"]:
            ds.triples = ds.triples[: cfg["kg_subset"]]
        t0 = time.perf_counter()
        retriever = SchemaAwareRetriever(
            triples=ds.triples, encoder=encoder, k=cfg["hops"],
            top_k=cfg["top_k"], precompute=True, bidirectional=cfg["bidir"],
        )
        print(f"[qkg]   retriever ready in {time.perf_counter()-t0:.1f}s")

        if cfg["checker_kind"] == "orgaccess":
            ctx = {int(i+1): c for i, c in enumerate(ds.metadata.get("context_schedule", []))}
            checker = OrgAccessChecker(active_contexts_at_t=ctx, strict=False)
        elif cfg["checker_kind"] == "icews18":
            checker = ICEWS18Checker()
        else:
            checker = FreebaseChecker()

        use_ans_rescore = ds_name in ("webqsp", "cwq")
        method = QuestKG(
            retriever=retriever, symbolic_checker=checker,
            task_type=cfg["task_type"],
            answer_rescoring=use_ans_rescore,
            max_path_length=cfg["hops"],
            answer_aggregation=cfg.get("agg", "max"),
        )

        queries = ds.queries[: cfg["n"]] if cfg["n"] else ds.queries
        print(f"[qkg]   running on {len(queries)} queries...")
        t1 = time.perf_counter()
        results = run_method(method, queries, method_name="quest_kg", is_questkg=True)
        elapsed = time.perf_counter() - t1
        agg = aggregate(results, task_type=TASK_TYPE_MAP[ds_name])
        agg.update({
            "method": "quest_kg", "dataset": ds_name,
            "llm": "none", "encoder": encoder.model_name,
            "seed": 0, "wallclock_s": round(elapsed, 1),
            "limit": cfg["n"], "kg_subset": cfg["kg_subset"],
            "top_k": cfg["top_k"], "bidirectional": cfg["bidir"],
        })
        df = to_dataframe(results)
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(agg, indent=2))
        primary = agg.get("primary_metric", "?")
        primary_val = agg.get("primary_value", 0.0)
        print(f"[qkg]   OK {elapsed:.0f}s  {primary}={primary_val:.3f}  EM={agg['exact_match']:.3f}  total {time.perf_counter()-t_all:.0f}s")

    print(f"[qkg] all done")


if __name__ == "__main__":
    main()
