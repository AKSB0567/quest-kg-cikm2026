"""Phase 4 ablations: QUEST-KG component removal + hop sweep, LOCAL only.

QUEST-KG is symbolic so we don't need an LLM. Runs on 1080 Ti in ~1-2 hours
in parallel with the Colab L4 headline run.

Variants tested per dataset:
  - full              : current best (bidirectional, rel_bias=0.4, answer_rescoring)
  - no_bidir          : retrieval out-edges only
  - no_rel_bias       : retrieval w/ relation_bias=0
  - no_answer_rescore : disable answer-side cos rescoring
  - no_anchor_penalty : ablate anchor-overlap penalty (set in inference)

Plus hop sweep k in {1, 2, 3, 4}.

Output: results/_ablations_local__<tag>.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from quest_kg.data.encoders import SentenceTransformerEncoder
    from quest_kg.data.loaders import load_dataset
    from quest_kg.eval.harness import aggregate, run_method
    from quest_kg.inference import QuestKG
    from quest_kg.retrieval.schema_aware import SchemaAwareRetriever
    from quest_kg.symbolic.icews18_temporal import ICEWS18Checker
    from quest_kg.symbolic.orgaccess_datalog import OrgAccessChecker
    from quest_kg.symbolic.webqsp_freebase import FreebaseChecker

    enc = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
    print(f"[abl] encoder ready on {enc.device}")

    DATASETS = {
        "orgaccess": dict(kg=20000,  top_k=32,  bidir_default=True,  n=200, task="yes_no",
                          eval_task="access_control", checker="orgaccess"),
        "icews18":   dict(kg=20000,  top_k=32,  bidir_default=False, n=150, task="entity",
                          eval_task="link_prediction", checker="icews18"),
        "webqsp":    dict(kg=100000, top_k=128, bidir_default=True,  n=200, task="entity",
                          eval_task="qa", checker="freebase"),
        "cwq":       dict(kg=100000, top_k=64,  bidir_default=True,  n=200, task="entity",
                          eval_task="qa", checker="freebase"),
    }

    def build_checker(name, ds):
        if name == "orgaccess":
            ctx = {int(i + 1): c for i, c in enumerate(ds.metadata.get("context_schedule", []))}
            return OrgAccessChecker(active_contexts_at_t=ctx, strict=False)
        if name == "icews18":
            return ICEWS18Checker()
        return FreebaseChecker()

    rows = []

    for ds_name, cfg in DATASETS.items():
        print(f"\n[abl] === {ds_name} ===")
        ds = load_dataset(ds_name, str(ROOT / "data"))
        if cfg["kg"] and len(ds.triples) > cfg["kg"]:
            ds.triples = ds.triples[: cfg["kg"]]
        queries = ds.queries[: cfg["n"]]
        use_ans_default = ds_name in ("webqsp", "cwq")

        # Component-removal variants
        variants = [
            ("full",              dict(rel=0.4, bidir=cfg["bidir_default"], ans=use_ans_default, max_hops=2)),
            ("no_bidir",          dict(rel=0.4, bidir=False,                ans=use_ans_default, max_hops=2)),
            ("no_rel_bias",       dict(rel=0.0, bidir=cfg["bidir_default"], ans=use_ans_default, max_hops=2)),
            ("no_answer_rescore", dict(rel=0.4, bidir=cfg["bidir_default"], ans=False,           max_hops=2)),
        ]
        # Hop sweep
        for k in (1, 2, 3, 4):
            variants.append((f"hop_k={k}",
                             dict(rel=0.4, bidir=cfg["bidir_default"], ans=use_ans_default, max_hops=k)))

        for vname, vcfg in variants:
            t0 = time.perf_counter()
            retriever = SchemaAwareRetriever(
                triples=ds.triples, encoder=enc, k=vcfg["max_hops"],
                top_k=cfg["top_k"], precompute=True,
                bidirectional=vcfg["bidir"],
                relation_bias=vcfg["rel"],
            )
            checker = build_checker(cfg["checker"], ds)
            qkg = QuestKG(
                retriever=retriever, symbolic_checker=checker,
                task_type=cfg["task"],
                answer_rescoring=vcfg["ans"],
                max_path_length=vcfg["max_hops"],
            )
            results = run_method(qkg, queries, method_name=f"qkg_{vname}",
                                 is_questkg=True)
            agg = aggregate(results, task_type=cfg["eval_task"])
            elapsed = time.perf_counter() - t0
            rows.append({
                "variant": vname, "dataset": ds_name, "n": len(queries),
                "primary_metric": agg.get("primary_metric"),
                "primary_value": agg.get("primary_value"),
                "EM": agg.get("exact_match"),
                "balanced_accuracy": agg.get("balanced_accuracy"),
                "MRR": agg.get("mrr"),
                "mean_latency_ms": agg.get("mean_latency_ms"),
                "wallclock_s": round(elapsed, 1),
                "abstention_rate": agg.get("abstention_rate"),
            })
            print(f"  {vname:>22s}: {agg.get('primary_metric')}="
                  f"{agg.get('primary_value', 0):.3f}  EM={agg.get('exact_match', 0):.3f}  "
                  f"lat={agg.get('mean_latency_ms', 0):.0f}ms  wall={elapsed:.0f}s")
            del retriever
            import gc; gc.collect()

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "_ablations_local__local-1080ti__symbolic.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 200)
    print("\n=== ABLATION RESULTS (primary metric per dataset) ===")
    print(df.pivot_table(
        index="variant", columns="dataset", values="primary_value", aggfunc="first"
    ).round(3).to_string())
    print(f"\nWrote {len(rows)} rows to {out.name}")


if __name__ == "__main__":
    main()
