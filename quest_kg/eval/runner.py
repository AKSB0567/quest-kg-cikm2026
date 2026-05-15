"""CLI runner: instantiate any method (quest-kg or baseline), run on a dataset,
write results to CSV + JSON summary.

Usage (from Colab):
    !python -m quest_kg.eval.runner \\
        --method quest_kg --dataset webqsp --llm qwen25-7b --seed 0 \\
        --data_root /content/drive/MyDrive/quest_kg/data \\
        --models_root /content/drive/MyDrive/quest_kg/models \\
        --out_dir /content/drive/MyDrive/quest_kg/results
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Lazy imports inside main() so this module doesn't pull heavy deps on import


METHOD_CHOICES = [
    "quest_kg",
    "vanilla_rag",
    "graphrag",
    "tog1",
    "tog2",
    "cok",
]


def build_method(method_name: str, dataset, encoder, llm, **kwargs):
    """Instantiate a method given the encoder + LLM + dataset."""
    triples = dataset.triples
    if method_name == "quest_kg":
        from quest_kg.inference import QuestKG
        from quest_kg.retrieval.schema_aware import SchemaAwareRetriever
        from quest_kg.symbolic.icews18_temporal import ICEWS18Checker
        from quest_kg.symbolic.orgaccess_datalog import OrgAccessChecker
        from quest_kg.symbolic.webqsp_freebase import FreebaseChecker

        retriever = SchemaAwareRetriever(triples=triples, encoder=encoder, k=2, top_k=32)
        if dataset.name in ("webqsp", "cwq"):
            checker = FreebaseChecker()  # permissive schema; tighten in §4.5
        elif dataset.name == "icews18":
            checker = ICEWS18Checker()
        elif dataset.name == "orgaccess":
            ctx = {int(i + 1): c for i, c in enumerate(dataset.metadata.get("context_schedule", []))}
            checker = OrgAccessChecker(active_contexts_at_t=ctx)
        else:
            checker = FreebaseChecker()
        return QuestKG(retriever=retriever, symbolic_checker=checker, p_star=0.05, h_star=0.6), True

    if method_name == "vanilla_rag":
        from baselines.vanilla_rag import VanillaRAG
        return VanillaRAG(triples, encoder, llm), False
    if method_name == "graphrag":
        from baselines.graphrag import GraphRAG
        return GraphRAG(triples, encoder, llm), False
    if method_name == "tog1":
        from baselines.tog import ToG1
        return ToG1(triples, encoder, llm), False
    if method_name == "tog2":
        from baselines.tog import ToG2
        return ToG2(triples, encoder, llm), False
    if method_name == "cok":
        from baselines.chain_of_knowledge import ChainOfKnowledge
        return ChainOfKnowledge(triples, encoder, llm), False
    raise KeyError(f"unknown method {method_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHOD_CHOICES)
    ap.add_argument("--dataset", required=True, choices=["webqsp", "cwq", "icews18", "orgaccess"])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--llm", default="qwen25-7b")
    ap.add_argument("--encoder", default="e5-large-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N queries (0 = all)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}__{args.dataset}__{args.llm}__seed{args.seed}"
    csv_path = out_dir / f"{tag}.csv"
    json_path = out_dir / f"{tag}.json"

    # ---- Load data + encoder + LLM ----
    from quest_kg.data.loaders import load_dataset as load_ds
    from quest_kg.data.encoders import get_encoder
    from quest_kg.data.llm import LLMInterface, llm_path_for
    from quest_kg.eval.harness import run_method, aggregate, to_dataframe

    print(f"[runner] loading dataset {args.dataset}...")
    ds = load_ds(args.dataset, args.data_root)
    print(f"[runner] {ds.summary()}")

    print(f"[runner] loading encoder {args.encoder}...")
    encoder = get_encoder(args.encoder)

    if args.method == "quest_kg":
        # QUEST-KG (inference-time) does not need an LLM; uses neuro-symbolic scoring.
        llm = None
    else:
        print(f"[runner] loading LLM {args.llm}...")
        llm_path = llm_path_for(args.llm, drive_root=str(Path(args.models_root).parent))
        llm = LLMInterface(llm_path, max_new_tokens=args.max_new_tokens)

    method, is_questkg = build_method(args.method, ds, encoder, llm)
    print(f"[runner] method instantiated: {method.__class__.__name__}")

    queries = ds.queries[: args.limit] if args.limit > 0 else ds.queries
    print(f"[runner] running on {len(queries)} queries...")
    t0 = time.perf_counter()
    results = run_method(method, queries, method_name=args.method, is_questkg=is_questkg)
    wallclock = time.perf_counter() - t0
    print(f"[runner] done in {wallclock:.1f}s")

    df = to_dataframe(results)
    df.to_csv(csv_path, index=False)
    summary = aggregate(results)
    summary.update({
        "method": args.method,
        "dataset": args.dataset,
        "llm": args.llm,
        "encoder": args.encoder,
        "seed": args.seed,
        "wallclock_s": wallclock,
        "csv_path": str(csv_path),
    })
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[runner] wrote {csv_path}")
    print(f"[runner] wrote {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
