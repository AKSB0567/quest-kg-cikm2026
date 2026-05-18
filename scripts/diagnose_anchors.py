"""Diagnose anchor- and label-format mismatches between queries and KG.

For each low-EM dataset, dump 5 sample queries showing:
  - question
  - q_entity / metadata anchors as provided by the loader
  - triples touching each anchor (first few)
  - QUEST-KG prediction surface form
  - gold answer surface form

This is what we use to decide whether to (a) plumb a MID->label dict through
inference, or (b) score candidate tails by surface label similarity.

Usage:
    .venv312/Scripts/python scripts/diagnose_anchors.py [webqsp|cwq|icews18|orgaccess]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def diagnose_webqsp(n_sample: int = 5):
    from quest_kg.data.loaders import load_webqsp
    ds = load_webqsp(ROOT / "data")
    print(f"\n=== WebQSP ===")
    print(f"  triples: {len(ds.triples)}, queries: {len(ds.queries)}")

    # Sample identifier styles in the KG
    entity_samples = Counter()
    for t in ds.triples[:5000]:
        for x in (t.s, t.o):
            xs = str(x)
            if xs.startswith("m.") or xs.startswith("g."):
                entity_samples["mid"] += 1
            elif " " in xs:
                entity_samples["surface_with_space"] += 1
            else:
                entity_samples["other"] += 1
    print(f"  KG identifier style (first 5k triples, both sides):")
    for k, v in entity_samples.most_common():
        print(f"    {k:>22s}: {v}")

    # Sample queries
    print(f"\n  Sample queries:")
    for q in ds.queries[:n_sample]:
        print(f"\n  qid={q['qid']}")
        print(f"    question:  {q['question']}")
        print(f"    q_entity:  {q['q_entity']}")
        print(f"    gold:      {q['answer']!r}")
        print(f"    all_gold:  {q['all_answers']}")
        # Find any KG triples touching the q_entity
        anchors = set(q['q_entity'])
        touching = [t for t in ds.triples if t.s in anchors or t.o in anchors][:3]
        if touching:
            print(f"    Touching triples (anchor present in KG):")
            for t in touching:
                print(f"      ({t.s} | {t.r} | {t.o})")
        else:
            # try treating q_entity as a surface label
            anchor_lower = {a.lower() for a in anchors}
            touching = [t for t in ds.triples
                        if str(t.s).lower() in anchor_lower
                        or str(t.o).lower() in anchor_lower][:3]
            if touching:
                print(f"    [case-insensitive match] Touching triples:")
                for t in touching:
                    print(f"      ({t.s} | {t.r} | {t.o})")
            else:
                print(f"    NO KG triples touch q_entity (anchor mismatch likely)")

    # Look at the existing QUEST-KG predictions
    csv = ROOT / "results" / "quest_kg__webqsp__local-1080ti__Qwen2.5-1.5B-Instruct__seed0.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv).head(n_sample)
        print(f"\n  QUEST-KG predictions (first {n_sample}):")
        for _, row in df.iterrows():
            print(f"    qid={row['qid']:<20s} pred={row['prediction']!r:<30s} gold={row['gold']!r}")


def diagnose_icews18(n_sample: int = 5):
    from quest_kg.data.loaders import load_icews18
    ds = load_icews18(ROOT / "data")
    print(f"\n=== ICEWS18 ===")
    print(f"  triples: {len(ds.triples)}, queries: {len(ds.queries)}")
    print(f"  task: {ds.task}")
    print(f"  metadata keys: {list(ds.metadata.keys())}")
    # Sample queries
    print(f"\n  Sample queries:")
    for q in ds.queries[:n_sample]:
        print(f"\n  qid={q['qid']}")
        print(f"    question:  {q['question']}")
        print(f"    gold:      {q['answer']}")
        print(f"    keys:      {list(q.keys())}")
        head = q.get("head") or q.get("subject")
        rel = q.get("rel") or q.get("relation")
        ts = q.get("timestamp") or q.get("ts")
        # Find triples with matching head+rel
        if head is not None and rel is not None:
            matches = [t for t in ds.triples
                       if str(t.s) == str(head) and str(t.r) == str(rel)][:5]
            print(f"    triples with head={head}, rel={rel}:")
            for t in matches:
                print(f"      ({t.s} | {t.r} | {t.o})")


def diagnose_orgaccess(n_sample: int = 5):
    from quest_kg.data.loaders import load_orgaccess
    ds = load_orgaccess(ROOT / "data")
    print(f"\n=== OrgAccess ===")
    print(f"  triples: {len(ds.triples)}, queries: {len(ds.queries)}")
    labels = Counter(q["answer"] for q in ds.queries)
    print(f"  label distribution: {dict(labels)}")
    pos = sum(1 for q in ds.queries if str(q["answer"]) == "1")
    print(f"  positive rate: {pos}/{len(ds.queries)} = {pos/max(len(ds.queries),1):.1%}")
    # First few positives + negatives
    positives = [q for q in ds.queries if str(q["answer"]) == "1"][:3]
    negatives = [q for q in ds.queries if str(q["answer"]) == "0"][:3]
    for label, group in (("POS", positives), ("NEG", negatives)):
        for q in group:
            print(f"  [{label}] qid={q['qid']} q='{q['question'][:80]}' user={q.get('user')} resource={q.get('resource')}")


def main():
    targets = sys.argv[1:] or ["webqsp", "icews18", "orgaccess"]
    for t in targets:
        try:
            if t == "webqsp":
                diagnose_webqsp()
            elif t == "icews18":
                diagnose_icews18()
            elif t == "orgaccess":
                diagnose_orgaccess()
            else:
                print(f"unknown target: {t}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[diagnose] {t} failed: {e}")


if __name__ == "__main__":
    main()
