"""Dataset loaders.

Each loader reads from `<root>/raw/<dataset_name>/` (populated by the
`scripts/download/*` scripts) and returns a unified `Dataset` object.

For Phase 2 we cover 4 datasets:
  - WebQSP  (multi-hop QA over Freebase, from rmanluo/RoG-webqsp)
  - CWQ     (compositional multi-hop QA, from rmanluo/RoG-cwq)
  - ICEWS18 (temporal link prediction, RE-Net mirror)
  - OrgAccess (synthetic dynamic policy, from our generator)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quest_kg.core.types import Provenance, Triple
from quest_kg.data.dataset import Dataset


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# WebQSP / CWQ via HuggingFace datasets (rmanluo mirrors)
# ---------------------------------------------------------------------------

def _load_rmanluo_hf(root: Path, dataset_name: str) -> Dataset:
    """Load a RoG-format HF dataset saved via `datasets.save_to_disk`.

    Expected structure on disk:
        root/raw/<dataset_name>/dataset/{train|validation|test}/...

    Each example has fields: id, question, q_entity, a_entity, answer,
    graph (list of [h, r, t] triples). We materialize a global KG by unioning
    every example's `graph` field.
    """
    from datasets import load_from_disk

    base = root / "raw" / dataset_name / "dataset"
    if not base.exists():
        raise FileNotFoundError(f"{base} not present — run notebooks/setup_all.ipynb first")
    ds = load_from_disk(str(base))

    # Pick a test split for evaluation
    split = None
    for cand in ("test", "validation", "train"):
        if cand in ds:
            split = ds[cand]
            break
    assert split is not None, "no usable split found"

    # Materialize KG: dedupe triples across all examples (test only for now;
    # add train/validation in Phase 3 if we need a larger KG)
    triples_seen: dict[tuple[str, str, str], int] = {}
    triples: list[Triple] = []
    queries: list[dict] = []
    for ex in split:
        gid = ex.get("id") or ex.get("qid") or ex.get("question_id")
        question = ex.get("question") or ex.get("query") or ""
        answer = ex.get("answer") or ex.get("answers") or ex.get("a_entity") or []
        if isinstance(answer, list):
            answer_str = answer[0] if answer else ""
            all_answers = answer
        else:
            answer_str = str(answer)
            all_answers = [answer_str]
        q_entity = ex.get("q_entity") or ex.get("topic_entity") or []
        graph_raw = ex.get("graph") or []
        per_query_idx: list[int] = []
        for tr in graph_raw:
            if len(tr) >= 3:
                h, r, t = tr[0], tr[1], tr[2]
                key = (h, r, t)
                if key not in triples_seen:
                    triples_seen[key] = len(triples)
                    triples.append(Triple(s=h, r=r, o=t))
                per_query_idx.append(triples_seen[key])
        queries.append({
            "qid": str(gid),
            "question": str(question),
            "answer": str(answer_str),
            "all_answers": [str(a) for a in all_answers],
            "q_entity": list(q_entity) if isinstance(q_entity, (list, tuple)) else [str(q_entity)],
            "expected_types": set(),  # filled per-example via Freebase types if available
            # Indices into the global `triples` list that belong to THIS query's
            # local graph (from HF dataset's `graph` field). Used to restrict
            # retrieval to the query's relevant subgraph -- matches graphrag's
            # per-query scope and avoids the kg_subset truncation problem.
            "graph_triple_idx": per_query_idx,
        })

    return Dataset(
        name=dataset_name,
        task="qa",
        triples=triples,
        queries=queries,
        metadata={"split": "test", "hf_source": f"rmanluo/RoG-{dataset_name}"},
    )


def load_webqsp(root: Path) -> Dataset:
    return _load_rmanluo_hf(root, "webqsp")


def load_cwq(root: Path) -> Dataset:
    return _load_rmanluo_hf(root, "cwq")


# ---------------------------------------------------------------------------
# ICEWS18
# ---------------------------------------------------------------------------

def load_icews18(root: Path) -> Dataset:
    """ICEWS18 in RE-Net format. Each line: head_idx\\trel_idx\\ttail_idx\\ttimestamp."""
    base = root / "raw" / "icews18"
    if not (base / ".done").exists():
        raise FileNotFoundError(f"{base} not ready")

    def parse(p: Path) -> list[tuple[int, int, int, int]]:
        out = []
        with open(p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    out.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
        return out

    train = parse(base / "train.txt")
    valid = parse(base / "valid.txt")
    test = parse(base / "test.txt")

    # KG = train + valid as known facts; test as queries (predict tail given head, rel, ts)
    triples: list[Triple] = [
        Triple(s=str(h), r=str(r), o=str(t), timestamp=ts)
        for (h, r, t, ts) in train + valid
    ]
    queries: list[dict] = [
        {
            "qid": f"icews18_{i}",
            "question": f"(head={h}, rel={r}, ?, ts={ts})",
            "head": str(h),
            "rel": str(r),
            "answer": str(t),
            "all_answers": [str(t)],
            "timestamp": ts,
            "expected_types": set(),
        }
        for i, (h, r, t, ts) in enumerate(test)
    ]
    return Dataset(
        name="icews18",
        task="temporal_link_pred",
        triples=triples,
        queries=queries,
        metadata={"n_train": len(train), "n_valid": len(valid), "n_test": len(test)},
    )


# ---------------------------------------------------------------------------
# OrgAccess (synthetic, from our generator)
# ---------------------------------------------------------------------------

def load_orgaccess(root: Path) -> Dataset:
    base = root / "raw" / "orgaccess"
    if not (base / ".done").exists():
        raise FileNotFoundError(f"{base} not ready")

    triples_data = json.loads((base / "triples.json").read_text())
    context_schedule = json.loads((base / "context_schedule.json").read_text())
    test = json.loads((base / "test.json").read_text())

    triples: list[Triple] = []
    for t in triples_data:
        triples.append(Triple(
            s=t["s"], r=t["r"], o=t["o"], timestamp=_safe_int(t.get("t")),
            prov=Provenance(extraction_conf=1.0, source_trust=1.0, edit_age_days=0.0),
        ))

    raw_queries: list[dict] = [
        {
            "qid": f"oa_{q['qid']}",
            "question": q["question"],
            "answer": str(q["label"]),  # "0" or "1"
            "all_answers": [str(q["label"])],
            "user": q["user"],
            "resource": q["resource"],
            "timestamp": q["timestamp"],
            "context": q["context"],
            "expected_types": set(),
        }
        for q in test
    ]

    # Deterministic class-balanced interleave so `--limit N` evaluates on roughly
    # N/2 positives + N/2 negatives instead of the 9% positive rate the synthetic
    # generator produces. Original ordering is preserved within each class.
    positives = [q for q in raw_queries if str(q["answer"]) == "1"]
    negatives = [q for q in raw_queries if str(q["answer"]) == "0"]
    queries: list[dict] = []
    for pos, neg in zip(positives, negatives):
        queries.append(pos)
        queries.append(neg)
    if len(positives) > len(negatives):
        queries.extend(positives[len(negatives):])
    elif len(negatives) > len(positives):
        queries.extend(negatives[len(positives):])

    return Dataset(
        name="orgaccess",
        task="access_control",
        triples=triples,
        queries=queries,
        metadata={
            "context_schedule": context_schedule,
            "n_test": len(test),
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "interleave_balanced": True,
        },
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOADERS = {
    "webqsp":    load_webqsp,
    "cwq":       load_cwq,
    "icews18":   load_icews18,
    "orgaccess": load_orgaccess,
}


def list_datasets() -> list[str]:
    return list(_LOADERS.keys())


def load_dataset(name: str, root: str | Path) -> Dataset:
    if name not in _LOADERS:
        raise KeyError(f"unknown dataset {name!r}; available: {list_datasets()}")
    return _LOADERS[name](Path(root))
