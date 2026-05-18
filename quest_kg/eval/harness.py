"""Shared evaluation harness used by both QUEST-KG and baselines.

Computes per-query metrics:
  - Exact Match (EM)
  - Token-level F1 (for QA tasks)
  - Hits@1 (any-correct)
  - Confidence (from method)
  - Latency (ms)
  - Abstention flag
And aggregates them across a dataset.

For ECE/Brier/NLL/AURC see `quest_kg.eval.metrics` (these run after the full
result CSV is produced).
"""
from __future__ import annotations

import re
import string
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np


# Map yes/no surface forms to canonical "1"/"0" so QA harness + access-control task align.
_YES_TOKENS = {"yes", "y", "true", "t", "1", "allow", "allowed", "permit", "permitted", "grant", "granted"}
_NO_TOKENS  = {"no", "n", "false", "f", "0", "deny", "denied", "refuse", "refused", "reject", "rejected"}


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, normalize yes/no."""
    s = s.lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Canonicalize yes/no surface forms to "1"/"0" so 'no' matches '0' under EM.
    head = s.split()[0] if s else s
    if head in _YES_TOKENS:
        return "1"
    if head in _NO_TOKENS:
        return "0"
    return s


def exact_match(pred: str, gold: str | list[str]) -> int:
    if isinstance(gold, list):
        return int(any(_normalize(pred) == _normalize(g) for g in gold))
    return int(_normalize(pred) == _normalize(gold))


def token_f1(pred: str, gold: str | list[str]) -> float:
    if isinstance(gold, list):
        return max((token_f1(pred, g) for g in gold), default=0.0)
    p_tok = _normalize(pred).split()
    g_tok = _normalize(gold).split()
    if not p_tok or not g_tok:
        return 0.0
    common = set(p_tok) & set(g_tok)
    if not common:
        return 0.0
    p_count = sum(1 for t in p_tok if t in common)
    g_count = sum(1 for t in g_tok if t in common)
    prec = p_count / len(p_tok)
    rec = g_count / len(g_tok)
    return 2 * prec * rec / (prec + rec)


@dataclass
class QueryResult:
    qid: str
    question: str
    gold: str
    prediction: str | None
    em: int
    f1: float
    confidence: float
    latency_ms: float
    abstained: bool
    extra: dict


def run_method(
    method,                    # has .predict(query, ...) -> obj with .prediction/.confidence/.abstained
    queries: Iterable[dict],
    method_name: str,
    is_questkg: bool = False,
    extract_expected_types: bool = False,
) -> list[QueryResult]:
    """Loop a method over all queries; capture metrics + timing."""
    results: list[QueryResult] = []
    for q in queries:
        qid = q.get("qid", str(len(results)))
        question = q.get("question", "")
        gold = q.get("answer", "")
        all_gold = q.get("all_answers", [gold])
        t0 = time.perf_counter()
        try:
            if is_questkg:
                pred = method.predict(
                    question,
                    expected_types=q.get("expected_types") or None,
                    query_meta=q,
                )
                prediction = pred.prediction
                confidence = pred.confidence
                abstained = pred.abstained
            else:
                pred = method.predict(question)
                prediction = pred.prediction
                confidence = pred.confidence
                abstained = pred.abstained
        except Exception as e:
            prediction, confidence, abstained = None, 0.0, True
            pred = None
            err = str(e)[:200]
        else:
            err = None
        latency_ms = (time.perf_counter() - t0) * 1000.0

        em = exact_match(prediction or "", all_gold) if prediction else 0
        f1 = token_f1(prediction or "", all_gold) if prediction else 0.0

        # Plumb candidate ranking -> per-query rank (for Hits@k / MRR on
        # link-prediction tasks). 1-indexed; rank=0 means gold absent.
        rank = 0
        ranking = (getattr(pred, "extra", None) or {}).get("candidate_ranking") if pred else None
        if ranking and gold:
            gold_n = _normalize(str(gold))
            for i, cand in enumerate(ranking):
                if _normalize(str(cand)) == gold_n:
                    rank = i + 1
                    break

        results.append(QueryResult(
            qid=str(qid), question=question, gold=str(gold), prediction=prediction,
            em=em, f1=f1, confidence=float(confidence),
            latency_ms=float(latency_ms), abstained=bool(abstained),
            extra={"method": method_name, "rank": rank,
                   **({"error": err} if err else {})},
        ))
    return results


def aggregate(results: list[QueryResult], task_type: str | None = None) -> dict:
    """Aggregate per-query results into a summary.

    `task_type` selects task-appropriate headline metrics:
      - "access_control" (binary yes/no): adds balanced_accuracy + positive-class P/R/F1 + macro_f1.
      - "link_prediction" (ranked entity): adds Hits@1, Hits@10, MRR from per-query
         `extra["rank"]` (1-indexed; 0 means gold not in candidates).
      - "qa" (open-domain): EM + token_f1 remain primary.

    EM/F1 are always reported so we can audit the impact of switching headlines.
    """
    if not results:
        return {}
    from quest_kg.eval.metrics import (
        balanced_accuracy, hits_at_k_from_ranks, macro_f1_binary,
        mrr_from_ranks, positive_class_prf,
    )

    n = len(results)
    n_eval = sum(1 for r in results if not r.abstained)
    em = sum(r.em for r in results) / n
    em_eval = (sum(r.em for r in results if not r.abstained) / n_eval) if n_eval else 0.0
    f1 = sum(r.f1 for r in results) / n
    mean_latency = float(np.mean([r.latency_ms for r in results]))
    p95_latency = float(np.percentile([r.latency_ms for r in results], 95))
    summary = {
        "n": n,
        "n_attempted": n_eval,
        "abstention_rate": (n - n_eval) / n,
        "exact_match": em,
        "exact_match_when_attempted": em_eval,
        "token_f1": f1,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "task_type": task_type or "qa",
    }

    if task_type == "access_control":
        preds = [_normalize(r.prediction or "") for r in results]
        golds = [_normalize(r.gold or "") for r in results]
        prf = positive_class_prf(preds, golds, positive="1")
        summary["balanced_accuracy"] = balanced_accuracy(preds, golds, positive="1")
        summary["macro_f1"] = macro_f1_binary(preds, golds, positive="1")
        summary["pos_precision"] = prf["precision"]
        summary["pos_recall"] = prf["recall"]
        summary["pos_f1"] = prf["f1"]
        summary["pos_support"] = prf["support_pos"]
        summary["neg_support"] = prf["support_neg"]
        summary["primary_metric"] = "balanced_accuracy"
        summary["primary_value"] = summary["balanced_accuracy"]

    elif task_type == "link_prediction":
        ranks = [int((r.extra or {}).get("rank", 0)) for r in results]
        summary["hits_at_1"] = hits_at_k_from_ranks(ranks, 1)
        summary["hits_at_3"] = hits_at_k_from_ranks(ranks, 3)
        summary["hits_at_10"] = hits_at_k_from_ranks(ranks, 10)
        summary["mrr"] = mrr_from_ranks(ranks)
        summary["n_in_candidates"] = int(sum(1 for r in ranks if r > 0))
        summary["primary_metric"] = "mrr"
        summary["primary_value"] = summary["mrr"]

    else:  # qa
        # Accuracy (Hits@1 ~= exact_match) is the headline metric for WebQSP/CWQ
        # in the paper draft, not token_f1. token_f1 stays in the output as a
        # secondary diagnostic but is not the primary comparison.
        summary["primary_metric"] = "accuracy"
        summary["primary_value"] = em
        summary["hits_at_1"] = em

    return summary


def to_dataframe(results: list[QueryResult]):
    """Optional pandas DataFrame export."""
    import pandas as pd
    rows = []
    for r in results:
        d = asdict(r)
        # Flatten extra
        d.update({f"x_{k}": v for k, v in (d.pop("extra") or {}).items()})
        rows.append(d)
    return pd.DataFrame(rows)
