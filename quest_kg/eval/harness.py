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
        results.append(QueryResult(
            qid=str(qid), question=question, gold=str(gold), prediction=prediction,
            em=em, f1=f1, confidence=float(confidence),
            latency_ms=float(latency_ms), abstained=bool(abstained),
            extra={"method": method_name, **({"error": err} if err else {})},
        ))
    return results


def aggregate(results: list[QueryResult]) -> dict:
    if not results:
        return {}
    n = len(results)
    n_eval = sum(1 for r in results if not r.abstained)
    em = sum(r.em for r in results) / n
    em_eval = (sum(r.em for r in results if not r.abstained) / n_eval) if n_eval else 0.0
    f1 = sum(r.f1 for r in results) / n
    mean_latency = float(np.mean([r.latency_ms for r in results]))
    p95_latency = float(np.percentile([r.latency_ms for r in results], 95))
    return {
        "n": n,
        "n_attempted": n_eval,
        "abstention_rate": (n - n_eval) / n,
        "exact_match": em,
        "exact_match_when_attempted": em_eval,
        "token_f1": f1,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
    }


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
