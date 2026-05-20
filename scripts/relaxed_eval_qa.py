"""Recompute WebQSP/CWQ Hits@1 with relaxed surface-form matching.

Current eval (quest_kg/eval/harness.py) lowercases + strips punctuation but
then does EXACT equality. Many literature evals are more permissive:
  - Strip parentheticals ("Tom Cruise (actor)" -> "Tom Cruise")
  - Strip descriptor suffixes ("- Season 26", " (film)")
  - Allow substring containment (prediction in gold, or gold in prediction)
  - Treat first surface form as the canonical (vs requiring all)

We re-score existing per-query CSVs without re-running inference. Reports:
  strict EM     (current)
  + parentheticals stripped
  + substring containment
  + both
For each dataset, we save the relaxed numbers.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import string


PAREN_RE = re.compile(r"\s*\([^)]*\)\s*")
DESCRIPTOR_RE = re.compile(r"\s*-\s*(Season|Episode|Series|Vol\.?|Volume|Part)\s+\S+\s*", re.IGNORECASE)


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_paren(s: str) -> str:
    s = PAREN_RE.sub(" ", s)
    s = DESCRIPTOR_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_strip(s: str) -> str:
    return _normalize(_strip_paren(s))


def relaxed_match(pred: str, gold_list: list[str], mode: str) -> int:
    """mode: 'strict' | 'paren' | 'substring' | 'both'"""
    pred_n = _norm_strip(pred) if mode in ("paren", "both") else _normalize(pred)
    for g in gold_list:
        g_n = _norm_strip(g) if mode in ("paren", "both") else _normalize(g)
        if not pred_n or not g_n:
            continue
        if pred_n == g_n:
            return 1
        if mode in ("substring", "both"):
            # require minimum length to avoid trivial matches
            if len(g_n) >= 3 and g_n in pred_n:
                return 1
            if len(pred_n) >= 3 and pred_n in g_n:
                return 1
    return 0


def load_gold_lists(ds_name: str) -> dict[str, list[str]]:
    """Return qid -> list of gold answer strings from the rmanluo HF dataset."""
    from quest_kg.data.loaders import load_dataset
    ds = load_dataset(ds_name, str(ROOT / "data"))
    out = {}
    for q in ds.queries:
        qid = str(q.get("qid") or q.get("id") or "")
        if not qid:
            continue
        gold = q.get("answer", [])
        if isinstance(gold, str):
            gold = [gold]
        all_g = q.get("all_answers", gold)
        if isinstance(all_g, str):
            all_g = [all_g]
        out[qid] = list(set(all_g + gold))
    return out


def score_csv(csv_path: Path, gold_lookup: dict[str, list[str]]):
    df = pd.read_csv(csv_path)
    rows = []
    modes = ["strict", "paren", "substring", "both"]
    counts = {m: 0 for m in modes}
    for _, r in df.iterrows():
        qid = str(r["qid"])
        pred = str(r.get("prediction", ""))
        # Try CSV gold column first; fall back to lookup
        gold_csv = r.get("gold", "")
        gold_list = gold_lookup.get(qid, [str(gold_csv)] if gold_csv else [])
        if not gold_list:
            continue
        for m in modes:
            counts[m] += relaxed_match(pred, gold_list, mode=m)
    n = len(df)
    return {m: counts[m] / max(n, 1) for m in modes}, n


def main():
    cells = [
        # Ours
        ("webqsp", "results/quest_kg_llm__webqsp__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("cwq",    "results/quest_kg_llm__cwq__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("webqsp", "results/quest_kg__webqsp__local-1080ti__symbolic__seed0.csv"),
        ("cwq",    "results/quest_kg__cwq__local-1080ti__symbolic__seed0.csv"),
        # LLM baselines — must use SAME eval for fair comparison
        ("webqsp", "results/vanilla_rag__webqsp__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("webqsp", "results/graphrag__webqsp__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("webqsp", "results/tog1__webqsp__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("webqsp", "results/tog2__webqsp__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("webqsp", "results/cok__webqsp__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("cwq",    "results/vanilla_rag__cwq__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("cwq",    "results/graphrag__cwq__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("cwq",    "results/tog1__cwq__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("cwq",    "results/tog2__cwq__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("cwq",    "results/cok__cwq__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
    ]
    print(f"{'config':<70s} {'strict':>8s} {'paren':>8s} {'subst':>8s} {'both':>8s} {'+both':>8s}")
    print("-" * 120)
    for ds_name, csv_rel in cells:
        csv_path = ROOT / csv_rel
        if not csv_path.exists():
            print(f"{csv_rel:<70s}  (missing)")
            continue
        print(f"  loading gold lists for {ds_name}...")
        gold_lookup = load_gold_lists(ds_name)
        scores, n = score_csv(csv_path, gold_lookup)
        gain = scores["both"] - scores["strict"]
        print(f"{csv_rel:<70s} "
              f"{scores['strict']:>8.3f} {scores['paren']:>8.3f} "
              f"{scores['substring']:>8.3f} {scores['both']:>8.3f} "
              f"{gain:>+8.3f}")
        # Save relaxed JSON sidecar
        out_json = {
            "csv": str(csv_path.relative_to(ROOT)),
            "n": n,
            "hits_at_1_strict": scores["strict"],
            "hits_at_1_paren_stripped": scores["paren"],
            "hits_at_1_substring": scores["substring"],
            "hits_at_1_relaxed_both": scores["both"],
            "gain_over_strict": gain,
        }
        out_path = csv_path.with_suffix(".relaxed.json")
        out_path.write_text(json.dumps(out_json, indent=2))
        print(f"    wrote {out_path.name}")


if __name__ == "__main__":
    main()
