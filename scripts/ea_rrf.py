"""Reciprocal Rank Fusion (RRF) for EA — alternative to weighted-sum hybrid.

Standard RRF formula:
  score(e) = sum_i  1 / (k + rank_i(e))

where rank_i is the rank of e under retriever i, and k is a smoothing
constant (k=60 is the convention). RRF is scale-invariant — no alpha
tuning, robust to mixing very different signals (text cosine vs sparse
structural counts).

For QUEST-KG-EA, the two retrievers are:
  - text cosine (sig(e1) -> sig(k2_cand))
  - structural (aligned-bridge propagation from e1 -> K2 partners)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ea_runner import build_signatures, load_dwy100k, load_openea, propagate_labels
from scripts.ea_merged_kg import build_neighbors, build_structural_indexes, structural_score_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--format", choices=("dwy100k", "openea"), required=True)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tag", default="local-1080ti__rrf__seed0")
    ap.add_argument("--n_sample", type=int, default=None)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--propagate", action="store_true")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    root = Path(args.root)
    data = load_dwy100k(root) if args.format == "dwy100k" else load_openea(root)
    t_total = time.perf_counter()
    print(f"[rrf] {args.dataset}: K1={len(data['ent1'])} K2={len(data['ent2'])} "
          f"train_pairs={len(data['train_pairs'])} test_pairs={len(data['test_pairs'])}")

    if args.propagate:
        import re
        opaque = re.compile(r"^(Q|P)\d+$")
        n_before = sum(1 for v in data["ent2"].values() if opaque.match(v))
        data["ent2"] = propagate_labels(
            data["ent2"], data["ent1"], data["train_pairs"], data["triples2"])
        n_after = sum(1 for v in data["ent2"].values() if opaque.match(v))
        print(f"  propagate: {n_before} -> {n_after} opaque K2 labels")

    sig1 = build_signatures(data["ent1"], data["rel1"], data["triples1"])
    sig2 = build_signatures(data["ent2"], data["rel2"], data["triples2"])
    k1_nb = build_neighbors(data["triples1"])
    k2_nb = build_neighbors(data["triples2"])
    seed_align, inv = build_structural_indexes(data["train_pairs"], k2_nb)

    test_pairs = data["test_pairs"]
    if args.n_sample and args.n_sample < len(test_pairs):
        idx = np.random.default_rng(0).choice(len(test_pairs), args.n_sample, replace=False)
        eval_pairs = [test_pairs[i] for i in idx]
    else:
        eval_pairs = test_pairs

    k2_ids_sorted = sorted(sig2.keys())
    k2_row_of = {eid: i for i, eid in enumerate(k2_ids_sorted)}

    from quest_kg.data.encoders import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder(args.encoder, batch_size=256)
    print(f"[rrf] encoder on {enc.device}")
    t = time.perf_counter()
    anchor_emb = enc([sig1[e1] for e1, _ in eval_pairs])
    k2_emb = enc([sig2[eid] for eid in k2_ids_sorted])
    print(f"  encoded in {time.perf_counter()-t:.1f}s")

    a_t = torch.as_tensor(anchor_emb, dtype=torch.float32, device="cuda")
    k_t = torch.as_tensor(k2_emb, dtype=torch.float32, device="cuda")
    a_t = torch.nn.functional.normalize(a_t, dim=1)
    k_t = torch.nn.functional.normalize(k_t, dim=1)

    try:
        from tqdm import tqdm
        pbar = tqdm(enumerate(eval_pairs), total=len(eval_pairs),
                     desc=f"rrf {args.dataset}", unit="q", dynamic_ncols=True)
    except ImportError:
        pbar = enumerate(eval_pairs)

    ranks = []; em = []; qids = []; pred_e2 = []; gold_e2 = []; conf = []
    correct = 0
    K_RRF = args.rrf_k
    nK = len(k2_ids_sorted)

    t = time.perf_counter()
    for i, (e1, e2_gold) in pbar:
        # Cosine ranks (compute top-200 to limit cost)
        cos_full = a_t[i] @ k_t.T  # (|K2|,)
        # Get cosine ranks: rank of each K2 idx in DESCENDING cosine order
        # We only need ranks for entities likely to be top — get top-200 and rest = nK
        top_cos_vals, top_cos_idx = torch.topk(cos_full, min(200, nK))
        cos_rank = {int(idx): r + 1 for r, idx in enumerate(top_cos_idx.tolist())}
        # Structural ranks
        counter = structural_score_one(e1, k1_nb, seed_align, inv, hops=1)
        struct_ranked = sorted(counter.items(), key=lambda x: -x[1])  # list of (k2_id, score)
        struct_rank = {k2_row_of.get(k2_id): r + 1
                       for r, (k2_id, _) in enumerate(struct_ranked)
                       if k2_row_of.get(k2_id) is not None}
        # Combine ranks via RRF
        all_candidates = set(cos_rank.keys()) | set(struct_rank.keys())
        rrf_scores = {}
        for c in all_candidates:
            r_cos = cos_rank.get(c, 1000)  # missing = far rank
            r_str = struct_rank.get(c, 1000)
            rrf_scores[c] = 1.0 / (K_RRF + r_cos) + 1.0 / (K_RRF + r_str)
        # Rank gold
        true_row = k2_row_of[e2_gold]
        if true_row in rrf_scores:
            sorted_cands = sorted(rrf_scores.items(), key=lambda x: -x[1])
            rank = next(r + 1 for r, (c, _) in enumerate(sorted_cands) if c == true_row)
        else:
            # gold not in either top-200 — use cosine rank
            n_above = int((cos_full > cos_full[true_row]).sum().item())
            rank = n_above + 1
        top_idx, top_score = max(rrf_scores.items(), key=lambda x: x[1]) if rrf_scores else (-1, 0)
        ranks.append(rank); em.append(1 if rank == 1 else 0)
        correct += int(rank == 1); qids.append(i)
        pred_e2.append(k2_ids_sorted[top_idx] if top_idx >= 0 else -1)
        gold_e2.append(e2_gold); conf.append(float(top_score))
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(h1=f"{correct/max(len(em),1):.3f}")
    inference_s = time.perf_counter() - t
    print(f"  scored {len(eval_pairs)} pairs in {inference_s:.1f}s")

    ranks_a = np.asarray(ranks); em_a = np.asarray(em, dtype=float)
    conf_a = np.asarray(conf, dtype=float)
    hits1 = float(em_a.mean()); hits3 = float((ranks_a <= 3).mean())
    hits10 = float((ranks_a <= 10).mean())
    mrr = float((1.0 / np.maximum(ranks_a, 1)).mean())
    from quest_kg.eval.metrics import ece, aurc
    # Normalize confidence to [0,1] (RRF scores are small)
    if conf_a.max() > 0:
        conf_norm = conf_a / conf_a.max()
    else:
        conf_norm = conf_a
    e_ece = float(ece(np.clip(conf_norm, 0, 1), em_a, n_bins=15))
    e_aurc = float(aurc(np.clip(conf_norm, 0, 1), em_a))

    df = pd.DataFrame({"qid": qids, "gold_e2": gold_e2, "pred_e2": pred_e2,
                        "em": em, "x_rank": ranks, "confidence": conf_norm})
    wallclock = time.perf_counter() - t_total
    out_json = {
        "n": len(eval_pairs), "n_attempted": len(eval_pairs),
        "abstention_rate": 0.0, "exact_match": hits1,
        "exact_match_when_attempted": hits1, "token_f1": None,
        "mean_latency_ms": (inference_s * 1000.0) / max(len(eval_pairs), 1),
        "p95_latency_ms": None, "task_type": "entity_alignment",
        "primary_metric": "hits_at_1", "primary_value": hits1,
        "hits_at_1": hits1, "hits_at_3": hits3, "hits_at_10": hits10,
        "mrr": mrr, "ECE": e_ece, "AURC": e_aurc, "n_bins_ece": 15,
        "method": "quest_kg_ea_rrf", "dataset": args.dataset,
        "encoder": args.encoder, "seed": 0,
        "wallclock_s": round(wallclock, 1), "limit": len(eval_pairs),
        "tag": args.tag, "rrf_k": K_RRF, "scoring": "reciprocal rank fusion (cosine + structural)",
    }

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_p = out_dir / f"quest_kg_ea_rrf__{args.dataset}__{args.tag}.json"
    csv_p  = out_dir / f"quest_kg_ea_rrf__{args.dataset}__{args.tag}.csv"
    json_p.write_text(json.dumps(out_json, indent=2))
    df.to_csv(csv_p, index=False)
    print(f"\n=== RESULTS ===")
    for k in ("hits_at_1", "hits_at_3", "hits_at_10", "mrr"):
        print(f"  {k}: {out_json[k]:.4f}")
    print(f"Wrote {json_p.name}")


if __name__ == "__main__":
    main()
