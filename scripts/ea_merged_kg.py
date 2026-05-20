"""Merged-KG EA — unified QUEST-KG methodology for entity alignment.

QUEST-KG operates on a unified knowledge graph. For QA the KG is the per-query
Freebase subgraph; for EA it is the merged graph:

    G_merged = (V_K1 cup V_K2,
                E_K1 cup E_K2 cup {(e1, ALIGNED_TO, e2) for (e1,e2) in train_pairs})

For each test anchor e1, multi-hop traversal flows: e1 -> K1-neighbors of e1
-> seed-aligned K2 partners -> K2-neighbors of those partners. A K2 candidate's
"evidential belief" is the count of seed-aligned K1 neighbors of e1 that hit
its K2 neighborhood. Combined with a residual cosine similarity (still useful
when text labels exist), this gives a hybrid score:

    score(k2_cand | e1) = alpha * cosine(sig(e1), sig(k2_cand))
                          + (1 - alpha) * structural(e1, k2_cand)

where structural is normalized by the number of available aligned bridges.

This is the same methodology used for QA (retrieve -> propagate -> symbolic
check), specialized to EA. Crucially: it does NOT depend on K2 having text
labels -- the structural channel works purely from triples, which solves the
Wikidata Q-number problem that ruined our text-only DBP-WD result (0.068).
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

from scripts.ea_runner import (
    build_signatures, load_dwy100k, load_openea, propagate_labels,
)


def build_neighbors(triples: list[tuple[int, int, int]]) -> dict[int, set[int]]:
    nb: dict[int, set[int]] = {}
    for h, _, t in triples:
        nb.setdefault(h, set()).add(t)
        nb.setdefault(t, set()).add(h)
    return nb


def build_structural_indexes(train_pairs, k2_nb):
    """Precompute index needed for per-anchor structural scoring.
    Returns (seed_align_dict, inv_dict) — both small enough to keep in RAM."""
    seed_align = {e1: e2 for e1, e2 in train_pairs}
    inv: dict[int, set[int]] = {}
    aligned_K2_partners = set(seed_align.values())
    for p in aligned_K2_partners:
        for nb in k2_nb.get(p, ()):
            inv.setdefault(p, set()).add(nb)
    return seed_align, inv


def k1_neighbors_k_hops(e1, k1_nb, hops: int) -> dict[int, int]:
    """BFS k1 neighbors up to `hops` hops. Returns {neighbor_id: min_hop_distance}."""
    if hops <= 0:
        return {e1: 0}
    dist = {e1: 0}
    frontier = {e1}
    for h in range(1, hops + 1):
        new_frontier = set()
        for n in frontier:
            for nb in k1_nb.get(n, ()):
                if nb not in dist:
                    dist[nb] = h
                    new_frontier.add(nb)
        frontier = new_frontier
        if not frontier:
            break
    return dist


def structural_score_one(e1: int,
                          k1_nb: dict[int, set[int]],
                          seed_align: dict[int, int],
                          inv: dict[int, set[int]],
                          hops: int = 1,
                          hop_decay: float = 0.5) -> Counter:
    """Sparse counter: k2_entity_id -> shared-aligned-neighbor weight for one e1.

    Now supports multi-hop K1 traversal. Each K1 neighbor at distance d contributes
    weight hop_decay^(d-1). Aligned partner counts are weighted accordingly.
    """
    counter: Counter = Counter()
    if hops <= 1:
        # Original 1-hop path (faster)
        e1_K1_neighbors = k1_nb.get(e1, set())
        partners = {seed_align[n] for n in e1_K1_neighbors if n in seed_align}
        if not partners:
            return counter
        for p in partners:
            for k2_cand in inv.get(p, ()):
                counter[k2_cand] += 1.0
        return counter

    # Multi-hop: BFS K1 neighbors, weight each contributing partner by hop_decay**dist
    dist = k1_neighbors_k_hops(e1, k1_nb, hops)
    # Strip self
    dist.pop(e1, None)
    if not dist:
        return counter
    for n, d in dist.items():
        if n in seed_align:
            p = seed_align[n]
            w = hop_decay ** (d - 1)
            for k2_cand in inv.get(p, ()):
                counter[k2_cand] += w
    return counter


def run_merged_kg(data: dict, dataset_name: str, encoder_name: str, tag: str,
                   n_sample: int | None = None, alpha: float = 0.5,
                   propagate: bool = False,
                   struct_hops: int = 1,
                   hop_decay: float = 0.5) -> tuple[dict, pd.DataFrame]:
    t_total = time.perf_counter()
    print(f"[merged-kg] {dataset_name}: K1={len(data['ent1'])} K2={len(data['ent2'])} "
          f"train_pairs={len(data['train_pairs'])} test_pairs={len(data['test_pairs'])}")

    if propagate:
        import re
        opaque = re.compile(r"^(Q|P)\d+$")
        n_before = sum(1 for v in data["ent2"].values() if opaque.match(v))
        data["ent2"] = propagate_labels(
            data["ent2"], data["ent1"], data["train_pairs"], data["triples2"])
        n_after = sum(1 for v in data["ent2"].values() if opaque.match(v))
        print(f"  label propagation: opaque K2 labels {n_before} -> {n_after}")

    # Build signatures (text channel)
    print("[merged-kg] building text signatures...")
    sig1 = build_signatures(data["ent1"], data["rel1"], data["triples1"])
    sig2 = build_signatures(data["ent2"], data["rel2"], data["triples2"])

    # Build neighbor sets (structural channel)
    print("[merged-kg] building K1/K2 neighbor sets...")
    k1_nb = build_neighbors(data["triples1"])
    k2_nb = build_neighbors(data["triples2"])

    # Test set
    test_pairs = data["test_pairs"]
    rng = np.random.default_rng(0)
    if n_sample and n_sample < len(test_pairs):
        idx = rng.choice(len(test_pairs), size=n_sample, replace=False)
        eval_pairs = [test_pairs[i] for i in idx]
    else:
        eval_pairs = test_pairs

    # K2 indexing
    k2_ids_sorted = sorted(sig2.keys())
    k2_row_of = {eid: i for i, eid in enumerate(k2_ids_sorted)}
    k2_sigs = [sig2[eid] for eid in k2_ids_sorted]

    # Text channel: encode anchors + K2 signatures
    from quest_kg.data.encoders import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder(encoder_name, batch_size=256)
    print(f"[merged-kg] encoder ready on {enc.device}")

    t = time.perf_counter()
    anchor_sigs = [sig1[e1] for e1, _ in eval_pairs]
    anchor_emb = enc(anchor_sigs)
    print(f"  encoded {len(anchor_sigs)} anchors in {time.perf_counter()-t:.1f}s")
    t = time.perf_counter()
    k2_emb = enc(k2_sigs)
    print(f"  encoded {len(k2_sigs)} K2 entities in {time.perf_counter()-t:.1f}s")

    a_t = torch.as_tensor(anchor_emb, dtype=torch.float32, device="cuda")
    k_t = torch.as_tensor(k2_emb, dtype=torch.float32, device="cuda")
    a_t = torch.nn.functional.normalize(a_t, dim=1)
    k_t = torch.nn.functional.normalize(k_t, dim=1)

    # Structural channel: build precomputed indexes (small)
    print("[merged-kg] building structural indexes (seed alignments + K2 inverted)...")
    seed_align, inv = build_structural_indexes(data["train_pairs"], k2_nb)
    print(f"  |seed_align|={len(seed_align)}  |inv|={len(inv)}")

    # Per-anchor: compute cosine vec on GPU (no big matmul stored), structural
    # sparse counter, combine into a single (|K2|,) tensor, then rank. NEVER
    # materialize (n_anchors x |K2|) which would OOM at full N=70K.
    t = time.perf_counter()
    try:
        from tqdm import tqdm
        pbar = tqdm(enumerate(eval_pairs), total=len(eval_pairs),
                     desc=f"merged-kg {dataset_name}", unit="q",
                     dynamic_ncols=True)
    except ImportError:
        pbar = enumerate(eval_pairs)

    ranks: list[int] = []
    em: list[int] = []
    qids: list[int] = []
    pred_e2: list[int] = []
    gold_e2: list[int] = []
    confidence: list[float] = []
    correct = 0
    nK = len(k2_ids_sorted)
    for i, (e1, e2_gold) in pbar:
        # Cosine (on GPU, no per-anchor copy to CPU until reduction)
        cos = a_t[i] @ k_t.T  # (|K2|,) on GPU
        cos01 = (cos + 1.0) / 2.0
        # Structural counter for this anchor
        counter = structural_score_one(e1, k1_nb, seed_align, inv,
                                         hops=struct_hops, hop_decay=hop_decay)
        # Compose hybrid score directly on GPU (sparse-add the counter)
        combined = alpha * cos01
        if counter:
            max_c = max(counter.values())
            for k2_cand, c in counter.items():
                row = k2_row_of.get(k2_cand)
                if row is not None:
                    combined[row] = combined[row] + (1.0 - alpha) * (c / max_c)
        true_row = k2_row_of[e2_gold]
        s_true = float(combined[true_row].item())
        # rank = 1 + number strictly greater
        n_above = int((combined > combined[true_row]).sum().item())
        rank = n_above + 1
        top_idx = int(torch.argmax(combined).item())
        ranks.append(rank); em.append(1 if rank == 1 else 0)
        correct += int(rank == 1)
        qids.append(i); pred_e2.append(k2_ids_sorted[top_idx])
        gold_e2.append(e2_gold); confidence.append(float(combined[top_idx].item()))
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(h1=f"{correct/max(len(em),1):.3f}")
    inference_s = time.perf_counter() - t
    print(f"  scored {len(eval_pairs)} pairs in {inference_s:.1f}s")

    ranks_a = np.asarray(ranks)
    em_a = np.asarray(em, dtype=float)
    conf_a = np.asarray(confidence, dtype=float)
    hits1 = float(em_a.mean())
    hits3 = float((ranks_a <= 3).mean())
    hits10 = float((ranks_a <= 10).mean())
    mrr = float((1.0 / np.maximum(ranks_a, 1)).mean())
    from quest_kg.eval.metrics import ece, aurc
    conf_norm = np.clip(conf_a, 0.0, 1.0)
    e_ece = float(ece(conf_norm, em_a, n_bins=15))
    e_aurc = float(aurc(conf_norm, em_a))

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
        "mean_conf": float(conf_norm.mean()),
        "method": "quest_kg_ea_merged_kg", "dataset": dataset_name,
        "llm": "none", "encoder": encoder_name, "seed": 0,
        "wallclock_s": round(wallclock, 1),
        "limit": len(eval_pairs), "kg_subset": None, "tag": tag,
        "candidate_pool_size": len(k2_ids_sorted),
        "scoring": f"hybrid: alpha={alpha} * cosine + {1-alpha} * structural(aligned-bridge propagation)",
        "alpha": alpha,
    }
    return out_json, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--format", choices=("dwy100k", "openea"), required=True)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tag", default="local-1080ti__merged-kg__seed0")
    ap.add_argument("--n_sample", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="alpha * cosine + (1-alpha) * structural")
    ap.add_argument("--propagate", action="store_true")
    ap.add_argument("--struct_hops", type=int, default=1,
                    help="Multi-hop K1 traversal depth for structural channel")
    ap.add_argument("--hop_decay", type=float, default=0.5,
                    help="Per-hop decay for distance>1 contributions")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    root = Path(args.root)
    data = load_dwy100k(root) if args.format == "dwy100k" else load_openea(root)
    out, df = run_merged_kg(data, args.dataset, args.encoder, args.tag,
                              n_sample=args.n_sample, alpha=args.alpha,
                              propagate=args.propagate,
                              struct_hops=args.struct_hops,
                              hop_decay=args.hop_decay)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_p = out_dir / f"quest_kg_ea_merged__{args.dataset}__{args.tag}.json"
    csv_p  = out_dir / f"quest_kg_ea_merged__{args.dataset}__{args.tag}.csv"
    json_p.write_text(json.dumps(out, indent=2))
    df.to_csv(csv_p, index=False)
    print(f"\n=== RESULTS for {args.dataset} ===")
    for k in ("hits_at_1", "hits_at_3", "hits_at_10", "mrr", "ECE", "AURC",
              "mean_latency_ms", "wallclock_s", "n"):
        print(f"  {k}: {out[k]}")
    print(f"Wrote {json_p.name}")


if __name__ == "__main__":
    main()
