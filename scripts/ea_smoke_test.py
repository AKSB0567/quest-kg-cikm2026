"""Iter 6j SMOKE TEST: Can QUEST-KG-style retrieval do entity alignment?

Approach:
  1. Load DBP-YG (DWY100K family, monolingual English, both sides have names).
  2. Build name strings for each entity in K1 (DBpedia) and K2 (YAGO).
  3. Encode names with MiniLM-L6-v2 (same encoder as the rest of the paper).
  4. For each of 1000 sampled test alignment pairs (e1, e2):
       a. Compute cosine similarity from emb(e1) to all 100K K2 embeddings.
       b. Find rank of the true e2.
  5. Report Hits@1, Hits@10, MRR.

If Hits@1 >= 0.40 on this pure-name baseline, the dataset is workable for
QUEST-KG's adapted retrieval. We then layer the QUEST-KG machinery on top
(1-hop neighborhood signatures via evidential MP).

Output: results/ea_smoke__dbp-yg__name-only__local-1080ti.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DBPYG = ROOT / "data" / "raw" / "dwy100k" / "BootEA" / "dataset" / "DWY100K" / "dbp_yg" / "mapping" / "0_3"
OUT_JSON = ROOT / "results" / "ea_smoke__dbp-yg__name-only__local-1080ti.json"
N_SAMPLE = 1000  # smoke test pairs
SEED = 0


def parse_ent_ids(path: Path) -> tuple[list[int], list[str]]:
    """Returns parallel lists (ids, names). Names extracted from URI tail
    for DBpedia, used as-is for YAGO. Underscores -> spaces."""
    ids, names = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        ent_id = int(parts[0])
        uri = parts[1] if len(parts) > 1 else ""
        # Extract local name from URI tail
        m = re.search(r"[^/]+$", uri)
        local = m.group(0) if m else uri
        name = local.replace("_", " ").strip()
        ids.append(ent_id)
        names.append(name)
    return ids, names


def parse_ref_pairs(path: Path) -> list[tuple[int, int]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a, b = line.split("\t")
        pairs.append((int(a), int(b)))
    return pairs


def main():
    t_total = time.perf_counter()
    print(f"[ea-smoke] loading DBP-YG from {DBPYG}")
    k1_ids, k1_names = parse_ent_ids(DBPYG / "ent_ids_1")
    k2_ids, k2_names = parse_ent_ids(DBPYG / "ent_ids_2")
    test_pairs = parse_ref_pairs(DBPYG / "ref_ent_ids")
    print(f"  K1 (DBpedia): {len(k1_ids)} entities  e.g. {k1_names[0]!r}, {k1_names[1]!r}")
    print(f"  K2 (YAGO):    {len(k2_ids)} entities  e.g. {k2_names[0]!r}, {k2_names[1]!r}")
    print(f"  test pairs:   {len(test_pairs)}")

    # Build index maps: ent_id -> row index in the (id-aligned) name list
    k1_idx_of = {eid: i for i, eid in enumerate(k1_ids)}
    k2_idx_of = {eid: i for i, eid in enumerate(k2_ids)}

    # Sample test pairs
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(test_pairs), size=min(N_SAMPLE, len(test_pairs)),
                             replace=False)
    sample_pairs = [test_pairs[i] for i in sample_idx]

    # Encode all K2 names + sampled K1 anchor names with MiniLM.
    # We don't need to encode all 100K K1 entities — only the ~1000 anchors.
    from quest_kg.data.encoders import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2",
                                       batch_size=256)
    print(f"[ea-smoke] encoder ready on {enc.device}")

    anchor_names = [k1_names[k1_idx_of[e1]] for e1, _ in sample_pairs]
    t = time.perf_counter()
    anchor_embs = enc(anchor_names)  # (N_SAMPLE, 384)
    print(f"  encoded {len(anchor_names)} anchor names in {time.perf_counter()-t:.1f}s")
    t = time.perf_counter()
    k2_embs = enc(k2_names)  # (100000, 384)
    print(f"  encoded {len(k2_names)} K2 names in {time.perf_counter()-t:.1f}s")

    # Cast to GPU and L2-normalize -> cosine similarity = dot product
    anchor_t = torch.as_tensor(anchor_embs, dtype=torch.float32, device="cuda")
    k2_t = torch.as_tensor(k2_embs, dtype=torch.float32, device="cuda")
    anchor_t = torch.nn.functional.normalize(anchor_t, dim=1)
    k2_t = torch.nn.functional.normalize(k2_t, dim=1)

    # Similarity: (N_SAMPLE, 100K). For each anchor, find rank of true target.
    t = time.perf_counter()
    # Process in chunks of 256 anchors at a time to avoid memory spikes
    ranks: list[int] = []
    sim_at_truth: list[float] = []
    for i in range(0, len(sample_pairs), 256):
        ai = anchor_t[i : i + 256]
        sims = ai @ k2_t.T  # (B, 100K)
        # For each anchor, get sim at true target, then rank = #candidates with sim > sim_at_truth + 1
        for j, (e1, e2) in enumerate(sample_pairs[i : i + 256]):
            true_row = k2_idx_of[e2]
            s_true = float(sims[j, true_row].item())
            n_above = int((sims[j] > sims[j, true_row]).sum().item())
            ranks.append(n_above + 1)
            sim_at_truth.append(s_true)
    print(f"  scored {len(sample_pairs)} anchors in {time.perf_counter()-t:.1f}s")

    ranks_a = np.asarray(ranks, dtype=np.int64)
    hits1 = float((ranks_a == 1).mean())
    hits3 = float((ranks_a <= 3).mean())
    hits10 = float((ranks_a <= 10).mean())
    mrr = float((1.0 / np.maximum(ranks_a, 1)).mean())
    median_rank = float(np.median(ranks_a))
    mean_rank = float(np.mean(ranks_a))

    print()
    print("=" * 60)
    print("DBP-YG smoke test results (name-only baseline)")
    print("=" * 60)
    print(f"  n_test_pairs      : {len(sample_pairs)}")
    print(f"  candidate_pool    : {len(k2_ids)} K2 entities")
    print(f"  Hits@1            : {hits1:.4f}")
    print(f"  Hits@3            : {hits3:.4f}")
    print(f"  Hits@10           : {hits10:.4f}")
    print(f"  MRR               : {mrr:.4f}")
    print(f"  median rank       : {median_rank:.0f}")
    print(f"  mean rank         : {mean_rank:.1f}")
    print(f"  total wall (s)    : {time.perf_counter()-t_total:.1f}")
    print()
    print("GATE: Hits@1 >= 0.60 -> proceed; 0.40-0.60 -> your call;")
    print("                    < 0.40 -> pivot to Option B (cite published only)")

    out = {
        "dataset": "dbp-yg",
        "task_type": "entity_alignment",
        "method": "name_only_minilm",
        "n_sample": len(sample_pairs),
        "n_candidates": len(k2_ids),
        "hits_at_1": hits1,
        "hits_at_3": hits3,
        "hits_at_10": hits10,
        "mrr": mrr,
        "median_rank": median_rank,
        "mean_rank": mean_rank,
        "wallclock_s": round(time.perf_counter() - t_total, 1),
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "seed": SEED,
        "tag": "ea-smoke__dbp-yg__name-only__local-1080ti",
        "notes": "Smoke test: pure name cosine baseline. If Hits@1 >= 0.60, full QUEST-KG adaptation is the next step (neighborhood-aware retrieval + evidential MP).",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON.name}")


if __name__ == "__main__":
    main()
