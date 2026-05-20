"""Unified entity-alignment runner — QUEST-KG adapted for EA.

Supports two dataset formats:
  - DWY100K (BootEA GitHub): dbp_wd, dbp_yg
  - OpenEA IDS100K (figshare): EN_FR_100K_V2, EN_DE_100K_V2, D_W_100K_V2, D_Y_100K_V2

QUEST-KG-EA approach:
  Signature(e) = entity_name + " | " + " ".join(rel_label + " " + neighbor_label for each 1-hop neighbor)
  For each test anchor e1: cosine sim against all K2 signatures -> rank.
  Confidence = sim_top1 (used for ECE / AURC).

Output: results/quest_kg_ea__{dataset}__{tag}.{json,csv}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

URI_TAIL = re.compile(r"[^/#]+$")


def extract_name(s: str) -> str:
    """Extract human-readable name from a URI or use as-is."""
    if not s:
        return ""
    m = URI_TAIL.search(s.strip())
    local = m.group(0) if m else s
    return local.replace("_", " ").strip()


def parse_id_label_file(path: Path) -> dict[int, str]:
    """Parse ent_ids_X or rel_ids_X format: 'id\tlabel_or_uri'."""
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        out[int(parts[0])] = extract_name(parts[1]) if len(parts) > 1 else ""
    return out


def parse_triples(path: Path) -> list[tuple[int, int, int]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, r, t = line.split("\t")
        out.append((int(h), int(r), int(t)))
    return out


def parse_pairs(path: Path) -> list[tuple[int, int]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a, b = line.split("\t")
        out.append((int(a), int(b)))
    return out


def propagate_labels(ent2_orig: dict[int, str], ent1: dict[int, str],
                      train_pairs: list[tuple[int, int]],
                      triples2: list[tuple[int, int, int]],
                      n_propagation_rounds: int = 1) -> dict[int, str]:
    """Anchor-based label propagation for KGs where one side has opaque IDs.

    1. For each (e1, e2) in train_pairs, copy label(e1) to e2 if e2 looks opaque
       (i.e., starts with 'Q' followed by digits — Wikidata convention).
    2. For each remaining opaque e2, gather labels from 1-hop neighbors that DO
       have labels; concatenate them as a multi-sentence pseudo-label.
    """
    ent2 = dict(ent2_orig)
    opaque = re.compile(r"^(Q|P)\d+$")

    # Direct seed propagation
    for e1, e2 in train_pairs:
        if e2 in ent2 and opaque.match(ent2[e2]) and e1 in ent1:
            ent2[e2] = ent1[e1]  # adopt K1 label

    # Neighbor propagation (1 round by default)
    for _ in range(n_propagation_rounds):
        out_n: dict[int, list[int]] = {}
        in_n: dict[int, list[int]] = {}
        for h, _, t in triples2:
            out_n.setdefault(h, []).append(t)
            in_n.setdefault(t, []).append(h)
        new_ent2 = dict(ent2)
        for ent_id, lab in ent2.items():
            if not opaque.match(lab):
                continue
            cand_labels = []
            for n in (out_n.get(ent_id, []) + in_n.get(ent_id, []))[:8]:
                if n in ent2 and not opaque.match(ent2[n]):
                    cand_labels.append(ent2[n])
            if cand_labels:
                new_ent2[ent_id] = "neighbor_of " + ", ".join(cand_labels[:4])
        ent2 = new_ent2
    return ent2


def build_signatures(ent_labels: dict[int, str], rel_labels: dict[int, str],
                      triples: list[tuple[int, int, int]],
                      max_neighbors: int = 8) -> dict[int, str]:
    """For each entity, build a signature combining its name and 1-hop neighborhood."""
    out_neighbors: dict[int, list[tuple[int, int]]] = {}
    in_neighbors: dict[int, list[tuple[int, int]]] = {}
    for h, r, t in triples:
        out_neighbors.setdefault(h, []).append((r, t))
        in_neighbors.setdefault(t, []).append((r, h))

    signatures: dict[int, str] = {}
    for ent_id, name in ent_labels.items():
        sig_parts = [name]
        out_nb = out_neighbors.get(ent_id, [])[:max_neighbors]
        for r, t in out_nb:
            sig_parts.append(f"{rel_labels.get(r, '?')} {ent_labels.get(t, '?')}")
        in_nb = in_neighbors.get(ent_id, [])[:max_neighbors]
        for r, h in in_nb:
            sig_parts.append(f"{ent_labels.get(h, '?')} {rel_labels.get(r, '?')}")
        signatures[ent_id] = " | ".join(sig_parts)
    return signatures


def load_dwy100k(root: Path) -> dict:
    """DWY100K format from BootEA GitHub repo."""
    sub = root / "mapping" / "0_3"
    ent1 = parse_id_label_file(sub / "ent_ids_1")
    ent2 = parse_id_label_file(sub / "ent_ids_2")
    rel1 = parse_id_label_file(sub / "rel_ids_1")
    rel2 = parse_id_label_file(sub / "rel_ids_2")
    triples1 = parse_triples(sub / "triples_1")
    triples2 = parse_triples(sub / "triples_2")
    train_pairs = parse_pairs(sub / "sup_ent_ids")
    test_pairs = parse_pairs(sub / "ref_ent_ids")
    return dict(
        ent1=ent1, ent2=ent2, rel1=rel1, rel2=rel2,
        triples1=triples1, triples2=triples2,
        train_pairs=train_pairs, test_pairs=test_pairs,
    )


def _find_openea_file(root: Path, candidates: list[str]) -> Path | None:
    """Look for one of `candidates` either directly in root or one level below."""
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for name in candidates:
            p = sub / name
            if p.exists():
                return p
    return None


def load_openea(root: Path) -> dict:
    """OpenEA IDS100K format. Files: ent_links, rel_triples_1/2, attr_triples_1/2,
    721_5fold/1/{train,valid,test}_links.

    Some OpenEA archives put these files inside a `Mapping/` subfolder; some use
    slightly different file names. We resolve robustly."""
    # Resolve actual paths (some archives nest these inside Mapping/ etc.)
    rel1_path = _find_openea_file(root, ["rel_triples_1", "rel_triples1", "triples_1"])
    rel2_path = _find_openea_file(root, ["rel_triples_2", "rel_triples2", "triples_2"])
    if rel1_path is None or rel2_path is None:
        contents = sorted(p.name for p in root.iterdir())
        sub_contents = {}
        for sub in root.iterdir():
            if sub.is_dir():
                sub_contents[sub.name] = sorted(p.name for p in sub.iterdir())[:20]
        raise FileNotFoundError(
            f"OpenEA: could not find rel_triples_1/2 under {root}.\n"
            f"  contents: {contents}\n"
            f"  subfolders: {sub_contents}"
        )
    # Use the parent of rel_triples_1 as the effective root (handles nesting)
    base = rel1_path.parent
    print(f"  OpenEA effective base: {base}")
    # Read attribute triples to get labels (OpenEA's entity labels are here)
    def parse_attr_triples(p: Path) -> dict[str, str]:
        labels: dict[str, str] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            ent, attr, val = parts[0], parts[1], parts[2]
            if "label" in attr.lower() or "name" in attr.lower():
                # strip language tag etc.
                val = re.sub(r'"', "", val).split("@")[0].split("^^")[0].strip()
                if val and ent not in labels:
                    labels[ent] = val
        return labels

    def parse_rel_triples(p: Path) -> tuple[list[tuple[str, str, str]], set[str], set[str]]:
        triples = []
        ents, rels = set(), set()
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, r, t = line.split("\t")
            triples.append((h, r, t))
            ents.add(h); ents.add(t); rels.add(r)
        return triples, ents, rels

    def parse_pair_file(p: Path) -> list[tuple[str, str]]:
        return [tuple(line.split("\t")) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    rel_tri_1, ents1, rels1 = parse_rel_triples(rel1_path)
    rel_tri_2, ents2, rels2 = parse_rel_triples(rel2_path)
    attr1_p = _find_openea_file(base, ["attr_triples_1", "attr_triples1"])
    attr2_p = _find_openea_file(base, ["attr_triples_2", "attr_triples2"])
    attr_labels_1 = parse_attr_triples(attr1_p) if attr1_p else {}
    attr_labels_2 = parse_attr_triples(attr2_p) if attr2_p else {}

    # Build string -> int id mappings
    ents1_list = sorted(ents1); ents2_list = sorted(ents2)
    rels1_list = sorted(rels1); rels2_list = sorted(rels2)
    e1_id_of = {e: i for i, e in enumerate(ents1_list)}
    e2_id_of = {e: i for i, e in enumerate(ents2_list)}
    r1_id_of = {r: i for i, r in enumerate(rels1_list)}
    r2_id_of = {r: i for i, r in enumerate(rels2_list)}

    ent1 = {i: attr_labels_1.get(e, extract_name(e)) for e, i in e1_id_of.items()}
    ent2 = {i: attr_labels_2.get(e, extract_name(e)) for e, i in e2_id_of.items()}
    rel1 = {i: extract_name(r) for r, i in r1_id_of.items()}
    rel2 = {i: extract_name(r) for r, i in r2_id_of.items()}
    triples1 = [(e1_id_of[h], r1_id_of[r], e1_id_of[t]) for h, r, t in rel_tri_1]
    triples2 = [(e2_id_of[h], r2_id_of[r], e2_id_of[t]) for h, r, t in rel_tri_2]

    # Find seed/test links — OpenEA uses 721_5fold/1/{train,valid,test}_links
    fold_root = base / "721_5fold" / "1"
    if fold_root.exists():
        train_str = parse_pair_file(fold_root / "train_links")
        test_str = parse_pair_file(fold_root / "test_links")
    else:
        # fallback: ent_links is the full set
        ent_links_p = _find_openea_file(base, ["ent_links"])
        if ent_links_p is None:
            raise FileNotFoundError(f"OpenEA: no 721_5fold/1 nor ent_links under {base}")
        all_str = parse_pair_file(ent_links_p)
        n = len(all_str)
        train_str = all_str[: int(0.3 * n)]
        test_str = all_str[int(0.3 * n) :]
    train_pairs = [(e1_id_of[a], e2_id_of[b]) for a, b in train_str if a in e1_id_of and b in e2_id_of]
    test_pairs = [(e1_id_of[a], e2_id_of[b]) for a, b in test_str if a in e1_id_of and b in e2_id_of]

    return dict(
        ent1=ent1, ent2=ent2, rel1=rel1, rel2=rel2,
        triples1=triples1, triples2=triples2,
        train_pairs=train_pairs, test_pairs=test_pairs,
    )


def train_projection(anchor_embs: np.ndarray, target_embs: np.ndarray,
                      n_epochs: int = 100, lr: float = 1e-3,
                      hidden_dim: int = 384) -> "torch.nn.Module":
    """Train a small MLP to project K1 (anchor) embeddings into K2 (target) space.

    Loss = MSE + cosine-similarity (anchor-aware contrastive). Trained on seed
    alignment pairs from sup_ent_ids. Mirrors what MTransE/BootEA do at the
    embedding-alignment level, but on top of MiniLM features instead of from
    scratch.
    """
    import torch.nn as nn
    dim = anchor_embs.shape[1]
    model = nn.Sequential(
        nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        nn.Linear(hidden_dim, dim),
    ).cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    a_t = torch.as_tensor(anchor_embs, dtype=torch.float32, device="cuda")
    t_t = torch.as_tensor(target_embs, dtype=torch.float32, device="cuda")
    # Pre-normalize for cosine
    a_norm = torch.nn.functional.normalize(a_t, dim=1)
    t_norm = torch.nn.functional.normalize(t_t, dim=1)
    for ep in range(n_epochs):
        pred = model(a_t)
        pred_norm = torch.nn.functional.normalize(pred, dim=1)
        # MSE on unnormalized; cosine sim loss on normalized
        loss_mse = torch.nn.functional.mse_loss(pred, t_t)
        loss_cos = (1.0 - (pred_norm * t_norm).sum(dim=1)).mean()
        loss = 0.3 * loss_mse + loss_cos
        optim.zero_grad(); loss.backward(); optim.step()
        if ep == 0 or (ep + 1) % 20 == 0:
            with torch.no_grad():
                top1 = (pred_norm @ t_norm.T).argmax(dim=1)
                acc = (top1 == torch.arange(len(a_t), device="cuda")).float().mean().item()
            print(f"    proj epoch {ep+1:3d}: loss={loss.item():.4f} train-top1={acc:.3f}")
    model.eval()
    return model


def run_one(data: dict, dataset_name: str, n_sample: int | None, tag: str,
            encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            propagate: bool = False,
            train_projection_flag: bool = True) -> dict:
    t_total = time.perf_counter()
    print(f"[ea] {dataset_name}: K1={len(data['ent1'])} K2={len(data['ent2'])} "
          f"train_pairs={len(data['train_pairs'])} test_pairs={len(data['test_pairs'])}")

    if propagate:
        opaque = re.compile(r"^(Q|P)\d+$")
        n_before = sum(1 for v in data["ent2"].values() if opaque.match(v))
        data["ent2"] = propagate_labels(
            data["ent2"], data["ent1"], data["train_pairs"], data["triples2"]
        )
        n_after = sum(1 for v in data["ent2"].values() if opaque.match(v))
        print(f"[ea] label propagation: opaque K2 labels {n_before} -> {n_after} ({n_before - n_after} replaced)")

    print("[ea] building signatures...")
    t = time.perf_counter()
    sig1 = build_signatures(data["ent1"], data["rel1"], data["triples1"])
    sig2 = build_signatures(data["ent2"], data["rel2"], data["triples2"])
    print(f"  built in {time.perf_counter()-t:.1f}s. Examples:")
    sample_e = next(iter(sig1)); print(f"    K1[{sample_e}]: {sig1[sample_e][:120]}")
    sample_e = next(iter(sig2)); print(f"    K2[{sample_e}]: {sig2[sample_e][:120]}")

    # Encode K2 fully; encode K1 only for the test anchors we'll evaluate
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

    from quest_kg.data.encoders import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder(encoder_name, batch_size=256)
    print(f"[ea] encoder on {enc.device}")

    t = time.perf_counter()
    anchor_sigs = [sig1[e1] for e1, _ in eval_pairs]
    anchor_emb = enc(anchor_sigs)
    print(f"  encoded {len(anchor_sigs)} anchors in {time.perf_counter()-t:.1f}s")
    t = time.perf_counter()
    k2_emb = enc(k2_sigs)
    print(f"  encoded {len(k2_sigs)} K2 entities in {time.perf_counter()-t:.1f}s")

    # Train projection K1 -> K2 on the seed alignment pairs (sup_ent_ids).
    # Mirrors MTransE/BootEA's embedding-alignment step but starts from MiniLM
    # features so it converges in ~30s instead of hours.
    train_pairs_data = data["train_pairs"]
    if train_projection_flag and len(train_pairs_data) > 100:
        print(f"[ea] training projection on {len(train_pairs_data)} seed pairs...")
        t = time.perf_counter()
        train_e1_sigs = [sig1[e1] for e1, _ in train_pairs_data]
        train_e2_sigs = [sig2[e2] for _, e2 in train_pairs_data]
        # Encode only a sample (5000) of seed pairs to keep training fast
        max_train = min(5000, len(train_e1_sigs))
        train_idx = np.random.default_rng(0).choice(
            len(train_e1_sigs), size=max_train, replace=False)
        train_e1_embs = enc([train_e1_sigs[i] for i in train_idx])
        train_e2_embs = enc([train_e2_sigs[i] for i in train_idx])
        proj_model = train_projection(train_e1_embs, train_e2_embs, n_epochs=100)
        print(f"  trained in {time.perf_counter()-t:.1f}s on {max_train} pairs")
        # Apply projection to all anchor embeddings
        with torch.no_grad():
            anchor_t = torch.as_tensor(anchor_emb, dtype=torch.float32, device="cuda")
            anchor_emb_proj = proj_model(anchor_t).cpu().numpy()
        anchor_emb = anchor_emb_proj  # replace
        print("[ea] projection applied to anchor embeddings")

    # GPU normalize + matmul in chunks
    a_t = torch.as_tensor(anchor_emb, dtype=torch.float32, device="cuda")
    k_t = torch.as_tensor(k2_emb, dtype=torch.float32, device="cuda")
    a_t = torch.nn.functional.normalize(a_t, dim=1)
    k_t = torch.nn.functional.normalize(k_t, dim=1)

    t = time.perf_counter()
    ranks: list[int] = []
    confidence: list[float] = []  # top-1 cos sim
    sim_at_truth: list[float] = []
    em: list[int] = []
    qids: list[int] = []
    pred_e2: list[int] = []
    gold_e2: list[int] = []
    BATCH = 256
    try:
        from tqdm import tqdm
        pbar = tqdm(range(0, len(eval_pairs), BATCH),
                    desc=f"scoring {dataset_name}", unit="batch",
                    dynamic_ncols=True)
    except ImportError:
        pbar = range(0, len(eval_pairs), BATCH)
    correct = 0
    for i in pbar:
        ai = a_t[i : i + BATCH]
        sims = ai @ k_t.T
        for j, (e1, e2) in enumerate(eval_pairs[i : i + BATCH]):
            true_row = k2_row_of[e2]
            s_true = float(sims[j, true_row].item())
            row_max_idx = int(torch.argmax(sims[j]).item())
            row_max_val = float(sims[j, row_max_idx].item())
            n_above = int((sims[j] > s_true).sum().item())
            rank = n_above + 1
            ranks.append(rank)
            confidence.append(row_max_val)
            sim_at_truth.append(s_true)
            em.append(1 if rank == 1 else 0)
            correct += int(rank == 1)
            qids.append(i + j)
            pred_e2.append(k2_ids_sorted[row_max_idx])
            gold_e2.append(e2)
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

    # ECE / AURC
    from quest_kg.eval.metrics import ece, aurc
    # Confidence needs to be in [0, 1]; cosine sim is in [-1, 1]. Normalize: (s+1)/2
    conf_norm = np.clip((conf_a + 1.0) / 2.0, 0.0, 1.0)
    e_ece = float(ece(conf_norm, em_a, n_bins=15))
    e_aurc = float(aurc(conf_norm, em_a))

    # Per-query CSV
    df = pd.DataFrame({
        "qid": qids,
        "gold_e2": gold_e2,
        "pred_e2": pred_e2,
        "em": em,
        "x_rank": ranks,
        "confidence": conf_norm,
    })

    wallclock = time.perf_counter() - t_total
    out_json = {
        "n": len(eval_pairs), "n_attempted": len(eval_pairs),
        "abstention_rate": 0.0,
        "exact_match": hits1,
        "exact_match_when_attempted": hits1,
        "token_f1": None,
        "mean_latency_ms": (inference_s * 1000.0) / max(len(eval_pairs), 1),
        "p95_latency_ms": None,
        "task_type": "entity_alignment",
        "primary_metric": "hits_at_1",
        "primary_value": hits1,
        "hits_at_1": hits1,
        "hits_at_3": hits3,
        "hits_at_10": hits10,
        "mrr": mrr,
        "ECE": e_ece,
        "AURC": e_aurc,
        "n_bins_ece": 15,
        "mean_conf": float(conf_norm.mean()),
        "method": "quest_kg_ea",
        "dataset": dataset_name,
        "llm": "none",
        "encoder": encoder_name,
        "seed": 0,
        "wallclock_s": round(wallclock, 1),
        "limit": len(eval_pairs),
        "kg_subset": None,
        "tag": tag,
        "candidate_pool_size": len(k2_ids_sorted),
        "signature_strategy": "name + 1-hop in/out neighborhood (max 8 each)",
    }
    return out_json, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset name, e.g., dbp-wd, dbp-yg, ids-en-fr, ids-en-de, ids-d-w, ids-d-y")
    ap.add_argument("--root", required=True, help="dataset root directory")
    ap.add_argument("--format", choices=("dwy100k", "openea"), required=True)
    ap.add_argument("--n_sample", type=int, default=None,
                    help="number of test pairs to evaluate (None = all)")
    ap.add_argument("--tag", default="local-1080ti__symbolic__seed0")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--propagate", action="store_true",
                    help="Anchor-based label propagation (use for KGs with opaque IDs e.g., Wikidata Q-numbers)")
    ap.add_argument("--projection", action="store_true",
                    help="Enable seed-pair projection training (default OFF — empirically hurt on DBP-YG)")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Encoder model; use multilingual variant for cross-lingual datasets")
    args = ap.parse_args()

    root = Path(args.root)
    if args.format == "dwy100k":
        data = load_dwy100k(root)
    else:
        data = load_openea(root)

    out, df = run_one(data, args.dataset, args.n_sample, args.tag,
                       encoder_name=args.encoder,
                       propagate=args.propagate,
                       train_projection_flag=args.projection)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"quest_kg_ea__{args.dataset}__{args.tag}.json"
    csv_path  = out_dir / f"quest_kg_ea__{args.dataset}__{args.tag}.csv"
    json_path.write_text(json.dumps(out, indent=2))
    df.to_csv(csv_path, index=False)
    print(f"\n=== RESULTS for {args.dataset} ===")
    for k in ("hits_at_1", "hits_at_3", "hits_at_10", "mrr", "ECE", "AURC",
              "mean_latency_ms", "wallclock_s", "n"):
        print(f"  {k}: {out[k]}")
    print(f"\nWrote {json_path.name} and {csv_path.name}")


if __name__ == "__main__":
    main()
