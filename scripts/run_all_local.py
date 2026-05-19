"""Overnight local runner: all 24 experiments on 1080 Ti with tqdm + STATUS file.

Resumable: each (method, dataset) writes a JSON summary as soon as it's done.
Re-running skips experiments with an existing JSON. Per-experiment failures
are caught and reported in the final summary; the next experiment continues.

Progress reporting:
  - tqdm bars to stdout (visible if you `tail -f run_all_local.log`)
  - `STATUS.txt` is rewritten every few seconds with a compact status line + grid
  - You can monitor live with:  tail -f STATUS.txt   or   tail -f run_all_local.log

Usage:
    .venv312/Scripts/python -X utf8 -u scripts/run_all_local.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "data"
OUT_DIR   = ROOT / "results"
LOG_DIR   = ROOT / "logs"
STATUS    = ROOT / "STATUS.txt"
LOG_FILE  = ROOT / "run_all_local.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Configuration --------------------------------

DATASETS = ["orgaccess", "icews18", "webqsp", "cwq"]   # fast -> slow
METHODS_NOLLM = ["quest_kg"]                            # no LLM needed
# `quest_kg_llm` is QUEST-KG's retrieval + reasoning with an LLM verbalization
# step for entity-QA datasets; on ICEWS18/OrgAccess it falls back to the
# symbolic prediction (LLM is not used). Lets us report both variants in §4.
METHODS_LLM   = ["quest_kg_llm", "vanilla_rag", "graphrag", "tog1", "tog2", "cok"]
ALL_METHODS   = METHODS_NOLLM + METHODS_LLM
# Per-dataset query limit. Larger N reduces variance vs 50-query original.
# OrgAccess is cheap (no LLM for quest_kg, small KG), so we run the full test.
# ICEWS18 has 49k test queries; cost-bounded at 150 (still ~3x previous N).
LIMIT_QUERIES_PER_DATASET = {
    "orgaccess": 750,
    "icews18": 150,
    "webqsp": 500,
    "cwq": 500,
}
# Per-dataset KG triple cap. 100k for QA datasets lifts the gold-reachability
# ceiling (at 20k only 10/50 WebQSP gold were in the subgraph).
KG_SUBSET_PER_DATASET = {
    "orgaccess": 20000,     # actual size is 1028, no cap needed
    "icews18": 20000,       # 50k regressed earlier (iter 2c)
    "webqsp": 100000,
    "cwq": 100000,
}
# Legacy globals kept for backward-compat (resume status fields, etc.)
LIMIT_QUERIES = 500
KG_SUBSET     = 100000
ENCODER_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
# Qwen2.5-3B-Instruct fp16: 5.75 GB on 1080 Ti, verified loadable.
# 7B with bnb 4-bit segfaults on Pascal sm_61; 3B fp16 is the largest model
# that fits reliably for matched-condition comparison.
LLM_ID        = "Qwen/Qwen2.5-3B-Instruct"
SEED          = 0
TAG           = f"local-1080ti__{LLM_ID.split('/')[-1]}__seed{SEED}"

# ------------------------ Logging / status helpers -------------------------

_log_fp = open(LOG_FILE, "a", encoding="utf-8", buffering=1)  # line-buffered


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_fp.write(line + "\n")


def write_status(state: dict) -> None:
    """Atomic-ish rewrite of STATUS.txt for `tail -f` monitoring."""
    lines = ["QUEST-KG local overnight run", "=" * 60]
    lines.append(f"Started:        {state['start_time']}")
    lines.append(f"Now:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Elapsed:        {state['elapsed']}")
    lines.append(f"ETA (rough):    {state.get('eta', 'pending')}")
    lines.append(f"Current:        {state.get('current', '(setup)')}")
    lines.append(f"Done:           {state['n_done']}/{state['n_total']}")
    lines.append("")
    lines.append("Grid (dataset \\ method):")
    header = f"{'dataset':<12}" + "".join(f"{m:>14}" for m in ALL_METHODS)
    lines.append(header)
    for ds in DATASETS:
        cells = []
        for m in ALL_METHODS:
            tag = f"{m}__{ds}__{TAG}"
            jpath = OUT_DIR / f"{tag}.json"
            if jpath.exists():
                try:
                    s = json.loads(jpath.read_text())
                    em = s.get("exact_match")
                    if em is None:
                        cells.append(f"{'  err':>14}")
                    else:
                        cells.append(f"{em:>13.3f}")
                except Exception:
                    cells.append(f"{'  ?':>14}")
            elif state.get("current_pair") == (m, ds):
                cells.append(f"{'  RUNNING':>14}")
            else:
                cells.append(f"{'  -':>14}")
        lines.append(f"{ds:<12}" + "".join(cells))
    lines.append("")
    lines.append("(EM > 0 = method produced matched predictions; `-` = pending; `err` = run errored)")
    STATUS.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------- Main --------------------------------------

def main():
    t_start = time.perf_counter()
    state = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed":    "0s",
        "n_done":     0,
        "n_total":    len(DATASETS) * len(ALL_METHODS),
        "current":    "(loading deps)",
    }
    write_status(state)

    # --- Lazy imports so error messages on missing deps appear in log ---
    log("Importing deps...")
    import torch
    from tqdm import tqdm
    from quest_kg.data.encoders import SentenceTransformerEncoder
    from quest_kg.data.loaders import load_dataset
    from quest_kg.eval.harness import aggregate, run_method, to_dataframe
    from quest_kg.inference import QuestKG
    from quest_kg.retrieval.schema_aware import SchemaAwareRetriever
    from quest_kg.symbolic.icews18_temporal import ICEWS18Checker
    from quest_kg.symbolic.orgaccess_datalog import OrgAccessChecker
    from quest_kg.symbolic.webqsp_freebase import FreebaseChecker
    from baselines.vanilla_rag import VanillaRAG
    from baselines.graphrag import GraphRAG
    from baselines.tog import ToG1, ToG2
    from baselines.chain_of_knowledge import ChainOfKnowledge

    log(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"device: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")

    # --- Load encoder once (~500MB on GPU) ---
    state["current"] = "loading encoder"; write_status(state)
    log(f"Loading encoder {ENCODER_NAME} ...")
    encoder = SentenceTransformerEncoder(ENCODER_NAME)
    log(f"  encoder ready on {encoder.device}")

    # --- LLM lazy: only loaded when first LLM-method runs ---
    llm = [None]   # mutable cell so closure can fill it

    def get_llm():
        if llm[0] is None:
            log(f"Loading LLM {LLM_ID} on CUDA ...")
            t0 = time.perf_counter()
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True, padding_side="left")
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                LLM_ID, torch_dtype=torch.float16, device_map="cuda",
                trust_remote_code=True,
            )
            model.eval()

            class TinyLLM:
                def __init__(self, model, tok):
                    self.model, self.tok = model, tok

                @torch.inference_mode()
                def generate(self, prompts, batch_size=2, max_new_tokens=24):
                    out = []
                    for i in range(0, len(prompts), batch_size):
                        b = prompts[i:i+batch_size]
                        enc = self.tok(b, return_tensors="pt", padding=True,
                                       truncation=True, max_length=1024).to("cuda")
                        gen = self.model.generate(
                            **enc, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=self.tok.eos_token_id,
                        )
                        pl = enc.input_ids.shape[1]
                        for j, seq in enumerate(gen):
                            out.append(self.tok.decode(seq[pl:], skip_special_tokens=True).strip())
                    return out

            llm[0] = TinyLLM(model, tok)
            log(f"  LLM ready in {time.perf_counter()-t0:.1f}s "
                f"(mem ~{torch.cuda.memory_allocated()/1024**3:.2f} GB)")
        return llm[0]

    # --- Iterate datasets x methods ---
    overall = tqdm(total=len(DATASETS) * len(ALL_METHODS), desc="all experiments", position=0)
    for ds_name in DATASETS:
        state["current"] = f"loading dataset {ds_name}"; write_status(state)
        log(f"\n=== Dataset: {ds_name} ===")
        try:
            ds = load_dataset(ds_name, DATA_ROOT)
        except Exception as e:
            log(f"  load_dataset({ds_name}) FAILED: {e}")
            traceback.print_exc(file=_log_fp)
            for m in ALL_METHODS:
                tag = f"{m}__{ds_name}__{TAG}"
                (OUT_DIR / f"{tag}.json").write_text(json.dumps(
                    {"method": m, "dataset": ds_name, "error": f"load_dataset failed: {e}", "n": 0},
                    indent=2))
                state["n_done"] += 1; overall.update(1)
            write_status(state)
            continue

        # Per-dataset subset: 100k for QA (more gold reachable), 20k otherwise.
        # Iter 5b: for WebQSP/CWQ, build KG from union of first N per-query
        # graphs (graph_triple_idx) rather than truncating to fixed 100k.
        # Lifts gold-reachability ceiling dramatically (CWQ 0.05 -> 0.10,
        # WebQSP 0.078 -> 0.20 at N=500). Baselines get the same union KG.
        ds_subset_cap = KG_SUBSET_PER_DATASET.get(ds_name, KG_SUBSET)
        if ds_name in ("webqsp", "cwq") and ds.queries and ds.queries[0].get("graph_triple_idx"):
            n_eval = LIMIT_QUERIES_PER_DATASET.get(ds_name, LIMIT_QUERIES) or len(ds.queries)
            queries_eval = ds.queries[: n_eval]
            all_idx_set: set[int] = set()
            for q in queries_eval:
                all_idx_set.update(q.get("graph_triple_idx", []))
            sorted_idx = sorted(all_idx_set)
            remap = {old: new for new, old in enumerate(sorted_idx)}
            ds.triples = [ds.triples[i] for i in sorted_idx]
            for q in queries_eval:
                q["graph_triple_idx"] = [remap[i] for i in q.get("graph_triple_idx", []) if i in remap]
            log(f"  per-query KG: union of first {n_eval} graphs = {len(ds.triples)} triples")
        elif ds_subset_cap and len(ds.triples) > ds_subset_cap:
            log(f"  subsetting KG: {len(ds.triples)} -> {ds_subset_cap}")
            ds.triples = ds.triples[:ds_subset_cap]
        log(f"  {ds.summary()}")

        # Build retriever once per dataset (shared across all methods + QUEST-KG).
        # Tuned per-dataset via grid search on 50 queries:
        #   WebQSP top_k=128 -> 3/50 hits  (best of 64/128/256)
        #   CWQ    top_k=64  -> 2/50 hits  (best)
        #   ICEWS18 top_k=32 (default)     (50k regressed in iter 2c)
        #   OrgAccess top_k=32             (default, tiny KG)
        top_k_for_dataset = {
            "webqsp": 128,
            "cwq": 64,
        }.get(ds_name, 32)
        # ICEWS18 is a directional event graph; reverse edges add noise without
        # uncovering valid candidate tails. QA datasets (Freebase) benefit from
        # bidirectional expansion because answers often appear as triple objects.
        bidir = ds_name != "icews18"
        state["current"] = f"{ds_name}: building retriever ({len(ds.triples)} triples)"; write_status(state)
        log(f"  building retriever (precompute=True) on {len(ds.triples)} triples, "
            f"top_k={top_k_for_dataset}, bidirectional={bidir} ...")
        t0 = time.perf_counter()
        retriever = SchemaAwareRetriever(
            triples=ds.triples, encoder=encoder, k=2, top_k=top_k_for_dataset, precompute=True,
            bidirectional=bidir,
        )
        log(f"  retriever ready in {time.perf_counter()-t0:.1f}s; "
            f"entity_emb={retriever._entity_emb.shape if retriever._entity_emb is not None else None}, "
            f"edge_emb={retriever._edge_emb.shape if retriever._edge_emb is not None else None}")

        # Choose symbolic + task_type per dataset.
        # `task_type` here is the QuestKG inference mode ("entity" / "yes_no").
        # `eval_task_type` is the metric-aggregator mode that selects the right
        # headline (qa/access_control/link_prediction). Keep them separate —
        # they coincidentally overlap on names sometimes.
        if ds_name in ("webqsp", "cwq"):
            checker = FreebaseChecker(); task_type = "entity"; eval_task_type = "qa"
        elif ds_name == "icews18":
            checker = ICEWS18Checker(); task_type = "entity"; eval_task_type = "link_prediction"
        elif ds_name == "orgaccess":
            ctx = {int(i+1): c for i, c in enumerate(ds.metadata.get("context_schedule", []))}
            checker = OrgAccessChecker(active_contexts_at_t=ctx, strict=False)
            task_type = "yes_no"; eval_task_type = "access_control"
        else:
            checker = FreebaseChecker(); task_type = "entity"; eval_task_type = "qa"

        # Per-dataset query limit (richer than the legacy single LIMIT_QUERIES).
        ds_limit = LIMIT_QUERIES_PER_DATASET.get(ds_name, LIMIT_QUERIES)
        queries = ds.queries[:ds_limit] if ds_limit else ds.queries
        log(f"  evaluating {len(queries)} queries (limit={ds_limit})")

        # Iterate methods
        for method_name in ALL_METHODS:
            tag = f"{method_name}__{ds_name}__{TAG}"
            json_path = OUT_DIR / f"{tag}.json"
            csv_path  = OUT_DIR / f"{tag}.csv"
            state["current"] = f"{ds_name} / {method_name}"
            state["current_pair"] = (method_name, ds_name)
            state["elapsed"] = str(timedelta(seconds=int(time.perf_counter()-t_start)))
            if state["n_done"]:
                avg = (time.perf_counter()-t_start) / state["n_done"]
                state["eta"] = str(timedelta(seconds=int(avg * (state["n_total"] - state["n_done"]))))
            write_status(state)

            if json_path.exists():
                log(f"  SKIP {tag} (already done)")
                state["n_done"] += 1; overall.update(1); write_status(state)
                continue

            log(f"  RUN  {tag}")
            try:
                if method_name in ("quest_kg", "quest_kg_llm"):
                    # Answer-side cosine rescoring only helps QA datasets where
                    # entities have meaningful surface forms. Disable for
                    # ICEWS18 (integer entity IDs) so it doesn't add noise.
                    use_ans_rescore = ds_name in ("webqsp", "cwq")
                    # quest_kg_llm: optional LLM verbalization for entity QA.
                    # Same retrieval + reasoning as pure QUEST-KG, only the
                    # final answer-extraction step is delegated to the LLM.
                    llm_for_qkg = None
                    if method_name == "quest_kg_llm" and ds_name in ("webqsp", "cwq"):
                        llm_for_qkg = get_llm()
                    method = QuestKG(retriever=retriever, symbolic_checker=checker,
                                     task_type=task_type,
                                     answer_rescoring=use_ans_rescore,
                                     llm=llm_for_qkg)
                    is_qkg = True
                else:
                    lm = get_llm()
                    if method_name == "vanilla_rag":
                        method = VanillaRAG(ds.triples, encoder, lm, top_k=6)
                    elif method_name == "graphrag":
                        method = GraphRAG(ds.triples, encoder, lm, k=2, top_k=16)
                    elif method_name == "tog1":
                        method = ToG1(ds.triples, encoder, lm, max_depth=2, beam_width=3)
                    elif method_name == "tog2":
                        method = ToG2(ds.triples, encoder, lm, max_depth=2, beam_width=3)
                    elif method_name == "cok":
                        method = ChainOfKnowledge(ds.triples, encoder, lm, max_steps=2, top_k=4)
                    is_qkg = False

                t0 = time.perf_counter()
                # Wrap queries with tqdm by passing the iterator through (run_method already iterates;
                # we surface a tqdm by chunking + intermittent status).
                results = []
                for i in tqdm(range(0, len(queries), 10), desc=f"{method_name} {ds_name}",
                              position=1, leave=False, ncols=80):
                    sub = queries[i:i+10]
                    results.extend(run_method(method, sub, method_name=method_name, is_questkg=is_qkg))
                elapsed = time.perf_counter() - t0
                agg = aggregate(results, task_type=eval_task_type)
                agg.update({
                    "method": method_name, "dataset": ds_name, "llm": LLM_ID,
                    "encoder": ENCODER_NAME, "seed": SEED, "wallclock_s": round(elapsed, 1),
                    "limit": LIMIT_QUERIES, "kg_subset": KG_SUBSET,
                })
                df = to_dataframe(results)
                df.to_csv(csv_path, index=False)
                json_path.write_text(json.dumps(agg, indent=2))
                log(f"    OK {elapsed:.0f}s  EM={agg['exact_match']:.3f}  F1={agg['token_f1']:.3f}  "
                    f"abst={agg['abstention_rate']:.2f}  lat={agg['mean_latency_ms']:.0f}ms")
            except Exception as e:
                log(f"    FAIL: {e}")
                traceback.print_exc(file=_log_fp)
                err = {"method": method_name, "dataset": ds_name, "error": str(e)[:500], "n": 0}
                json_path.write_text(json.dumps(err, indent=2))
            finally:
                state["n_done"] += 1
                overall.update(1)
                write_status(state)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Free retriever between datasets (entity/edge embeddings can be big)
        del retriever
        gc.collect()

    state["current"] = "DONE"; state["elapsed"] = str(timedelta(seconds=int(time.perf_counter()-t_start)))
    state["eta"] = "00:00:00"
    write_status(state)
    overall.close()

    # Final aggregate
    log("\n=== Final pivot ===")
    import pandas as pd
    rows = []
    for ds_name in DATASETS:
        for m in ALL_METHODS:
            jp = OUT_DIR / f"{m}__{ds_name}__{TAG}.json"
            if not jp.exists():
                continue
            try:
                s = json.loads(jp.read_text())
                rows.append({
                    "method": m, "dataset": ds_name,
                    "EM": s.get("exact_match"),
                    "F1": s.get("token_f1"),
                    "abst": s.get("abstention_rate"),
                    "lat_ms": s.get("mean_latency_ms"),
                    "n": s.get("n"),
                    "err": s.get("error", ""),
                })
            except Exception:
                pass
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"_summary_{TAG}.csv", index=False)
    log("\nLong-form table:\n" + df.to_string(index=False))
    log("\nEM pivot:\n" + df.pivot(index="method", columns="dataset", values="EM").round(3).to_string())
    log("\nDone. Results in: " + str(OUT_DIR))
    _log_fp.close()


if __name__ == "__main__":
    main()
