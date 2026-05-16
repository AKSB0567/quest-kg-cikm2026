"""Local 1080 Ti smoke test with a real LLM (no Drive, no AWQ, no auth).

Downloads `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB fp16, non-gated) the first time
and `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) for encoding.
Runs QUEST-KG + 1 baseline (vanilla_rag) on real OrgAccess queries.

What this validates that the NumPy-only smoke could not:
  - PyTorch CUDA path works end-to-end on the 1080 Ti.
  - HF Transformers can load + generate from a real LLM on the 1080 Ti.
  - sentence-transformers encoder produces vectors that drive useful retrieval.
  - Baselines actually return predictions (not mock answers).

Run from repo root:
    .venv312/Scripts/python scripts/local_smoke_real_llm.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import torch

    print(f"[smoke] torch={torch.__version__}, cuda={torch.cuda.is_available()}, "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    from quest_kg.data.loaders import load_orgaccess
    from quest_kg.data.encoders import SentenceTransformerEncoder
    from quest_kg.data.llm import LLMInterface
    from quest_kg.retrieval.schema_aware import SchemaAwareRetriever
    from quest_kg.symbolic.orgaccess_datalog import OrgAccessChecker
    from quest_kg.inference import QuestKG
    from quest_kg.eval.harness import aggregate, run_method
    from baselines.vanilla_rag import VanillaRAG

    # --- Load data ---
    print(f"\n[smoke] loading OrgAccess from {ROOT}/data/raw/orgaccess/ ...")
    ds = load_orgaccess(ROOT / "data")
    print(f"[smoke] {ds.summary()}")

    # --- Encoder (small, fast, will download ~90 MB first time) ---
    print(f"\n[smoke] loading encoder all-MiniLM-L6-v2 ...")
    t0 = time.perf_counter()
    enc = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
    print(f"[smoke] encoder ready in {time.perf_counter() - t0:.1f}s, device={enc.device}")

    # --- Build retriever (precompute embeddings) ---
    print(f"\n[smoke] building retriever on {len(ds.triples)} triples (precompute=True) ...")
    t0 = time.perf_counter()
    retriever = SchemaAwareRetriever(
        triples=ds.triples, encoder=enc, k=2, top_k=32, precompute=True,
    )
    print(f"[smoke] retriever ready in {time.perf_counter() - t0:.1f}s; "
          f"entity_emb={retriever._entity_emb.shape}, edge_emb={retriever._edge_emb.shape}")

    # --- QUEST-KG inference on 20 queries ---
    ctx = {int(i + 1): c for i, c in enumerate(ds.metadata.get("context_schedule", []))}
    checker = OrgAccessChecker(active_contexts_at_t=ctx, strict=False)
    qkg = QuestKG(retriever=retriever, symbolic_checker=checker, task_type="yes_no")
    queries = ds.queries[:20]
    print(f"\n[smoke] running QUEST-KG on {len(queries)} queries ...")
    t0 = time.perf_counter()
    qkg_results = run_method(qkg, queries, method_name="quest_kg", is_questkg=True)
    qkg_agg = aggregate(qkg_results)
    print(f"[smoke] QUEST-KG done in {time.perf_counter() - t0:.1f}s, agg={qkg_agg}")
    # Show 3 sample predictions
    print("  Sample predictions (QUEST-KG):")
    for r in qkg_results[:3]:
        print(f"    qid={r.qid} gold={r.gold!r:>5}  pred={r.prediction!r:>20}  "
              f"em={r.em} f1={r.f1:.2f} abst={r.abstained}")

    # --- LLM load + vanilla_rag baseline ---
    print(f"\n[smoke] loading Qwen2.5-1.5B-Instruct on CUDA (first run downloads ~3 GB) ...")
    t0 = time.perf_counter()
    # We use the raw HF interface here rather than LLMInterface because LLMInterface
    # is currently tuned for AWQ models on Drive; a tiny fp16 model is simpler:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[smoke] LLM ready in {time.perf_counter() - t0:.1f}s "
          f"(mem alloc ~{torch.cuda.memory_allocated()/1024**3:.2f} GB)")

    # Wrap with the same interface VanillaRAG expects
    class TinyLLM:
        def __init__(self, model, tok):
            self.model, self.tok = model, tok

        @torch.inference_mode()
        def generate(self, prompts, batch_size=2, max_new_tokens=24):
            out = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=1024).to("cuda")
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )
                pl = enc.input_ids.shape[1]
                for j, seq in enumerate(gen):
                    out.append(self.tok.decode(seq[pl:], skip_special_tokens=True).strip())
            return out

    tiny_llm = TinyLLM(model, tok)

    # Smoke test generation
    print("\n[smoke] LLM generation smoke test:")
    sample = tiny_llm.generate(
        ["The capital of France is", "2 + 2 ="],
        batch_size=2, max_new_tokens=10,
    )
    for q, a in zip(["The capital of France is", "2 + 2 ="], sample):
        print(f"    {q!r} -> {a!r}")

    # --- vanilla_rag baseline ---
    print(f"\n[smoke] running vanilla_rag on {len(queries)} queries ...")
    rag = VanillaRAG(triples=ds.triples, encoder=enc, llm=tiny_llm, top_k=6)
    t0 = time.perf_counter()
    rag_results = run_method(rag, queries, method_name="vanilla_rag", is_questkg=False)
    rag_agg = aggregate(rag_results)
    print(f"[smoke] vanilla_rag done in {time.perf_counter() - t0:.1f}s, agg={rag_agg}")
    print("  Sample predictions (vanilla_rag):")
    for r in rag_results[:3]:
        print(f"    qid={r.qid} gold={r.gold!r:>5}  pred={r.prediction!r:>40}  em={r.em} f1={r.f1:.2f}")

    # --- Final verdict ---
    print("\n" + "=" * 72)
    print(f"QUEST-KG    : EM={qkg_agg['exact_match']:.3f}  F1={qkg_agg['token_f1']:.3f}  "
          f"abst={qkg_agg['abstention_rate']:.2f}  lat={qkg_agg['mean_latency_ms']:.0f}ms")
    print(f"vanilla_rag : EM={rag_agg['exact_match']:.3f}  F1={rag_agg['token_f1']:.3f}  "
          f"abst={rag_agg['abstention_rate']:.2f}  lat={rag_agg['mean_latency_ms']:.0f}ms")
    print("=" * 72)
    print("PASS — pipeline + LLM + baseline run cleanly on 1080 Ti CUDA")


if __name__ == "__main__":
    main()
