# QUEST-KG (CIKM 2026)

Evidence-aware, uncertainty-calibrated Graph-RAG for trustworthy knowledge graph reasoning.

**Status:** Active sprint, target submission 2026-05-23 (CIKM 2026).

---

## Repository layout

```
quest_kg/                 Core package
├── retrieval/            Schema-aware Graph-RAG retrieval (FAISS-GPU + Neo4j + ontology align)
├── mp/                   Evidential message passing (φ_ij with provenance + schema + uncertainty)
├── symbolic/             Per-task symbolic constraint checkers (Datalog rules, ontology checks)
├── abstention/           Posterior-mass + entropy abstention head
├── extractor/            EAV/ERE triple extractor (reuses T1 LLaMA-3.1-8B QLoRA)
├── train/                Training loops (L_task + λ₁L_cal + λ₂L_cons)
├── eval/                 Metrics: Hits@k, MRR, F1, EM, ECE, Brier, NLL, AURC
└── utils/                Shared utilities incl. Colab checkpoint helpers

scripts/
├── download/             Dataset download scripts (DBP15K, WebQSP, CWQ, ICEWS18, OrgAccess)
└── quantize/             4-bit AWQ quantization for 7-8B LLMs

notebooks/
├── setup_all.ipynb            One-shot: mount Drive + clone + install + HF login +
│                              download data + quantize 4 LLMs + smoke test.
│                              Fully idempotent. Re-runs skip everything already done.
└── template_resumable.ipynb   Template for checkpoint-resumable experiment notebooks

baselines/                Per-baseline reproduction harnesses
configs/                  Hyperparameters
data/                     (gitignored) raw + processed datasets
paper/                    LaTeX source
tests/                    Unit tests
```

## Compute setup

- **Local workstation:** Windows 11, GTX 1080 Ti (11 GB VRAM), CUDA 12.6. Used for development, FAISS, light experiments, paper writing.
- **Cloud:** Colab Pro ($10/mo) — T4/V100/sometimes A100, 12-hr sessions, **no background execution.** All long-running experiments must checkpoint to Drive and resume across sessions.
- **Drive:** [`quest_kg/`](https://drive.google.com/drive/folders/1Ee8ABDrva-dUFMsk7AC6OeaACJfEedMM) — data, model weights, checkpoints, results.

## Datasets

| Dataset | Task | Source |
|---|---|---|
| DBP15K (zh_en, ja_en, fr_en) | Cross-lingual entity alignment between DBpedia editions | HF mirrors / JAPE |
| WebQSP | Multi-hop QA over Freebase | Microsoft |
| ComplexWebQuestions (CWQ) | Compositional multi-hop QA | TAU-NLP |
| ICEWS18 | Temporal event reasoning | RE-Net / TKG-Benchmark |
| OrgAccess | Dynamic enterprise access control | Synthetic (this repo) |

## LLM backbones

| Model | Gating | Use | Storage |
|---|---|---|---|
| Qwen-2.5-3B-Instruct (AWQ 4-bit) | open | QUEST-KG backbone (small) | Drive `/models/qwen25-3b/` |
| Mistral-7B-Instruct-v0.2 (AWQ 4-bit) | open | QUEST-KG backbone (medium, different family) | Drive `/models/mistral-7b/` |
| Qwen-2.5-7B-Instruct (AWQ 4-bit) | open | QUEST-KG backbone (medium) | Drive `/models/qwen25-7b/` |
| LLaMA-3.1-8B-Instruct (AWQ 4-bit) | **gated** (optional) | QUEST-KG backbone (large) | Drive `/models/llama31-8b/` |
| GPT-4o (API, added later) | API | Proprietary baseline | OpenAI API |

**Note on gated LLaMA:** request access at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct (usually auto-approved). The quantize script skips gated models gracefully if you don't have access; the other 3 backbones still cover the open-source LLM-grid story.

## Quick start

1. Clone:
   ```
   git clone https://github.com/AKSB0567/quest-kg-cikm2026.git
   cd quest-kg-cikm2026
   ```
2. Install deps locally (CUDA 12.6 on 1080 Ti):
   ```
   pip install -r requirements.txt
   ```
3. In Colab Pro: open `notebooks/setup_all.ipynb` and Run All. It performs every setup step in sequence: mount Drive, clone repo, install deps, verify CUDA, HF login, download all 5 datasets, quantize all 4 LLMs to 4-bit AWQ, smoke-test models, print summary. Every step is idempotent — re-running skips anything already done (via `.done` markers).
4. Optional: prep secrets for less friction. In Colab's secret manager (key icon in left sidebar), add `HF_TOKEN` (HuggingFace read token) and optionally `GH_TOKEN` (if the repo is private).

## Sprint timeline

See `paper/PLAN.md` for the day-by-day timeline (May 15 → May 23 submission).
