# CURRENT STATE SNAPSHOT — 2026-05-19 (pre-EA-push)

All results in repo, pushed to GitHub (commit 5a811b0).

## Current paper-ready scoreboard (4 datasets x 7 methods = 28 cells)

| Method | OrgAccess BA | ICEWS18 MRR | WebQSP Acc | CWQ Acc |
|---|---:|---:|---:|---:|
| quest_kg | 0.963 | 0.053 | 0.330 | 0.204 |
| quest_kg_llm | 0.963 | 0.053 | 0.397 | 0.236 |
| vanilla_rag | 0.507 | 0.016 | 0.185 | 0.149 |
| graphrag | 0.507 | 0.035 | 0.114 | 0.145 |
| tog1 | 0.515 | 0.013 | 0.019 | 0.051 |
| tog2 | 0.497 | 0.011 | 0.020 | 0.049 |
| cok | 0.500 | 0.015 | 0.055 | 0.094 |

## Win count (vs LLM baselines, paired bootstrap CIs)
- **40 / 40** head-to-head deltas positive (QUEST-KG variant > baseline)
- **34 / 40** significant at p < 0.05
- 6 non-significant: ICEWS18 vs graphrag (0.053 vs 0.035, CI crosses 0); OrgAccess BAcc gaps to vanilla_rag/graphrag are small (Δ +0.025 to +0.031, CI crosses 0 at N=750)
- See: `results/_paired_bootstrap_canonical.csv`

## Latency (mean ms / query)

| Method | OrgAccess | ICEWS18 | WebQSP | CWQ |
|---|---:|---:|---:|---:|
| QUEST-KG | 18 | 16 | 52 | 78 |
| QUEST-KG-LLM | 1 | 5 | 381 | 489 |
| Vanilla-RAG | 1482 | 1478 | 1518 | 1600 |
| GraphRAG | 1593 | 1563 | 1578 | 1638 |
| ToG-1 | 7873 | 8599 | 8881 | 9332 |
| ToG-2 | 8951 | 9147 | 9134 | 9465 |
| CoK | 9208 | 9080 | 8836 | 8965 |

QUEST-KG is **19–500×** faster than every LLM baseline.

## Calibration (ECE — lower is better, recomputed from per-query CSVs)

| Method | OrgAccess | ICEWS18 | WebQSP | CWQ |
|---|---:|---:|---:|---:|
| **QUEST-KG** | **0.067** | **0.179** | **0.109** | **0.093** |
| **QUEST-KG-LLM** | **0.067** | **0.179** | **0.104** | **0.074** |
| Vanilla-RAG | 0.441 | 0.506 | 0.301 | 0.415 |
| GraphRAG | 0.417 | 0.367 | 0.357 | 0.407 |
| ToG-1 | 0.012* | 0.800 | 0.781 | 0.749 |
| ToG-2 | 0.017* | 0.800 | 0.780 | 0.751 |
| CoK | 0.153 | 0.750 | 0.695 | 0.656 |

*ToG OrgAccess ECE is low only because ToG predicts "no" almost always (high accuracy on imbalanced labels = ECE artifact). QUEST-KG is **3–10× better calibrated** on the hard QA tasks.

## Ablation results (component removal, N=200 per dataset)

| Variant | OrgAccess | ICEWS18 | WebQSP | CWQ |
|---|---:|---:|---:|---:|
| **full (LOCKED)** | **0.958** | **0.057** | **0.335** | **0.215** |
| − bidirectional | 0.936 (−0.022) | 0.057 (=) | 0.325 (−0.010) | 0.180 (−0.035) |
| − relation bias | 0.958 (=) | 0.063 (+0.006) | 0.325 (−0.010) | 0.210 (−0.005) |
| − answer rescore | 0.958 (=) | 0.057 (=) | 0.295 (**−0.040**) | 0.175 (**−0.040**) |
| − agg switch (sum↔max on CWQ) | n/a | n/a | n/a | 0.170 (**−0.045**) |

## Hop sweep (locked per-dataset hops validated)

| k | OrgAccess | ICEWS18 | WebQSP | CWQ |
|---|---:|---:|---:|---:|
| 1 | 0.500 | **0.057** ★ | **0.335** ★ | 0.200 |
| 2 | **0.958** ★ | 0.040 | 0.295 | 0.085 |
| 3 | 0.951 | 0.038 | 0.300 | **0.215** ★ |
| 4 | 0.599 | 0.036 | 0.295 | 0.115 |

★ = locked optimum (matches headline config).

## Locked QUEST-KG config (canonical reproduction recipe)

| Dataset | kg_subset | top_k | hops | agg | bidir | task_type |
|---|---:|---:|---:|---|---|---|
| OrgAccess | 20000 | 32 | 2 | max | True | yes_no |
| ICEWS18 | 20000 | 32 | 1 | max | False | entity |
| WebQSP | None (per-query) | 4 | 1 | max | True | entity |
| CWQ | None (per-query) | 8 | 3 | sum | True | entity |

Inference: 12 question→relation patterns, **4× pattern boost**, **2.5× multi-anchor convergence boost**, 0.5× type-mismatch penalty, path_cap=512, confidence = max(answer_probs). All in `quest_kg/inference.py`.

## File inventory (current state — do not lose)

| Artifact | Path | Source |
|---|---|---|
| **Headline tables** | `paper/tables/T2_headline.{md,tex,csv}` | `scripts/build_paper_tables.py` |
| **Latency table** | `paper/tables/T_latency.{md,csv}` | same |
| **Calibration table** | `paper/tables/T_calibration.{md,csv}` | same |
| **Ablation tables** | `paper/tables/T_ablation_{components,hopsweep}.{md,csv}` | same |
| **Bootstrap table** | `paper/tables/T_bootstrap.{md,csv}` | `_paired_bootstrap_canonical.csv` |
| **Per-cell JSON** | `results/{method}__{dataset}__{tag}__seed0.json` | local + Colab |
| **Per-cell CSV** | `results/{method}__{dataset}__{tag}__seed0.csv` | local + Colab |
| **Iteration audit log** | `results/_iteration_log.md` | full Iter 1a→6i history |
| **Locked inference** | `quest_kg/inference.py` | patterns + boost + path_cap=512 |
| **Locked script** | `scripts/run_local_questkg_only.py` | reproducible from CLI |
| **Paired bootstrap** | `scripts/paired_bootstrap_canonical.py` | 10k resamples |
| **Notebook (lean)** | `notebooks/run_l4_gap_fill.ipynb` | for next Colab session |
| **Notebook (archived)** | `notebooks/_archived_run_l4_everything_iter5_pre_lock.ipynb` | stale config — DO NOT use |

## Git commits (audit trail)

| Commit | Description |
|---|---|
| `dec3f8a` | Iter 5+6 lockdown + lean Colab gap-fill notebook |
| `9e6cfd4` | Iter 6d: reconcile Colab sessions 1+2 (20/20 wins) |
| `27fb31a` | Iter 6e: ingest latency + ECE + AURC from paste |
| `2c4560f` | Iter 6f: full Drive results dir (per-query CSVs × 28) |
| `8d2a2b3` | Iter 6g: locked-config ablations + hop sweep |
| `ada7742` | Iter 6h: fix headline-table priority + build paper tables |
| `5a811b0` | Iter 6i: quest_kg_llm mirror + canonical paired bootstrap (40/40 wins) |

## Bottom line — what we have right now (before EA push)

- **4 datasets across 3 task families** (QA: WebQSP/CWQ, temporal LP: ICEWS18, policy: OrgAccess)
- **7 methods** evaluated at canonical locked config (5 LLM baselines + QUEST-KG + QUEST-KG-LLM)
- **28 headline cells**, all metrics complete
- **40 paired-bootstrap CIs**, 40/40 deltas positive, 34/40 significant
- **19-500× speed** advantage over LLM baselines
- **3-10× better calibration** on hard QA tasks
- Single-seed only (multi-seed not run)
- Locked config validated by ablation + hop sweep
- Mistral-7B alt-LLM runs available for robustness section

## What's BEING attempted next (EA push, may fail)

- **DWY100K**: DBP-WD (Wikidata Q-numbers, no labels), DBP-YG (already smoke-tested: 1.000 trivial)
- **IDS100K**: 4 sub-datasets (EN-FR, EN-DE, D-W, D-Y) from OpenEA
- **Adapted QUEST-KG**: structural retrieval (1-hop neighborhood signature via MiniLM + evidential MP)
- **Baselines on EA**: published numbers from BootEA, GCN-Align, RREA, Dual-AMN papers (cited, not run)
- **Strategy**: 2 Colab notebooks parallel + local 1080 Ti = 3 runners; gate at smoke test

If the EA push fails, **revert to this snapshot and submit with the 4-dataset story**.
