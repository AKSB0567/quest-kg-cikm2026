# Reviewer-concern coverage matrix

Maps every concern raised across Trials 1 (ICLR 2026) and 2 (ACL ARR Jan 2026) to the
CIKM phase/artifact that closes it.

| # | Concern | Source reviewer | Closed by |
|---|---|---|---|
| 1 | No standalone triple-extractor evaluation (P/R) | T1 uPSH | Table A2 (Phase 1B Step 1.10) |
| 2 | No robustness study under KG completeness gaps | T1 uPSH | Fig 9a edge-deletion (Phase 5) |
| 3 | Unclear backend LLM at each stage | T1 uPSH, T2 HRzC | Table of LLM-per-stage mapping in §4.5 |
| 4 | No head-to-head GraphRAG/ToG-2/CoK on shared corpus | T1 kFJG | Table 4 matched-condition (Phase 3 Step 2.4) |
| 5 | Only medical results, no generalization | T1 kFJG | 6 datasets across 3 task families |
| 6 | Ablations are component-removal only, no design alternatives | T1 kFJG | Table 3 rows 6-9 design alternatives (Phase 4 Step 2.6) |
| 7 | Dual-grounding calibration + abstention under-analyzed | T1 kFJG | Table 5 (ECE/Brier/NLL) + Fig 7 reliability + Fig 8 risk-coverage (Phase 5 Steps 3.1-3.3) |
| 8 | Missing related work (KG-Rank) | T1 vJV5 | §2 rewrite (Phase 7 Step 5.3) |
| 9 | Model too complex, design choices not motivated | T1 vJV5 | §3.1.1 design rationale + ablation table |
| 10 | Extraction errors propagation unclear | T1 vJV5 | Fig 6 error decomposition (Phase 4 Step 2.9) |
| 11 | Figures blurry / AI-plotted | T2 ALL | Vector PDF architecture Fig 1 (Phase 0 Step 0.8) |
| 12 | Experimental config unclear | T2 dp8z, HRzC | Notation table + §4.5 implementation details + appendix hyperparams |
| 13 | "Calibrated uncertainty" not defined | T2 dp8z | §3.5 evidential Dirichlet formal definition |
| 14 | No notation table | T2 dp8z | End of §3.1 (Phase 0 Step 0.3) |
| 15 | Table 3 caption/results mismatch | T2 dp8z | All tables sanity-checked against captions in Phase 8 |
| 16 | No ablations actually delivered (T2 promised) | T2 AC | Table 3 with full data (Phase 4) |
| 17 | Limited baselines (only 3 defenses) | T2 AC | 15+ baselines across 3 task families (Phase 2) |
| 18 | No case studies | T2 AC, HRzC, 8dzU | Fig 3 (4 panels) + Table 7 worked examples (Phase 6 Steps 4.1-4.2) |
| 19 | Baseline categorization questionable (SCAV miscategorized) | T2 HRzC | Each baseline correctly classified in §4.3 with cite to original paper |
| 20 | ASR ~0% claim not explained | T2 HRzC | n/a (different task in CIKM) but: provide mechanism-level reasoning in §5 |
| 21 | KG construction + leakage concern | T2 HRzC | §3.2.1 dedicated KG protocol subsection (Phase 0 Step 0.5) |
| 22 | Too much jargon | T2 8dzU | Notation table + concise §3 prose |
| 23 | KG structure not illustrated | T2 8dzU | Fig 3 reasoning-graph panels (Phase 6 Step 4.1) |
| 24 | Reproducibility concerns | T2 8dzU | Anonymous repo + appendix hyperparams + reproducibility statement |

## Anticipated CIKM-fresh concerns to pre-empt

| # | Anticipated concern | Pre-emptive coverage |
|---|---|---|
| A1 | "Why CIKM and not ACL/NeurIPS?" | Lead with enterprise/dynamic KG management story (OrgAccess + ICEWS18) |
| A2 | Multimodal/web-scale missing | Explicit scope statement in §1 + acknowledge in §5.4 limitations |
| A3 | Real-world deployment evidence | Latency + cost table (Fig 10 + Table 6) + scalability paragraph in §5.3 |
| A4 | Statistical significance | Paired bootstrap + 95% CIs in Table 2 footnote (Phase 7 Step 5.1) |
| A5 | Open-source release | Anonymous repo + plan-to-release statement |
