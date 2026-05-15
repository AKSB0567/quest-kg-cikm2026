# Sprint plan — QUEST-KG (CIKM 2026)

Deadline: **2026-05-23**. Today: 2026-05-15.

## Daily milestones (gate: each day's deliverables must exist before moving on)

| Day | Date | Deliverable | Gate |
|---|---|---|---|
| 1 | Thu eve May 15 + Fri May 16 | §3 method spec, Algorithm 1, notation table, architecture figure (vector), KG-construction & leakage protocol, code skeleton | §3 self-contained, paper-ready math |
| 2 | Sat May 17 | Core impl on 1080 Ti: retrieval, evidential MP, symbolic checker, abstention; smoke test on 100 examples/dataset | E2E pipeline returns answers |
| 3 | Sun May 18 | Baselines reproduced under matched conditions (EA + QA + policy); single eval harness | Each baseline within ±1% of published |
| 4 | Mon May 19 | Headline runs Table 2, matched-condition Table 4, retrieval/extractor P/R Tables A1/A2 | No TBDs in main tables |
| 5 | Tue May 20 | Ablations Table 3 (component-removal + design-alternative), hop sweep, threshold sweep, error decomposition, ECE/Brier/NLL, reliability + AURC plots, latency stack, edge-deletion + noise robustness | All claims in §4 backed by numbers |
| 6 | Wed May 21 | Schema-drift, temporal robustness, case studies (port T3 Fig 2), worked example table, failure-mode taxonomy, paired bootstrap significance | All figures rendered |
| 7 | Thu May 22 | Paper writing (advisor leads), self-review pass as each prior reviewer | Compiled PDF ≤9 pages, no broken refs |
| 8 | Fri May 23 | Final polish, anonymous repo verify, submit | Submission confirmation screenshot |

## Hard rules

- Every claim points to a numeric cell. No "preliminary" or "promising" without numbers.
- All experiments use CUDA on GPU (no CPU-only runs).
- All long-running experiments use `ResumableRun` for checkpoint/resume.
- Daily 8 pm checkpoint: if a gate failed, decide cuts NOW, not tomorrow.
- No new ideas after Day 4.

## Risk register

| Risk | Trigger | Fallback |
|---|---|---|
| Baseline X won't reproduce | Day 3 evening | Cite published, exclude from matched-condition table with footnote |
| QUEST-KG loses to GraphRAG on accuracy | Day 4 morning | Pivot to calibration + robustness as primary win |
| Colab session blast radius | Anywhere | Resume from checkpoint; pay-as-you-go A100 unit boost |
| GPT-4o rate limits | Day 4 PM | Use only one proprietary backbone, label clearly |
| ICEWS18 too heavy for sprint | Day 6 | Move temporal experiments to appendix |
| Paper >9 pages | Day 6 eve | Push posterior diagnostics to appendix, drop §5.5 future directions |

## Reviewer-concern coverage matrix (which prior pain point each phase closes)

See `paper/REVIEW_COVERAGE.md` for the table of {prior reviewer concern → which Phase/Table addresses it}.
