# QUEST-KG Methodology — locked spec for §3

This document is the **single source of truth** for the math, algorithms, and
operational definitions of QUEST-KG. The advisor turns this into §3 prose for
the CIKM submission. Nothing in the implementation, experiments, or paper text
should disagree with this spec — if a change is needed, edit this file first,
then propagate.

Status: **Phase 0 LOCKED on 2026-05-15.**

---

## 3.1 Overview

QUEST-KG performs trustworthy reasoning over a knowledge graph by treating
**edge admission** into a query-specific subgraph as the central decision.
Three tightly-integrated components implement this:

1. **Schema-aware Graph-RAG retrieval** — given a query, retrieve a *seed*
   subgraph by combining dense semantic similarity, ontology-aware predicate
   alignment, and bounded multi-hop expansion.
2. **Evidential message passing** — propagate node beliefs and per-node
   uncertainties through retrieved edges, with attention weights modulated
   by provenance reliability, schema consistency, semantic similarity, and
   uncertainty of source nodes.
3. **Neuro-symbolic reasoning + abstention** — score candidate reasoning
   paths under symbolic ontology constraints; commit to a prediction only
   when retained posterior probability mass exceeds threshold `p★` and
   entropy is below threshold `H★`, otherwise abstain.

Inputs: query `q`, global KG `G = (V, E, R, Π)` where `Π` carries
per-triple provenance metadata, optional textual evidence corpus `T`.
Outputs: prediction `ŷ`, retrieved subgraph `Gq`, per-node posterior belief
`b_i` and uncertainty `u_i`, top-scoring reasoning path `π*`, abstain flag.

---

## 3.2 Notation

| Symbol | Type | Role |
|---|---|---|
| `q` | string | input query |
| `G = (V, E, R, Π)` | tuple | global KG: entities, edges, relation types, provenance index |
| `Gq = (Vq, Eq, Rq)` | tuple | query-specific retrieved subgraph |
| `t = (s, r, o)` | triple | (subject, relation, object) |
| `π_t ∈ Π` | record | provenance for triple `t`: `{source_id, extraction_conf, edit_recency_days, source_trust}` |
| `q_emb ∈ R^d` | vector | query embedding (encoder fixed; d = 768 default) |
| `v_i ∈ R^d` | vector | entity embedding for node `v_i` |
| `h_i ∈ R^d` | vector | node representation after MP iteration `t` |
| `b_i ∈ [0,1]` | scalar | evidential belief at node `v_i` (Dirichlet expected probability) |
| `u_i ∈ [0,1]` | scalar | evidential uncertainty at node `v_i` (Dirichlet vacuity) |
| `α_i^c ∈ R₊` | scalar | Dirichlet pseudo-count for node `v_i`, class `c` |
| `s_ij^sem` | scalar | semantic similarity between query and edge (s_i, r, s_j) |
| `s_ij^prov` | scalar | provenance reliability of edge (s_i, r, s_j) |
| `s_ij^schema` | scalar | schema-consistency indicator for edge under query type |
| `φ_ij` | scalar | unnormalized attention logit for edge (i→j) |
| `α_ij ∈ [0,1]` | scalar | normalized attention weight for edge (i→j) |
| `W_r` | matrix | relation-specific transformation, `d × d` |
| `γ_1..γ_4` | scalars | attention weights, learned and constrained to `Σ γ_k = 1`, `γ_k ≥ 0` |
| `k` | int | retrieval hop budget; default `k = 2` (sweep `k ∈ {1,2,3,4}` in ablation) |
| `K_top` | int | top-K candidates per hop; default `K_top = 32` |
| `τ` | float | abstention threshold in `[0,1]`; swept in §4.6 |
| `p★` | float | posterior-mass admission threshold; default `0.05` |
| `H★` | float | entropy threshold for abstention; default `0.6` bits |
| `δ(π)` | bool | symbolic-constraint violation indicator for path `π` |
| `S(π)` | scalar | composite path score |
| `Π_q` | set | candidate reasoning paths for query `q` |

---

## 3.3 Schema-Aware Graph-RAG Retrieval

### Step 1: Anchor entity linking

Embed the query: `q_emb = Encode(q)` using a fixed sentence-transformer
(`all-MiniLM-L12-v2` or `e5-large`; pick one once and fix). Compute cosine
similarity to all entity embeddings:

  `s_sem(q, v_i) = (q_emb · v_i) / (||q_emb|| ||v_i||)`

Select the top-`K_top` candidates and apply **ontology-type filtering**: only
keep candidates whose type is compatible with the query's expected answer
type (computed by a lightweight intent classifier — for WebQSP/CWQ we use
the existing Freebase type system; for OrgAccess we use schema-derived types).

### Step 2: Bounded multi-hop expansion

Starting from anchor set `A`, expand outward by `k` hops. At each hop, for
each frontier node `v_i`, score candidate outgoing edges `(v_i, r, v_j)`:

  `EdgeScore(v_i, r, v_j; q) = γ_1·s_ij^sem + γ_2·s_ij^prov + γ_3·s_ij^schema`

Keep the top-`K_top` edges per frontier node; add their targets to the next
frontier. Stop at hop `k` or when no new edges score above a min-keep
threshold `ε_keep = 0.05` (these are pruned before MP).

The retrieved subgraph is `Gq = (Vq, Eq, Rq)`.

### Step 3: Operationalize the three scores

**Semantic** `s_ij^sem ∈ [0,1]`:
  Linearize the edge as text "<head_label> <relation_label> <tail_label>",
  embed with the same encoder used for `q_emb`, take cosine to `q_emb`,
  normalize to `[0,1]`.

**Provenance** `s_ij^prov ∈ [0,1]`:
  `s_ij^prov = sigmoid(α · π_t.extraction_conf + β · π_t.source_trust − γ · decay(π_t.edit_recency_days))`
  where `decay(d) = 1 − exp(−d / τ_decay)` with `τ_decay = 365` days, and
  `(α, β, γ) = (1.0, 0.7, 0.3)` (fixed defaults — these come from grid
  search on a 200-triple held-out set; documented in §4.5).

For datasets where source trust isn't available (WebQSP, CWQ from Freebase),
fix `source_trust = 1.0` for all triples and document this.

**Schema consistency** `s_ij^schema ∈ {0, 1}`:
  Binary indicator that returns `1` iff `(type(v_i), r, type(v_j))` is admissible
  under the ontology schema, else `0`. For WebQSP/CWQ: check against Freebase
  domain/range. For ICEWS18: check actor-event compatibility table. For
  OrgAccess: check Datalog predicate signatures.

---

## 3.4 Evidential Message Passing

Each node carries a Dirichlet posterior over a binary "supports the answer"
variable, parameterized by pseudo-counts `(α_i^pos, α_i^neg)`:

  `b_i = α_i^pos / (α_i^pos + α_i^neg + 2)`  (expected probability)
  `u_i = 2 / (α_i^pos + α_i^neg + 2)`        (Dirichlet vacuity)

Initialization: for anchor entities, `α_i^pos = 1 + κ·s_sem(q, v_i)`,
`α_i^neg = 1`; for non-anchors, `α_i^pos = α_i^neg = 1` (uniform prior).
Choose `κ = 4` (fixed default; documented in §4.5).

### Attention with provenance + schema + uncertainty

For each retained edge `(v_j → v_i)`, compute the unnormalized logit:

  `φ_ij = γ_1·s_ji^sem + γ_2·s_ji^prov + γ_3·s_ji^schema − γ_4·u_j`

where `u_j` is the **source node's uncertainty** (so high-uncertainty
neighbors get down-weighted). The four `γ_k` are learned with softmax
parameterization to enforce `Σ γ_k = 1, γ_k ≥ 0`.

Normalize over neighbors of `v_i`:

  `α_ij = softmax_j(φ_ij)`

### Belief aggregation

The message from neighbor `v_j` to `v_i` carries pseudo-count contributions:

  `Δα_i^pos += α_ij · b_j · w_r^pos`
  `Δα_i^neg += α_ij · (1 − b_j) · w_r^neg`

where `w_r^pos`, `w_r^neg` are learned per-relation scalars
(initialized to `1.0`).

After `T = k` iterations of MP (one per retrieval hop), the final
posterior at each node is `(b_i, u_i)` as defined above.

### Why this formulation

- **Evidence-conservative**: uncertain neighbors send weaker messages.
- **Provenance-aware**: low-trust edges contribute less mass.
- **Calibration-friendly**: Dirichlet posteriors give a principled uncertainty,
  not a post-hoc heuristic (closes T1 kFJG's calibration concern).

---

## 3.5 Neuro-Symbolic Reasoning + Abstention

### Path enumeration

`Π_q = { all paths in Gq from any anchor to any candidate answer entity, length ≤ k }`.
For multi-hop QA we limit to ≤32 candidate paths per query (best-first by
path-level evidence; see below). For entity alignment, candidate "paths" are
length-1 edges. For OrgAccess, each path corresponds to a derivation chain
of Datalog rules.

### Symbolic constraints `δ(π)`

`δ(π) = 1` iff path `π` violates *any* symbolic ontology rule applicable to
the task. Per-task constraint sets:

**WebQSP / CWQ (Freebase-based QA):**
- Domain/range check: every triple's (head_type, relation, tail_type) must be in the Freebase schema.
- Functional-relation cardinality: relations marked functional admit exactly one outgoing edge per subject.
- Type-of-answer match: terminal entity's type must be in the query's expected-answer type set.

**ICEWS18 (temporal events):**
- Temporal consistency: timestamps along the path must be monotonically non-decreasing.
- Actor-event compatibility: (actor_type, event_type) pairs must be in the ICEWS code book.

**OrgAccess (synthetic dynamic policy):**
Six Datalog rules (full list in appendix):
- `has_role(U, R, t) ∧ grants(R, Res) ∧ governed_by(Res, P) ∧ valid_in(P, ctx(t)) → access(U, Res, t)`
- `revoked_at(R, t') ∧ t' ≤ t → ¬has_role(U, R, t)` (revocation)
- ...etc.

If `δ(π) = 1`, the path is excluded from scoring.

### Path scoring

  `S(π) = exp( Σ_(i,j) ∈ edges(π) [ log α_ij + log w_r ] ) · Π_(v_i ∈ π) b_i`

Equivalently in log space:
  `log S(π) = Σ log α_ij + Σ log w_r + Σ log b_i`

### Answer-side rescoring (post-MP)

Empirically, path attention captures structural coherence but not "is the
endpoint actually the answer to the question". We multiply each surviving
path's score by a **query-conditioned tail-similarity term** before picking
the argmax:

  `S'(π) = S(π) · exp( λ_ans · cos(q_emb, v_tail(π)) ) · B_qr(π) · P_anchor(π) · P_meta(π)`

where:
- `λ_ans = 4`: tunable scalar; lifts paths whose tail is semantically close
  to the question. Disabled for ICEWS18 (numeric entity IDs have no
  meaningful encoder embedding).
- `B_qr(π) = 2` if the last relation's tokens align with a question-pattern
  hint (e.g. "where … from" -> {place_of_birth, location.…}), else 1.
- `P_anchor(π) = exp(-2 · overlap(tail_tokens, anchor_tokens))`: penalises
  candidate tails that are mostly anchor tokens (= "about the anchor" rather
  than the answer).
- `P_meta(π) = exp(-4)` when the last relation is a Freebase metadata
  relation (`type.object.type`, `common.topic.notable_types`, …); these
  produce generic type nodes ("Person", "Invention") not specific answers.

`λ_ans, B_qr, P_anchor, P_meta` are deterministic symbolic terms — no LLM
involved. They preserve the §3.4 evidential-MP and §3.5 abstention math
intact; they only re-rank surviving paths during answer extraction.

The predicted answer is the terminal entity of `π* = argmax S'(π)`.

### Optional LLM verbalization (QUEST-KG-LLM variant)

For open-domain QA where the answer surface form must be verbalized in
natural language, we offer an **optional hybrid** mode:

1. QUEST-KG retrieval + evidential MP + symbolic checking are unchanged.
2. The top-N (default N=8) surviving paths' triples are passed to an LLM
   along with up to 16 candidate answer entities (the paths' deduplicated
   non-MID, non-anchor endpoints).
3. The LLM picks one candidate by index from a numbered list. If the
   LLM output does not snap to a candidate, the symbolic prediction is kept
   (no degradation vs pure-symbolic).

The hybrid retains QUEST-KG's calibrated abstention (the LLM only fires
when QUEST-KG has not abstained). Reported in §4 as `QUEST-KG-LLM`.

### Abstention

Compute retained posterior mass across surviving paths:
  `M_q = Σ_{π ∈ Π_q, δ(π)=0} S(π) / Z` where `Z = Σ_(all paths) S(π)`

Compute entropy over candidate-answer distribution:
  `H_q = − Σ_a P(a | Gq) log P(a | Gq)`
  where `P(a | Gq) = Σ_{π → a} S(π) / Z`

**Abstain iff** `M_q < p★` or `H_q > H★`. Default `p★ = 0.05, H★ = 0.6 bits`.
Both are swept in §4.6 (R-C curve).

---

## 3.6 Training objective

  `L = L_task + λ_1 · L_cal + λ_2 · L_cons`

**Task loss** `L_task` (per task):
- WebQSP/CWQ: cross-entropy over candidate answer entities.
- ICEWS18: cross-entropy over candidate tail entities given (head, relation, timestamp).
- OrgAccess: binary CE over access decision.

**Calibration loss** `L_cal` (evidential learning, Sensoy et al. 2018):
  `L_cal = E_(b,u) [ KL( Dirichlet(α) || Dirichlet(1) ) · I[wrong prediction] ]`
This regularizes Dirichlet posteriors toward the uninformative prior when the
prediction is wrong, encouraging high uncertainty on errors.

**Consistency loss** `L_cons`:
  `L_cons = Σ_π δ(π) · |S(π)|`
Soft-penalizes paths that violate symbolic constraints (mass should drop on
violating paths even when the constraint check is hard-thresholded at inference).

Defaults: `λ_1 = 0.1`, `λ_2 = 0.5`. Tuned on validation.

---

## 3.7 Hyperparameters (lock these now; no churn during experiments)

| Param | Default | Sweep? |
|---|---|---|
| Embedding dim `d` | 768 | no |
| Sentence encoder | `intfloat/e5-large-v2` | no |
| Retrieval hops `k` | 2 | yes (§4.6: `{1,2,3,4}`) |
| Top-K per hop | 32 | no |
| MP iterations `T` | `k` | no |
| Edge keep threshold `ε_keep` | 0.05 | no |
| Provenance `(α, β, γ)` | `(1.0, 0.7, 0.3)` | no |
| Recency decay `τ_decay` | 365 days | no |
| Anchor prior boost `κ` | 4 | no |
| Abstention `p★` | 0.05 | yes (R-C curve) |
| Abstention `H★` | 0.6 bits | yes (R-C curve) |
| `(γ_1, γ_2, γ_3, γ_4)` | learned, init `(0.4, 0.3, 0.2, 0.1)` | learned |
| Per-relation `(w_r^pos, w_r^neg)` | learned, init 1.0 | learned |
| `λ_1, λ_2` | 0.1, 0.5 | tuned on val |
| Optimizer | AdamW, lr 2e-4, wd 0.01 | no |
| Batch size | 16 queries | no |
| Max epochs | 30 with early stop (val ECE plateau) | no |
| Random seeds | 3 (default), 5 for headline numbers | yes |

---

## 3.8 What this closes from prior reviewers

| Concern | How §3 closes it |
|---|---|
| T1 uPSH: "no standalone extractor eval" | Extractor is decoupled; §3.3 step 3 documents how triples come into the KG; §4 reports retrieval P/R + extractor P/R separately. |
| T1 uPSH: "robustness under KG gaps" | Edge-deletion robustness experiment uses `Gq` retrieval that gracefully shrinks; §4.7 reports accuracy/ECE vs deletion %. |
| T1 kFJG: "matched-condition baselines" | Same retrieval `Gq` plumbed into RAG / GraphRAG / ToG / our MP module — matched-condition Table 4. |
| T1 kFJG: "no design alternatives in ablation" | §4.6 ablates evidential MP vs Bayesian-GNN vs MC-dropout vs uniform attention. |
| T1 kFJG: "calibration + abstention under-analyzed" | §3.4 Dirichlet posteriors give principled uncertainty; §3.5 abstention rule is parameterized by `(p★, H★)`; §4.7 reports ECE, Brier, NLL, AURC, reliability diagrams. |
| T1 vJV5: "design choices not motivated" | Each `s_*` term has an operational definition; `γ_k` learned with softmax constraint; symbolic-constraint set listed per task. |
| T1 vJV5: "extraction error propagation" | Edge-keep threshold `ε_keep` prunes low-confidence triples before MP; §4.6 reports per-stage error decomposition. |
| T2 dp8z: "calibrated uncertainty undefined" | §3.4 Dirichlet posterior definition with closed-form `b_i, u_i`. |
| T2 dp8z: "no notation table" | §3.2. |
| T2 HRzC: "KG construction + leakage" | §3.3 step 3 separates "extracted from training only" provenance from inference; §4.5 protocol section. |
| T2 HRzC, AC: "no case studies" | §3.5 path enumeration + scoring + abstention gives a 4-tuple `(Gq, {(b_i,u_i)}, π*, abstain?)` per query — case study figure shows this directly. |
| T2 8dzU: "KG structure not illustrated" | Architecture figure (TikZ source separate file). |

---

## Open decisions (one is still live)

- **Encoder choice**: between `all-MiniLM-L12-v2` (faster, 384d) and
  `intfloat/e5-large-v2` (slower, 1024d). I'm defaulting to `e5-large-v2`
  for accuracy. If T-time becomes a bottleneck on Day 3, we drop to MiniLM
  and re-run retrieval. Note this in §4.5 if changed.
