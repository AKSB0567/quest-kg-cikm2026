"""End-to-end QUEST-KG inference pipeline (Algorithm 1).

Wires together:
  1. SchemaAwareRetriever  -> Subgraph Gq
  2. evidential_mp_numpy   -> per-node (b_i, u_i), attention weights alpha_ij
  3. Per-task SymbolicChecker -> mark paths with delta(pi)
  4. Path scoring          -> S(pi)
  5. should_abstain        -> (p*, H*) abstention rule

Defaults are tuned to be **permissive** at inference (low p*, high H*) so we
get predictions out and let calibration analysis decide where to draw the line.
"""
from __future__ import annotations

import math
import re

import numpy as np


# Freebase MID/GID patterns (m.02_286, g.1257kpk96) - opaque identifiers that
# should never be the final answer when a surface label is available.
_MID_RE = re.compile(r"^[mg]\.[0-9a-z_]+$")


def _is_opaque_identifier(s: str | None) -> bool:
    if not s:
        return False
    return bool(_MID_RE.match(s.strip()))


# Freebase metadata-relation fragments. Paths whose last edge uses one of these
# relations lead to generic type/category nodes ("Invention", "Person",
# "Building"), not specific answer entities. Penalise during answer extraction.
_METADATA_REL_FRAGMENTS = frozenset({
    "type.object.type", "common.topic.notable_types",
    "kg.object_profile", "type.type.instance",
    "type.namespace.keys", "common.topic.alias",
    "common.topic.image", "freebase.object_profile",
    "type.object.name", "type.object.key",
    "common.notable_for", "common.topic.notable_for",
})


def _is_metadata_relation(r: str) -> bool:
    if not r:
        return False
    r_clean = r.lstrip("~").lower()
    return any(frag in r_clean for frag in _METADATA_REL_FRAGMENTS)


# Question-pattern -> expected relation keywords. When the question contains
# any pattern token, paths whose last relation contains any expected token get
# a multiplicative bonus during answer extraction. Symbolic, deterministic.
_QUESTION_RELATION_HINTS: list[tuple[set[str], set[str]]] = [
    # "where ... from / born / live": location/place/birth relations
    ({"where", "from", "born", "birthplace", "live", "located", "located_in"},
     {"place", "location", "birth", "born", "city", "country", "state", "region",
      "nationality", "containedby", "place_lived"}),
    # "who plays / played / acted": actor/person relations
    ({"plays", "played", "actor", "actress", "starring", "cast"},
     {"actor", "cast", "starring", "performance", "person", "character", "role"}),
    # "when ... born / die / start": date/time relations
    ({"when", "year", "date", "born", "died", "start", "started", "ended"},
     {"date", "year", "time", "birth", "death", "start", "end", "founding"}),
    # "what language / speak": language relations
    ({"language", "languages", "speak", "spoken"},
     {"language", "spoken", "official_language"}),
    # "what religion": religion relations
    ({"religion", "religious", "worship", "believe", "religions"},
     {"religion", "denomination", "faith"}),
    # "currency / money": currency relations
    ({"currency", "money", "dollar"},
     {"currency", "monetary"}),
    # "capital ... of": capital relations
    ({"capital"},
     {"capital", "administrative", "seat"}),
    # CWQ Iter 5e: "what country / nation": country/containedby relations
    ({"country", "nation", "countries", "nations", "land"},
     {"country", "nationality", "containedby", "located", "contains", "borders",
      "in_country", "located_in"}),
    # CWQ Iter 5e: "what college / university / school / educational institution"
    ({"college", "university", "school", "educational", "institution", "alma"},
     {"education", "school", "university", "college", "graduated", "alma",
      "institution", "campuses"}),
    # CWQ Iter 5e: "who/which man/woman/leader/ruler": person/leader relations
    ({"who", "man", "woman", "leader", "ruler", "president", "minister",
      "politician", "head", "official"},
     {"leader", "person", "president", "minister", "head_of_state", "head",
      "spouse", "official", "child", "parent", "founder"}),
    # CWQ Iter 5e: "what team / sports / franchise / club"
    ({"team", "teams", "sports", "franchise", "club", "league"},
     {"team", "sports", "franchise", "club", "league", "roster", "championship",
      "fielded"}),
    # CWQ Iter 5e: "what movie / film": film relations
    ({"movie", "film", "films", "movies"},
     {"film", "movie", "directed", "cinematography", "production"}),
    # CWQ Iter 5e: "what stadium / venue / arena": venue relations
    ({"stadium", "venue", "arena", "field", "ground"},
     {"stadium", "venue", "arena", "home_arena", "home_stadium", "ground"}),
    # Iter 5n: tested adding spouse/governor/parent/child/voice/attended patterns —
    # WebQSP unchanged (0.366), CWQ regressed -0.002. Reverted.
]


def _question_relation_boost(question_tokens: set[str], relation_tokens: set[str]) -> float:
    """Multiplicative path-score factor based on question/relation alignment.
    Returns 1.0 (no effect) when no pattern matches, otherwise >1 to lift
    candidates whose last relation matches the expected answer type.

    Iter 5i-k: 2.0 -> 3.0 -> 4.0 -> 5.0. Plateau at 4.0 (CWQ 0.178).
    Iter 5l: lock at 4.0 + add type-mismatch penalty. When question has
    pattern hits but path's last relation matches NO r_pattern, multiply
    by 0.5x (penalize wrong-type tail relations).
    """
    boost = 1.0
    any_q_match = False
    any_r_match = False
    for q_pattern, r_pattern in _QUESTION_RELATION_HINTS:
        q_hit = bool(question_tokens & q_pattern)
        r_hit = bool(relation_tokens & r_pattern)
        if q_hit:
            any_q_match = True
        if r_hit:
            any_r_match = True
        if q_hit and r_hit:
            boost *= 4.0
    # Hard penalty: question signals type X, path tail doesn't match any X.
    if any_q_match and not any_r_match:
        boost *= 0.5
    return boost

from quest_kg.abstention.posterior import should_abstain
from quest_kg.core.types import Path, PredictResult, Subgraph, Triple
from quest_kg.mp.evidential import evidential_mp_numpy
from quest_kg.retrieval.schema_aware import SchemaAwareRetriever


def _enumerate_paths(sg: Subgraph, max_length: int, cap: int = 256,
                     bidirectional: bool = True) -> list[Path]:
    """Enumerate paths from any anchor, up to `max_length` hops; cap total at `cap`.

    When `bidirectional=True`, walks BOTH forward (s -> o) and reverse (o -> s)
    edges so anchors that are referenced as the tail of a triple (e.g.
    `(Jamaican Creole | main_country | Jamaica)` with anchor=Jamaica) can still
    reach their semantic neighbours. For reverse traversal we synthesise a
    virtual `Triple` whose `.s` is the side we're at and `.o` is the side we're
    walking to, with relation prefixed by `~`. This keeps `path.tail` semantically
    meaningful (= endpoint farthest from anchor) for downstream answer extraction.

    When `bidirectional=False` (e.g. ICEWS18), only walks forward edges. The
    directional event graph has no meaningful reverse semantics.
    """
    paths: list[Path] = []
    out_idx: dict[str, list[Triple]] = {}
    in_idx: dict[str, list[Triple]] = {}
    for t in sg.triples:
        out_idx.setdefault(t.s, []).append(t)
        in_idx.setdefault(t.o, []).append(t)

    def _expand(node: str) -> list[Triple]:
        """All edges (forward + reverse) at `node`. Reverse edges are
        synthesised so the virtual triple's `.s = node` and `.o = the other end`."""
        out = list(out_idx.get(node, []))
        if not bidirectional:
            return out
        for t in in_idx.get(node, []):
            # Reverse: at o=node, going back to s. Virtual triple's tail is t.s.
            out.append(Triple(
                s=node, r=f"~{t.r}", o=t.s,
                s_type=getattr(t, "o_type", ""), o_type=getattr(t, "s_type", ""),
                prov=getattr(t, "prov", None),
                timestamp=getattr(t, "timestamp", None),
            ))
        return out

    queue: list[list[Triple]] = []
    for a in sg.anchors:
        for step in _expand(a):
            queue.append([step])

    while queue:
        cur = queue.pop()
        paths.append(Path(triples=list(cur)))
        if len(paths) >= cap:
            break
        if len(cur) >= max_length:
            continue
        for step in _expand(cur[-1].o):
            queue.append(cur + [step])
    return paths


def _score_path(p: Path, sg: Subgraph) -> float:
    """Path score S(pi) = exp(sum log alpha + sum log belief).

    If attention or belief is missing for some node/edge, fall back to 0.5
    (neutral) rather than dropping the term entirely — keeps short paths
    competitive with longer ones.
    """
    log_s = 0.0
    for t in p.triples:
        a = sg.attention.get((t.s, t.o))
        if a is None or a <= 0:
            a = 1.0 / max(len(sg.triples), 1)
        log_s += math.log(max(a, 1e-12))
    for node in p.node_seq():
        b = sg.node_states[node].belief if node in sg.node_states else 0.5
        log_s += math.log(max(b, 1e-12))
    # Normalize by path length so 1-hop and 2-hop paths are comparable
    log_s /= max(p.length, 1)
    return math.exp(log_s)


class QuestKG:
    """End-to-end inference glue (Algorithm 1).

    Permissive defaults so we don't abstain on everything by accident:
      p_star = 0.0       -> never abstain on low retained mass
      h_star = 999.0     -> never abstain on entropy
    Calibration analysis (§4.6) will sweep these for the R-C curve.

    `task_type` selects the answer-extraction strategy:
      - "entity"    : default. Returns the tail of the top scoring valid path.
                      (WebQSP, CWQ, ICEWS18 tail prediction.)
      - "yes_no"    : Returns "1" if a valid path connects any anchor to the
                      designated target entity (passed via query_meta['target']),
                      else "0". (OrgAccess.)
    """

    def __init__(
        self,
        retriever: SchemaAwareRetriever,
        symbolic_checker,
        p_star: float = 0.0,
        h_star: float = 999.0,
        mp_gammas: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
        kappa: float = 4.0,
        max_path_length: int | None = None,
        path_cap: int = 512,
        task_type: str = "entity",
        answer_rescoring: bool = True,
        answer_rescoring_lambda: float = 4.0,
        answer_aggregation: str = "max",
        llm=None,
        llm_rerank_top_n: int = 8,
    ):
        self.retriever = retriever
        self.symbolic_checker = symbolic_checker
        self.p_star = p_star
        self.h_star = h_star
        self.mp_gammas = mp_gammas
        self.kappa = kappa
        self.max_path_length = max_path_length or retriever.k
        self.path_cap = path_cap
        assert task_type in ("entity", "yes_no")
        self.task_type = task_type
        # Answer-side cosine rescoring helps surface-form QA (WebQSP, CWQ) but
        # hurts numeric-ID link prediction (ICEWS18) where the "entity" is an
        # opaque integer with no semantic embedding.
        self.answer_rescoring = bool(answer_rescoring)
        self.answer_rescoring_lambda = float(answer_rescoring_lambda)
        # "max" picks the single best path's tail; "sum" path-votes (aggregate
        # scores across paths reaching the same tail). max wins when retrieval
        # is k=1 (each tail uniquely reached); sum wins at k>=2 where many
        # paths reach the same node and aggregation reflects evidence count.
        assert answer_aggregation in ("max", "sum")
        self.answer_aggregation = answer_aggregation
        # Optional LLM verbalization for QA: when set, QUEST-KG keeps its
        # retrieval + symbolic reasoning, then asks the LLM to pick the answer
        # from the top-N candidate paths. The "QUEST-KG-LLM" hybrid variant.
        # Default None preserves pure-symbolic behaviour for OrgAccess/ICEWS18.
        self.llm = llm
        self.llm_rerank_top_n = int(llm_rerank_top_n)

    def predict(
        self,
        query: str,
        expected_types: set[str] | None = None,
        query_meta: dict | None = None,
    ) -> PredictResult:
        # ---- Stage 1: retrieval -------------------------------------------------
        explicit_anchors = None
        if query_meta:
            # Combine user/resource/q_entity/head/etc. into anchor set.
            cand: list[str] = []
            for k in ("user", "resource", "head", "target"):
                v = query_meta.get(k)
                if isinstance(v, str):
                    cand.append(v)
                elif isinstance(v, (list, tuple, set)):
                    cand.extend(x for x in v if isinstance(x, str))
            if "q_entity" in query_meta and query_meta["q_entity"]:
                qe = query_meta["q_entity"]
                if isinstance(qe, str):
                    cand.append(qe)
                else:
                    cand.extend(x for x in qe if isinstance(x, str))
            explicit_anchors = cand or None
        # Per-query graph (CWQ/WebQSP): restrict retrieval candidates to the
        # query's local HF-dataset graph. Each rmanluo/RoG-{cwq,webqsp} example
        # ships with its own subgraph, which is what graphrag uses. Without
        # this restriction, our retrieval searches the 2.3M-triple global KG
        # truncated to first 100k, which loses 95%+ of query-relevant triples.
        restrict_triples = None
        if query_meta and query_meta.get("graph_triple_idx"):
            restrict_triples = query_meta["graph_triple_idx"]
        sg = self.retriever.retrieve(
            query, expected_types=expected_types, explicit_anchors=explicit_anchors,
            restrict_triples=restrict_triples,
        )

        # Anchor similarities for MP init (use cached entity embeddings)
        anchor_sims: dict[str, float] = {}
        if sg.anchors:
            try:
                q_emb = np.asarray(self.retriever.encoder(query), dtype=np.float32)
                q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
                for a in sg.anchors:
                    if (self.retriever._entity_emb is not None
                            and a in self.retriever._entity_idx):
                        e = self.retriever._entity_emb[self.retriever._entity_idx[a]]
                        e_norm = e / (np.linalg.norm(e) + 1e-12)
                        anchor_sims[a] = float(np.dot(q_norm, e_norm))
                    else:
                        anchor_sims[a] = 1.0
            except Exception:
                anchor_sims = {a: 1.0 for a in sg.anchors}

        # ---- Stage 2: evidential MP --------------------------------------------
        if sg.triples:
            evidential_mp_numpy(
                sg,
                anchor_sims=anchor_sims,
                gammas=self.mp_gammas,
                kappa=self.kappa,
                n_iters=self.retriever.k,
            )
        else:
            # No edges retrieved -> assign neutral belief to anchors
            from quest_kg.core.types import NodeState
            for n in sg.nodes:
                sg.node_states[n] = NodeState(alpha_pos=1.0, alpha_neg=1.0)

        # ---- Stage 3: paths + symbolic + abstention ----------------------------
        bidir_paths = getattr(self.retriever, "bidirectional", True)
        candidate_paths = _enumerate_paths(
            sg, max_length=self.max_path_length, cap=self.path_cap,
            bidirectional=bidir_paths,
        )
        for p in candidate_paths:
            try:
                p.violates_symbolic = bool(self.symbolic_checker.violates(p))
            except Exception:
                p.violates_symbolic = False
            p.score = _score_path(p, sg)

        abstained, M_q, H_q, probs = should_abstain(
            candidate_paths, p_star=self.p_star, h_star=self.h_star
        )

        valid = [p for p in candidate_paths if not p.violates_symbolic]

        # Answer-side re-scoring for entity QA: combine the path score with
        # cosine similarity between the tail surface form and the query, plus
        # a question-pattern -> relation-keyword boost. Penalise candidate
        # tails that overlap heavily with the anchor surface form (they are
        # usually "about the anchor" rather than the answer to the question).
        # All signals are symbolic + deterministic; no LLM at inference.
        if self.task_type == "entity" and self.answer_rescoring and valid:
            try:
                from quest_kg.retrieval.schema_aware import _content_tokens, _relation_tokens
            except Exception:
                _content_tokens = None
                _relation_tokens = None
            q_tokens = _content_tokens(query) if _content_tokens else set()
            anchor_tokens: set[str] = set()
            if _content_tokens:
                for a in sg.anchors:
                    anchor_tokens |= _content_tokens(a)
            try:
                q_emb_a = np.asarray(self.retriever.encoder(query), dtype=np.float32)
                q_norm_a = q_emb_a / (np.linalg.norm(q_emb_a) + 1e-12)
                ent_emb = self.retriever._entity_emb
                ent_idx = self.retriever._entity_idx
                for p in valid:
                    tail = p.tail
                    if ent_emb is not None and tail in ent_idx:
                        v = ent_emb[ent_idx[tail]]
                        v_norm = v / (np.linalg.norm(v) + 1e-12)
                        s_ans = float(np.dot(q_norm_a, v_norm))
                    else:
                        s_ans = 0.0
                    p.score = float(p.score) * math.exp(self.answer_rescoring_lambda * s_ans)
                    if q_tokens and _relation_tokens and p.triples:
                        last_rel = p.triples[-1].r or ""
                        rel_clean = last_rel.lstrip("~")
                        r_tokens = _relation_tokens(rel_clean)
                        p.score = float(p.score) * _question_relation_boost(q_tokens, r_tokens)
                    # Metadata-relation penalty: paths ending in type/notable_for/
                    # alias relations lead to type/category nodes ("Invention",
                    # "Person"), not specific answer entities. Heavy penalty.
                    if p.triples and _is_metadata_relation(p.triples[-1].r or ""):
                        p.score = float(p.score) * math.exp(-4.0)
                    # Anchor-overlap penalty: if the tail's content tokens are
                    # mostly anchor tokens, it's likely a sentence/description
                    # node "about the anchor" rather than the answer.
                    # Hard penalty when anchor tokens are a subset of tail tokens:
                    # those are "derived" forms (SS X, X Museum, X's biography)
                    # that almost never answer the question about X itself.
                    if _content_tokens and anchor_tokens and tail:
                        tail_tokens = _content_tokens(tail)
                        if tail_tokens:
                            overlap = len(tail_tokens & anchor_tokens) / len(tail_tokens)
                            p.score = float(p.score) * math.exp(-2.0 * overlap)
                            if anchor_tokens.issubset(tail_tokens):
                                p.score = float(p.score) * math.exp(-3.0)
            except Exception:
                pass

        # Multi-anchor convergence boost. CWQ-style "what X that Y" questions
        # have >=2 anchors; the correct answer is reachable from ALL of them.
        # Boost paths whose tail is also reached from a different anchor.
        # No-op when only one anchor (WebQSP simple questions).
        if self.task_type == "entity" and len(sg.anchors) >= 2 and valid:
            tail_anchor_set: dict[str, set[str]] = {}
            for p in valid:
                if p.tail and not _is_opaque_identifier(p.tail):
                    tail_anchor_set.setdefault(p.tail, set()).add(p.head)
            for p in valid:
                n_conv = len(tail_anchor_set.get(p.tail, set()))
                if n_conv >= 2:
                    # Iter 5g: bumped 2.0 -> 2.5 (CWQ 0.162 -> 0.174).
                    # Iter 5h: tested 3.0 — no further gain over 2.5 (plateau).
                    # Reverted to 2.5: smaller boost = lower over-fit risk.
                    p.score = float(p.score) * math.exp(2.5 * (n_conv - 1))
                # Iter 5p tested 0.5x penalty on single-anchor tails — no-op
                # (CWQ stayed 0.178). Conv boost already establishes ranking.

        ranked = sorted(valid, key=lambda x: x.score, reverse=True)
        top = ranked[0] if ranked else None

        # ---- Task-specific answer extraction ----------------------------------
        if abstained:
            prediction = None
        elif self.task_type == "yes_no":
            user   = (query_meta or {}).get("user")   or (query_meta or {}).get("head")
            target = (query_meta or {}).get("resource") or (query_meta or {}).get("target")
            # Predict "1" iff a valid (non-violating) path connects user to target.
            granted = bool(user and target and any(
                p.head == user and p.tail == target for p in valid
            ))
            prediction = "1" if granted else "0"
        else:
            # Entity QA: configurable aggregation.
            #   "max": top-1 non-MID path tail (good for k=1 retrieval).
            #   "sum": path-vote — sum scores per tail (good for k>=2).
            prediction = None
            if self.answer_aggregation == "sum":
                tail_mass: dict[str, float] = {}
                for p in ranked:
                    if p.tail is None or _is_opaque_identifier(p.tail):
                        continue
                    tail_mass[p.tail] = tail_mass.get(p.tail, 0.0) + float(p.score)
                if tail_mass:
                    prediction = max(tail_mass.items(), key=lambda kv: kv[1])[0]
            else:
                for p in ranked:
                    if not _is_opaque_identifier(p.tail):
                        prediction = p.tail
                        break
            if prediction is None and top is not None:
                # Fallback: walk one extra hop from the MID tail.
                tail_idx2: dict[str, list[Triple]] = {}
                for t in sg.triples:
                    tail_idx2.setdefault(t.s, []).append(t)
                follow = [t for t in tail_idx2.get(top.tail, [])
                          if not _is_opaque_identifier(t.o)]
                if follow:
                    prediction = follow[0].o
                else:
                    prediction = top.tail

        # Optional LLM verbalization for entity QA: ask the LLM to pick the
        # answer from QUEST-KG's top-N candidate paths. Keeps retrieval +
        # symbolic reasoning + abstention; only the final answer-extraction
        # step is delegated to the LLM. "QUEST-KG-LLM" hybrid variant.
        if (self.llm is not None and self.task_type == "entity"
                and not abstained and valid):
            try:
                top_paths = ranked[: self.llm_rerank_top_n]
                # Compact evidence: list of (s,r,o) for paths' triples
                ev_lines = []
                seen_keys = set()
                for p in top_paths:
                    for t in p.triples:
                        k_str = (t.s, t.r.lstrip("~"), t.o)
                        if k_str in seen_keys:
                            continue
                        seen_keys.add(k_str)
                        ev_lines.append(f"({t.s} | {t.r.lstrip('~')} | {t.o})")
                # Candidate answer list = unique non-MID tails (primary) plus
                # non-anchor intermediate nodes (secondary). Cap at 16 -- grid
                # search showed WebQSP drops at 24 (more distractors for the
                # 1.5B model) while CWQ is insensitive 16-24.
                cand_set: list[str] = []
                seen_cands: set[str] = set()
                for p in top_paths:
                    for n in (p.tail, *p.node_seq()[1:-1]):
                        if n is None or n in seen_cands or n in sg.anchors:
                            continue
                        if _is_opaque_identifier(n):
                            continue
                        seen_cands.add(n)
                        cand_set.append(n)
                        if len(cand_set) >= 16:
                            break
                    if len(cand_set) >= 16:
                        break
                if cand_set:
                    prompt = (
                        "You are a knowledge-graph QA system. Pick the SINGLE "
                        "best answer to the question from the numbered candidate "
                        "list below, using ONLY the supporting facts. Output "
                        "ONLY the number of the chosen candidate, nothing else.\n\n"
                        f"Facts:\n" + "\n".join(ev_lines[:32]) + "\n\n"
                        f"Candidates:\n" + "\n".join(
                            f"{i+1}. {c}" for i, c in enumerate(cand_set)) + "\n\n"
                        f"Question: {query}\n"
                        "Answer number:"
                    )
                    raw = self.llm.generate([prompt], max_new_tokens=8)[0]
                    raw = raw.strip().split("\n")[0].strip().strip("-*. ").strip()
                    snapped = None
                    # Primary: parse a candidate number
                    digits = re.findall(r"\d+", raw)
                    if digits:
                        try:
                            n = int(digits[0])
                            if 1 <= n <= len(cand_set):
                                snapped = cand_set[n - 1]
                        except ValueError:
                            pass
                    # Secondary: substring snap (LLM emitted a name)
                    if snapped is None:
                        raw_lower = raw.lower()
                        for c in cand_set:
                            if c.lower() == raw_lower:
                                snapped = c
                                break
                        if snapped is None:
                            for c in cand_set:
                                if (c.lower() in raw_lower
                                        and len(c) > 2):
                                    snapped = c
                                    break
                    # Only override symbolic prediction if LLM snapped to a
                    # candidate. Garbled / off-list LLM output keeps symbolic.
                    if snapped is not None:
                        prediction = snapped
            except Exception:
                pass  # fall back to symbolic prediction

        # Candidate ranking for link-prediction / QA: ordered list of distinct
        # endpoints by best path score. We include both the path tail (primary)
        # AND non-anchor intermediate nodes (secondary, for QA where the answer
        # may sit mid-path -- e.g., a 2-hop path Jamaica -> X -> Y can have the
        # answer at X rather than Y depending on relation semantics).
        seen_tails: set[str] = set()
        candidate_ranking: list[str] = []
        for p in ranked:
            # Primary: path tail
            t = p.tail
            if t is not None and t not in seen_tails:
                seen_tails.add(t)
                candidate_ranking.append(t)
            # Secondary: non-anchor intermediate nodes (path tail is already covered)
            for node in p.node_seq()[1:-1]:
                if node not in seen_tails and node not in sg.anchors:
                    seen_tails.add(node)
                    candidate_ranking.append(node)
            if len(candidate_ranking) >= 64:
                candidate_ranking = candidate_ranking[:64]
                break

        # Iter 5q: report top answer probability as confidence instead of M_q.
        # M_q saturates at 1.0 for QA datasets because FreebaseChecker has no
        # schema/types configured (no violations -> all paths "kept" -> mass=1).
        # max(answer_probs) is the canonical P(predicted_answer | Gq), which
        # gives meaningful calibration data without changing the methodology.
        conf_out = max(probs.values()) if probs else M_q
        return PredictResult(
            query=query,
            prediction=prediction,
            abstained=abstained,
            confidence=conf_out,
            entropy=H_q,
            subgraph=sg,
            top_path=top,
            candidate_paths=candidate_paths,
            extra={
                "answer_probs": probs,
                "n_valid_paths": len(valid),
                "task_type": self.task_type,
                "candidate_ranking": candidate_ranking,
            },
        )
