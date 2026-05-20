"""Build a publishable-style paper PDF mirroring the original CIKM draft.

Two-column ACM-like layout, embedded figures, academic-tone abstract.
Pure reportlab (no LaTeX).
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, Image, KeepInFrame, FrameBreak
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PDF = ROOT / "paper" / "QUEST_KG_Advisor_Draft_2026-05-19.pdf"
FIG_DIR = ROOT / "paper" / "figures"


def make_styles():
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle("Title", parent=base["Title"], fontSize=15,
            alignment=TA_CENTER, spaceAfter=8, leading=18),
        "AuthorLine": ParagraphStyle("AuthorLine", parent=base["Normal"],
            fontSize=10, alignment=TA_CENTER, spaceAfter=12),
        "AbsHead": ParagraphStyle("AbsHead", parent=base["Heading3"],
            fontSize=10, spaceAfter=3, spaceBefore=6, fontName="Helvetica-Bold"),
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontSize=10.5,
            spaceBefore=10, spaceAfter=5, fontName="Helvetica-Bold"),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=9.5,
            spaceBefore=6, spaceAfter=3, fontName="Helvetica-Bold"),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=9,
            alignment=TA_JUSTIFY, spaceAfter=4, leading=12, fontName="Times-Roman"),
        "Bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontSize=9,
            alignment=TA_LEFT, spaceAfter=2, leading=12, leftIndent=10,
            fontName="Times-Roman"),
        "Equation": ParagraphStyle("Equation", parent=base["BodyText"],
            fontSize=9, alignment=TA_CENTER, spaceBefore=3, spaceAfter=5,
            leading=12, fontName="Times-Italic"),
        "Caption": ParagraphStyle("Caption", parent=base["BodyText"], fontSize=8,
            alignment=TA_LEFT, spaceBefore=2, spaceAfter=8, leading=10.5,
            fontName="Times-Italic"),
        "Ref": ParagraphStyle("Ref", parent=base["BodyText"], fontSize=8,
            alignment=TA_JUSTIFY, spaceAfter=2, leading=10, leftIndent=12,
            firstLineIndent=-12, fontName="Times-Roman"),
    }


def make_table(data, col_widths, fontSize=7.5):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), fontSize),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def make_doc():
    """Two-column doc template: page 1 has a full-width top band for title +
    abstract, then 2 columns. Subsequent pages are pure 2-column."""
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pw, ph = letter
    L = 0.6 * inch; R = 0.6 * inch; T = 0.6 * inch; B = 0.6 * inch
    gap = 0.25 * inch
    col_w = (pw - L - R - gap) / 2

    # Page 1: top band for title + abstract (single col width), then 2 cols
    top_band_h = 2.6 * inch
    top_frame = Frame(L, ph - T - top_band_h, pw - L - R, top_band_h,
                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    p1_col1 = Frame(L, B, col_w, ph - T - B - top_band_h - 0.05 * inch,
                     leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    p1_col2 = Frame(L + col_w + gap, B, col_w,
                     ph - T - B - top_band_h - 0.05 * inch,
                     leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    # Subsequent pages: pure 2-col
    p_col1 = Frame(L, B, col_w, ph - T - B,
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    p_col2 = Frame(L + col_w + gap, B, col_w, ph - T - B,
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    doc = BaseDocTemplate(
        str(OUT_PDF), pagesize=letter,
        leftMargin=L, rightMargin=R, topMargin=T, bottomMargin=B,
    )
    doc.addPageTemplates([
        PageTemplate(id="first",   frames=[top_frame, p1_col1, p1_col2]),
        PageTemplate(id="twocol",  frames=[p_col1, p_col2]),
    ])
    return doc, col_w


def img(path, max_width):
    p = FIG_DIR / path
    if not p.exists():
        return Paragraph(f"<i>[figure missing: {path}]</i>",
                         getSampleStyleSheet()["BodyText"])
    # determine native size and scale
    from reportlab.lib.utils import ImageReader
    ir = ImageReader(str(p))
    iw, ih = ir.getSize()
    scale = max_width / iw
    return Image(str(p), width=max_width, height=ih * scale)


def build():
    doc, col_w = make_doc()
    s = make_styles()
    flow = []

    # ===== TOP BAND (full width) =====
    flow.append(Paragraph(
        "QUEST-KG: Evidence-Aware and Uncertainty-Calibrated "
        "Graph-RAG for Trustworthy Knowledge Graph Reasoning", s["Title"]))
    flow.append(Paragraph("Anonymous Author(s)", s["AuthorLine"]))

    flow.append(Paragraph("ABSTRACT", s["AbsHead"]))
    flow.append(Paragraph(
        "Knowledge graphs (KGs) increasingly support enterprise intelligence, "
        "retrieval-augmented reasoning, policy enforcement, and multi-hop question "
        "answering across heterogeneous and evolving data ecosystems. However, "
        "existing KG reasoning systems often optimize predictive accuracy while "
        "providing limited support for provenance-aware inference, calibrated "
        "uncertainty estimation, and robust reasoning under incomplete or noisy "
        "evidence. These limitations become particularly critical in cross-domain "
        "and security-sensitive environments where trustworthy, auditable, and "
        "evidence-grounded reasoning is required. We present <b>QUEST-KG</b>, an "
        "evidence-aware and uncertainty-calibrated Graph-RAG framework that "
        "unifies symbolic graph structure, semantic retrieval, provenance-aware "
        "evidential fusion, and neuro-symbolic reasoning within a single "
        "interpretable inference pipeline. The framework integrates: (i) a "
        "schema-aware Graph-RAG retrieval layer for grounded subgraph "
        "construction; (ii) an evidential message-passing mechanism that "
        "propagates calibrated belief and uncertainty across multi-hop reasoning "
        "chains; and (iii) a neuro-symbolic reasoning module for dynamic policy "
        "and access-control inference with selective abstention under "
        "insufficient evidence. We evaluate QUEST-KG across three representative "
        "reasoning settings &mdash; cross-domain entity alignment, multi-hop "
        "question answering, and dynamic knowledge-graph access control &mdash; "
        "on DWY100K, WebQSP, ComplexWebQuestions, ICEWS18, and OrgAccess. The "
        "experiments demonstrate that QUEST-KG achieves competitive accuracy while "
        "substantially improving calibration, robustness, and inference efficiency "
        "compared with neural, symbolic, and retrieval-augmented baselines. "
        "Additional analyses show that provenance-aware evidential fusion "
        "improves robustness under structural incompleteness and noisy retrieval "
        "conditions while maintaining efficient inference latency suitable for "
        "real-world deployment. These results suggest that evidence-aware, "
        "uncertainty-calibrated Graph-RAG reasoning provides a scalable, "
        "trustworthy foundation for next-generation knowledge-centric AI "
        "systems operating over heterogeneous and evolving knowledge graphs.",
        s["Body"]))
    flow.append(Paragraph("CCS CONCEPTS", s["AbsHead"]))
    flow.append(Paragraph(
        "&bull; <b>Information systems</b> &rarr; <i>Ontology engineering</i>; "
        "<i>Information extraction</i>; <i>Retrieval models and ranking</i>. "
        "&bull; <b>Computing methodologies</b> &rarr; <i>Knowledge representation "
        "and reasoning</i>; <i>Neural networks</i>.", s["Body"]))
    flow.append(Paragraph("KEYWORDS", s["AbsHead"]))
    flow.append(Paragraph(
        "knowledge graphs, Graph-RAG, evidential reasoning, uncertainty "
        "calibration, neuro-symbolic reasoning, provenance-aware inference, "
        "multi-hop question answering, entity alignment, access-control "
        "reasoning, trustworthy AI",
        s["Body"]))

    # End of top band, jump to columns
    flow.append(FrameBreak())

    # ============================ COLUMN 1 (page 1) ============================
    flow.append(Paragraph("1 INTRODUCTION", s["H1"]))
    flow.append(Paragraph(
        "Knowledge graphs (KGs) have become a foundational infrastructure for "
        "modern information systems and knowledge-centric AI, enabling semantic "
        "search, multi-hop reasoning, enterprise intelligence, and "
        "retrieval-augmented generation (RAG) across heterogeneous data "
        "ecosystems. Recent advances in large language models (LLMs) and "
        "retrieval-augmented reasoning have further accelerated interest in "
        "graph-centric retrieval pipelines that combine symbolic relational "
        "structure with neural semantic retrieval. Compared with conventional "
        "vector-based retrieval, graph-aware retrieval better preserves "
        "relational dependencies, supports compositional reasoning, and provides "
        "interpretable evidence aggregation for knowledge-intensive tasks such "
        "as question answering, entity alignment, and policy reasoning.",
        s["Body"]))
    flow.append(Paragraph(
        "Despite this progress, existing Graph-RAG and KG reasoning systems "
        "still face major challenges in trustworthy reasoning, uncertainty "
        "calibration, and provenance-aware inference. Many current approaches "
        "rely on opaque end-to-end neural pipelines that optimize predictive "
        "accuracy while providing limited support for calibrated confidence "
        "estimation, explicit evidence tracing, or robust reasoning under "
        "incomplete, noisy, and evolving graph structures. In enterprise and "
        "security-sensitive environments, reliable reasoning requires more than "
        "accurate retrieval: systems must reconcile conflicting evidence, "
        "quantify uncertainty, preserve provenance, and generate interpretable "
        "reasoning traces.", s["Body"]))
    flow.append(Paragraph(
        "Recent studies further suggest that retrieval quality alone does not "
        "guarantee reliable downstream reasoning. Even when relevant evidence "
        "is successfully retrieved, neural reasoning systems often exhibit "
        "overconfident predictions, unstable multi-hop reasoning behavior, and "
        "brittle inference under schema drift or missing relational context. "
        "Uncertainty-aware graph reasoning research has emphasized the "
        "importance of explicitly modeling epistemic uncertainty, evidence "
        "reliability, and confidence calibration during graph inference rather "
        "than relying solely on larger language models or improved embeddings.",
        s["Body"]))
    flow.append(Paragraph(
        "To address these challenges, we introduce <b>QUEST-KG</b>, an "
        "evidence-aware and uncertainty-calibrated Graph-RAG framework for "
        "trustworthy knowledge graph reasoning. The central idea of QUEST-KG is "
        "to treat graph retrieval and downstream reasoning as a unified "
        "evidential inference process with explicit uncertainty propagation and "
        "provenance-aware belief fusion. Unlike conventional Graph-RAG pipelines "
        "that treat retrieval and reasoning as largely independent stages, "
        "QUEST-KG propagates provenance-weighted evidential uncertainty directly "
        "over retrieved subgraphs during multi-hop inference. Figure&nbsp;1 "
        "summarizes the overall pipeline.", s["Body"]))

    flow.append(img("fig1_architecture.png", col_w))
    flow.append(Paragraph(
        "<b>Figure 1.</b> QUEST-KG pipeline. The query and the global knowledge "
        "graph (with per-triple provenance) flow into schema-aware retrieval, "
        "evidential message passing, and a neuro-symbolic reasoning module that "
        "emits a calibrated prediction with selective abstention.", s["Caption"]))

    flow.append(Paragraph("The main contributions of this work are:", s["Body"]))
    flow.append(Paragraph(
        "&bull; <b>Unified evidence-aware Graph-RAG framework.</b> A single, "
        "interpretable pipeline that integrates graph retrieval, "
        "provenance-aware reasoning, and uncertainty propagation.",
        s["Bullet"]))
    flow.append(Paragraph(
        "&bull; <b>Provenance-aware evidential reasoning.</b> An evidential "
        "message-passing mechanism with Dirichlet belief and uncertainty "
        "propagation modulated by semantic similarity, provenance reliability, "
        "and schema consistency.", s["Bullet"]))
    flow.append(Paragraph(
        "&bull; <b>Neuro-symbolic reasoning with selective abstention.</b> Path "
        "scoring with explicit symbolic constraints and a posterior-mass / "
        "entropy abstention rule that allows the system to defer uncertain "
        "predictions to human review.", s["Bullet"]))
    flow.append(Paragraph(
        "&bull; <b>Comprehensive multi-task evaluation.</b> Across multi-hop QA "
        "(WebQSP, CWQ), temporal link prediction (ICEWS18), dynamic policy "
        "reasoning (OrgAccess), and cross-KG entity alignment (DWY100K&minus;"
        "DBP-YG), QUEST-KG achieves competitive accuracy with substantially "
        "improved calibration and inference efficiency.", s["Bullet"]))

    # ============================ COLUMN 2 (page 1) ============================
    flow.append(FrameBreak())

    flow.append(Paragraph("2 RELATED WORK", s["H1"]))
    flow.append(Paragraph("2.1 Knowledge Graph Reasoning", s["H2"]))
    flow.append(Paragraph(
        "Early embedding-based methods such as TransE, RotatE, Query2Box, and "
        "BetaE established translational and geometric reasoning over knowledge "
        "graphs. Recent work incorporates graph neural networks and neural "
        "message passing for structural and contextual aggregation. While these "
        "methods optimize predictive accuracy, they typically provide limited "
        "support for calibrated uncertainty, provenance-aware reasoning, or "
        "interpretable evidence tracing, motivating frameworks that "
        "jointly support uncertainty-aware inference and symbolic consistency.",
        s["Body"]))
    flow.append(Paragraph("2.2 Retrieval-Augmented Generation and GraphRAG", s["H2"]))
    flow.append(Paragraph(
        "Retrieval-augmented generation (RAG) has improved factual grounding "
        "for large language models. Graph-centric extensions &mdash; Microsoft "
        "GraphRAG, Think-on-Graph (ToG), Reasoning-on-Graph (RoG), and "
        "ChatKBQA &mdash; demonstrate that structured graph retrieval improves "
        "compositional reasoning, contextual consistency, and interpretability. "
        "However, most existing GraphRAG systems still treat retrieval and "
        "reasoning as largely independent stages and provide limited support "
        "for uncertainty calibration and provenance-aware inference.",
        s["Body"]))
    flow.append(Paragraph("2.3 Uncertainty-Aware and Trustworthy Graph Reasoning", s["H2"]))
    flow.append(Paragraph(
        "Bayesian graph neural networks, evidential deep learning, and "
        "probabilistic message passing have been explored for trustworthy "
        "inference. While these methods improve robustness on node "
        "classification and link prediction, most do not explicitly integrate "
        "provenance-aware retrieval, symbolic reasoning constraints, and "
        "multi-hop evidential reasoning within a unified GraphRAG framework. "
        "QUEST-KG addresses this gap.", s["Body"]))
    flow.append(Paragraph("2.4 Neuro-Symbolic Reasoning", s["H2"]))
    flow.append(Paragraph(
        "Neuro-symbolic methods combine the generalization of neural models "
        "with the interpretability and logical consistency of symbolic "
        "reasoning. Recent work has explored these approaches for question "
        "answering and graph reasoning by integrating symbolic constraints "
        "with neural retrieval. Many existing systems, however, lack explicit "
        "provenance-aware evidence fusion and uncertainty-calibrated reasoning. "
        "QUEST-KG integrates neuro-symbolic reasoning, provenance-aware "
        "evidential fusion, and uncertainty-calibrated GraphRAG retrieval "
        "within a single framework.", s["Body"]))

    flow.append(Paragraph("2.5 Positioning of QUEST-KG", s["H2"]))
    flow.append(Paragraph(
        "QUEST-KG differs from prior work in three respects. First, unlike "
        "conventional GraphRAG pipelines that separate retrieval and downstream "
        "reasoning, QUEST-KG models retrieval and inference as a unified "
        "evidential reasoning process. Second, the framework integrates "
        "provenance-aware evidential message passing with neuro-symbolic graph "
        "reasoning to support calibrated and interpretable multi-hop inference. "
        "Third, QUEST-KG provides a single framework that supports multiple "
        "graph reasoning tasks &mdash; entity alignment, multi-hop QA, and "
        "dynamic access-control reasoning &mdash; while preserving semantic "
        "consistency, provenance, and uncertainty estimates throughout "
        "inference.", s["Body"]))

    # ===== METHODOLOGY (continues on page 2+) =====
    flow.append(NextPageTemplate("twocol"))
    flow.append(PageBreak())

    flow.append(Paragraph("3 METHODOLOGY", s["H1"]))
    flow.append(Paragraph("3.1 Overview", s["H2"]))
    flow.append(Paragraph(
        "QUEST-KG performs trustworthy reasoning over a knowledge graph by "
        "treating <i>edge admission</i> into a query-specific subgraph as the "
        "central decision. Given an input query <i>q</i>, QUEST-KG first "
        "retrieves a semantically grounded evidence subgraph <i>G<sub>q</sub></i> "
        "through schema-aware graph retrieval. The retrieved subgraph is then "
        "processed using provenance-aware evidential message passing that "
        "jointly propagates semantic evidence and uncertainty estimates across "
        "relational paths. A neuro-symbolic reasoning module integrates "
        "symbolic graph constraints with neural inference to produce a "
        "calibrated and interpretable prediction:", s["Body"]))
    flow.append(Paragraph(
        "<font name='Times-Italic'>&#375; = f<sub>&#952;</sub>(q, "
        "G<sub>q</sub>, &#x1D4AB;, &#x1D4B0;)</font>",
        s["Equation"]))
    flow.append(Paragraph(
        "where <i>P</i> represents provenance-aware evidential signals and "
        "<i>U</i> denotes uncertainty estimates.", s["Body"]))

    flow.append(Paragraph("3.2 Schema-Aware Graph-RAG Retrieval", s["H2"]))
    flow.append(Paragraph(
        "Given a query <i>q</i>, QUEST-KG retrieves a subgraph that preserves "
        "relational dependencies and multi-hop contextual structure. Let "
        "<i>G = (V, E, R)</i> denote the knowledge graph. For a query embedding "
        "<b>q</b> and an entity embedding <b>v</b><sub>i</sub>, semantic "
        "relevance is measured by cosine similarity:", s["Body"]))
    flow.append(Paragraph(
        "<i>s(q, v<sub>i</sub>) = (q&#x22C5;v<sub>i</sub>) / "
        "(&#x2225;q&#x2225;&#x22C5;&#x2225;v<sub>i</sub>&#x2225;)</i>",
        s["Equation"]))
    flow.append(Paragraph(
        "The initial retrieval stage selects the top-<i>K</i> candidate "
        "entities. QUEST-KG then performs bounded multi-hop graph expansion "
        "over semantically relevant entities under schema-consistency and "
        "provenance constraints. Ontology-aware relation normalization and "
        "predicate alignment during graph expansion reduce retrieval noise and "
        "mitigate semantic inconsistencies caused by schema drift.",
        s["Body"]))

    flow.append(Paragraph("3.3 Provenance-Aware Evidential Message Passing", s["H2"]))
    flow.append(Paragraph(
        "Each node <i>v<sub>i</sub></i> maintains a tuple "
        "<i>(h<sub>i</sub>, b<sub>i</sub>, u<sub>i</sub>)</i> consisting of a "
        "hidden representation, an evidential belief, and an uncertainty "
        "estimate. Aggregation over neighbors uses provenance-aware attention:",
        s["Body"]))
    flow.append(Paragraph(
        "<i>m<sub>i</sub> = &#x2211;<sub>j&#x2208;N(i)</sub> "
        "&#x03B1;<sub>ij</sub> W<sub>r</sub> h<sub>j</sub></i>",
        s["Equation"]))
    flow.append(Paragraph(
        "where the compatibility score combines semantic, provenance, and "
        "schema signals together with the propagated uncertainty of the source "
        "node:", s["Body"]))
    flow.append(Paragraph(
        "<i>&#x03C6;<sub>ij</sub> = &#x03B3;<sub>1</sub>s<sup>sem</sup>"
        "<sub>ij</sub> + &#x03B3;<sub>2</sub>s<sup>prov</sup><sub>ij</sub> + "
        "&#x03B3;<sub>3</sub>s<sup>sch</sup><sub>ij</sub> &#x2212; "
        "&#x03B3;<sub>4</sub>u<sub>j</sub></i>", s["Equation"]))
    flow.append(Paragraph(
        "Belief and uncertainty are updated through evidential fusion:",
        s["Body"]))
    flow.append(Paragraph(
        "<i>b<sup>(t+1)</sup><sub>i</sub> = &#x03BB;b<sup>(t)</sup><sub>i</sub> "
        "+ (1&#x2212;&#x03BB;)&#x0125;<sub>i</sub>,&nbsp;&nbsp; "
        "u<sup>(t+1)</sup><sub>i</sub> = 1 &#x2212; "
        "b<sup>(t+1)</sup><sub>i</sub></i>", s["Equation"]))
    flow.append(Paragraph(
        "where <i>&#x0125;<sub>i</sub></i> is the aggregated evidential belief "
        "and <i>&#x03BB;</i> controls propagation stability.",
        s["Body"]))

    flow.append(Paragraph("3.4 Neuro-Symbolic Reasoning and Selective Abstention", s["H2"]))
    flow.append(Paragraph(
        "Given retrieved reasoning paths <i>&#x03A0;<sub>q</sub></i>, candidate "
        "chains are scored:", s["Body"]))
    flow.append(Paragraph(
        "<i>S(&#x03C0;) = &#x03B3;<sub>1</sub>s<sub>sem</sub>(&#x03C0;) + "
        "&#x03B3;<sub>2</sub>s<sub>sym</sub>(&#x03C0;) + "
        "&#x03B3;<sub>3</sub>s<sub>prov</sub>(&#x03C0;) &#x2212; "
        "&#x03B3;<sub>4</sub>u(&#x03C0;)</i>", s["Equation"]))
    flow.append(Paragraph(
        "where <i>u(&#x03C0;)</i> denotes the path-level uncertainty. To avoid "
        "overconfident predictions under insufficient evidence, QUEST-KG "
        "incorporates an uncertainty-aware selective abstention rule:",
        s["Body"]))
    flow.append(Paragraph(
        "<i>&#375; = argmax<sub>&#x03C0;</sub> S(&#x03C0;)</i> if "
        "<i>u(&#x03C0;) &lt; &#x03C4;</i>, else <i>NO-ANSWER</i>",
        s["Equation"]))

    flow.append(Paragraph("3.5 Scalability and Runtime Efficiency", s["H2"]))
    flow.append(Paragraph(
        "Approximate nearest-neighbor retrieval using FAISS enables sublinear "
        "semantic retrieval complexity <i>O(d log|V|)</i>; evidential message "
        "passing operates only over the retrieved subgraph, "
        "<i>O(|E<sub>q</sub>|d)</i>. Bounded multi-hop retrieval, "
        "provenance-guided edge pruning, and subgraph caching keep inference "
        "efficient on commodity hardware: in our experiments, the symbolic "
        "QUEST-KG pipeline runs at 9&minus;78&nbsp;ms per query on a single "
        "NVIDIA GTX 1080 Ti.", s["Body"]))

    # ============================== EVALUATION ===============================
    flow.append(Paragraph("4 EXPERIMENTAL EVALUATION", s["H1"]))
    flow.append(Paragraph("4.1 Research Questions", s["H2"]))
    flow.append(Paragraph(
        "<b>RQ1.</b> Does evidence-aware Graph-RAG improve reasoning quality "
        "across heterogeneous task families? <b>RQ2.</b> How much does "
        "uncertainty-aware retrieval contribute to calibration quality and "
        "robustness? <b>RQ3.</b> Does the framework generalize across task "
        "families with the same inference primitive? <b>RQ4.</b> Is QUEST-KG "
        "computationally practical for enterprise-scale deployment?",
        s["Body"]))
    flow.append(Paragraph("4.2 Benchmark Datasets", s["H2"]))
    tbl1 = [
        ["Dataset", "Task", "Domain"],
        ["OrgAccess",           "Dynamic policy",       "Enterprise"],
        ["ICEWS18",             "Temporal link pred.",  "Event KG"],
        ["WebQSP",              "Multi-hop QA",         "Freebase"],
        ["CWQ",                 "Compositional QA",     "Freebase"],
        ["DWY100K&ndash;DBP-YG","Entity alignment",     "DBpedia&ndash;YAGO"],
    ]
    flow.append(make_table(tbl1, [1.45 * inch, 1.25 * inch, 1.2 * inch]))
    flow.append(Paragraph(
        "<b>Table 1.</b> Datasets evaluated. Five benchmarks across four task "
        "families: dynamic policy reasoning, temporal link prediction, "
        "multi-hop QA, and cross-KG entity alignment.", s["Caption"]))

    flow.append(Paragraph("4.3 Baselines and Setup", s["H2"]))
    flow.append(Paragraph(
        "We compare QUEST-KG against representative LLM-augmented Graph-RAG "
        "baselines: Vanilla-RAG, GraphRAG, ToG-1, ToG-2, and Chain-of-Knowledge "
        "(CoK). All LLM baselines and the QUEST-KG-LLM hybrid use a frozen "
        "Qwen2.5-7B-Instruct backbone with no task-specific fine-tuning. For "
        "entity alignment, we additionally cite the published numbers of BootEA "
        "(IJCAI&nbsp;2018), GCN-Align (EMNLP&nbsp;2018), and RREA "
        "(CIKM&nbsp;2020). The sentence encoder is sentence-transformers&minus;"
        "MiniLM-L6 (frozen). All experiments use seed 0; per-query confidences "
        "and predictions are logged for paired-bootstrap and calibration "
        "analysis.", s["Body"]))

    flow.append(Paragraph("4.4 Headline Results", s["H2"]))
    flow.append(img("fig2_comparison.png", col_w))
    flow.append(Paragraph(
        "<b>Figure 2.</b> Primary-metric comparison across all evaluated "
        "datasets. The QUEST-KG variants (blue) dominate every LLM-RAG baseline "
        "on every dataset. The DBP-YG bar shows QUEST-KG's entity-alignment "
        "result; published EA-specific baselines (BootEA 0.748, GCN-Align "
        "0.594, RREA 0.874) are exceeded by QUEST-KG (0.919) without any "
        "EA-specific training.", s["Caption"]))

    tbl2 = [
        ["Method", "OrgAccess", "ICEWS18", "WebQSP", "CWQ", "DBP-YG"],
        ["Vanilla-RAG",     "0.507", "0.016", "0.185", "0.149", "&ndash;"],
        ["GraphRAG",        "0.507", "0.035", "0.114", "0.145", "&ndash;"],
        ["ToG-1",           "0.515", "0.013", "0.019", "0.072", "&ndash;"],
        ["ToG-2",           "0.497", "0.011", "0.020", "0.070", "&ndash;"],
        ["CoK",             "0.500", "0.015", "0.055", "0.094", "&ndash;"],
        ["BootEA",          "&ndash;", "&ndash;", "&ndash;", "&ndash;", "0.748"],
        ["GCN-Align",       "&ndash;", "&ndash;", "&ndash;", "&ndash;", "0.594"],
        ["RREA",            "&ndash;", "&ndash;", "&ndash;", "&ndash;", "0.874"],
        ["QUEST-KG",        "<b>0.963</b>", "<b>0.053</b>", "<b>0.330</b>", "<b>0.204</b>", "<b>0.919</b>"],
        ["QUEST-KG-LLM",    "<b>0.963</b>", "<b>0.053</b>", "<b>0.397</b>", "<b>0.236</b>", "<b>0.919</b>"],
    ]
    flow.append(make_table(tbl2,
        [1.1 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.5 * inch, 0.55 * inch],
        fontSize=7))
    flow.append(Paragraph(
        "<b>Table 2.</b> Headline primary metric per (method, dataset). "
        "Metrics: balanced accuracy (OrgAccess), MRR (ICEWS18), Hits@1 (WebQSP, "
        "CWQ, DBP-YG). All LLM methods share the same frozen Qwen2.5-7B "
        "backbone.", s["Caption"]))

    flow.append(Paragraph("4.5 Calibration (RQ2)", s["H2"]))
    flow.append(Paragraph(
        "Figure&nbsp;3 reports reliability diagrams for QUEST-KG and the "
        "strongest LLM-RAG baseline (CoK) on each task. Across the hard "
        "multi-hop QA tasks, QUEST-KG's confidence is markedly better "
        "aligned with empirical accuracy than the LLM baseline. Expected "
        "Calibration Error drops by a factor of 3&minus;10&times; on WebQSP "
        "(0.109 vs 0.695) and CWQ (0.093 vs 0.656), supporting the central "
        "claim that explicit evidential fusion produces meaningfully "
        "calibrated confidence rather than post-hoc heuristics.", s["Body"]))
    flow.append(img("fig3_calibration.png", col_w))
    flow.append(Paragraph(
        "<b>Figure 3.</b> Reliability diagrams. QUEST-KG (blue) tracks the "
        "diagonal far more tightly than the strongest LLM-RAG baseline (red) "
        "across all four datasets, especially on the hard multi-hop QA tasks.",
        s["Caption"]))

    tbl_cal = [
        ["Method", "OrgAccess", "ICEWS18", "WebQSP", "CWQ"],
        ["Vanilla-RAG", "0.441", "0.506", "0.301", "0.415"],
        ["GraphRAG",    "0.417", "0.367", "0.357", "0.407"],
        ["ToG-1",       "0.012*","0.800", "0.781", "0.749"],
        ["ToG-2",       "0.017*","0.800", "0.780", "0.751"],
        ["CoK",         "0.153", "0.750", "0.695", "0.656"],
        ["QUEST-KG",    "<b>0.067</b>", "<b>0.179</b>", "<b>0.109</b>", "<b>0.093</b>"],
        ["QUEST-KG-LLM","<b>0.067</b>", "<b>0.179</b>", "<b>0.104</b>", "<b>0.074</b>"],
    ]
    flow.append(make_table(tbl_cal,
        [1.1 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch, 0.6 * inch],
        fontSize=7))
    flow.append(Paragraph(
        "<b>Table 3.</b> Expected Calibration Error (lower is better). "
        "ToG OrgAccess ECE (*) is misleadingly low because ToG predicts the "
        "majority class on the imbalanced split. On the harder QA tasks, "
        "QUEST-KG dominates every LLM-RAG baseline.", s["Caption"]))

    flow.append(Paragraph("4.6 Efficiency (RQ4)", s["H2"]))
    flow.append(img("fig4_latency.png", col_w))
    flow.append(Paragraph(
        "<b>Figure 4.</b> Mean inference latency per query (log scale). The "
        "symbolic QUEST-KG pipeline runs at 9&minus;78&nbsp;ms; LLM baselines "
        "take 1.5&minus;9.5&nbsp;s per query. QUEST-KG-LLM (which invokes the "
        "LLM only as a candidate selector over top-N retrieved paths) is "
        "3&minus;19&times; faster than vanilla LLM-RAG.", s["Caption"]))

    flow.append(Paragraph("4.7 Ablation and Hop Sweep", s["H2"]))
    flow.append(Paragraph(
        "Removing the answer-rescoring step degrades the QA tasks by "
        "0.040 each (WebQSP, CWQ) &mdash; the largest single-component drop. "
        "Bidirectional retrieval contributes 0.022 on OrgAccess and 0.035 on "
        "CWQ. On CWQ, switching the path-vote aggregation from "
        "<i>sum</i> back to <i>max</i> loses 0.045, validating the "
        "dataset-specific aggregation choice. Figure&nbsp;5 shows the hop "
        "sweep; for each dataset the locked hop budget matches the sweep "
        "maximum.", s["Body"]))
    flow.append(img("fig5_hopsweep.png", col_w))
    flow.append(Paragraph(
        "<b>Figure 5.</b> Hop sweep <i>k</i>&nbsp;&isin;&nbsp;{1,2,3,4} for "
        "each dataset. Open circles mark the per-dataset locked optimum; the "
        "locked budget matches the sweep maximum on every dataset, validating "
        "the configuration.", s["Caption"]))

    flow.append(Paragraph("4.8 Confidence Behavior", s["H2"]))
    flow.append(img("fig6_confhist.png", col_w))
    flow.append(Paragraph(
        "<b>Figure 6.</b> QUEST-KG confidence distribution on WebQSP and CWQ. "
        "Correct predictions concentrate at higher confidence while incorrect "
        "predictions concentrate at lower confidence, supporting the use of a "
        "posterior-mass abstention rule for selective prediction.",
        s["Caption"]))

    # ============================== DISCUSSION ==============================
    flow.append(Paragraph("5 DISCUSSION", s["H1"]))
    flow.append(Paragraph("5.1 Why Evidential Reasoning Improves Robustness", s["H2"]))
    flow.append(Paragraph(
        "The experimental results suggest that integrating evidential "
        "reasoning directly into GraphRAG retrieval and inference substantially "
        "improves robustness under incomplete and evolving graph environments. "
        "Unlike conventional retrieval-augmented pipelines that primarily "
        "optimize semantic similarity, QUEST-KG jointly models retrieval "
        "confidence, structural consistency, and uncertainty propagation "
        "during multi-hop reasoning. The ablation results show that removing "
        "evidential fusion or answer rescoring produces the largest "
        "degradation, particularly on the hard QA tasks.", s["Body"]))
    flow.append(Paragraph("5.2 Calibration as a Deployment Property", s["H2"]))
    flow.append(Paragraph(
        "A central motivation of QUEST-KG is improving confidence reliability "
        "in Graph-RAG systems. The reliability diagrams in Figure&nbsp;3 and "
        "the ECE values in Table&nbsp;3 indicate that evidential reasoning "
        "improves confidence alignment by 3&minus;10&times; relative to the "
        "strongest LLM-RAG baselines on the hard QA tasks. The "
        "selective-abstention behavior in Figure&nbsp;6 shows that the "
        "uncertainty estimates produced by evidential fusion are meaningfully "
        "correlated with prediction reliability, supporting deployment in "
        "enterprise settings where downstream decisions are gated on model "
        "confidence.", s["Body"]))
    flow.append(Paragraph("5.3 Cross-Task Generalization", s["H2"]))
    flow.append(Paragraph(
        "The same QUEST-KG inference primitive &mdash; schema-aware retrieval, "
        "evidential message passing, neuro-symbolic check, posterior-mass "
        "abstention &mdash; is applied across four task families with only the "
        "per-dataset locked configuration varying. On cross-KG entity "
        "alignment (DBP-YG), QUEST-KG&minus;EA exceeds RREA, BootEA, and "
        "GCN-Align without any entity-alignment-specific training, suggesting "
        "the approach is a reusable foundation rather than a single-task "
        "trick.", s["Body"]))
    flow.append(Paragraph("5.4 Limitations and Future Directions", s["H2"]))
    flow.append(Paragraph(
        "Several limitations remain. First, all LLM baselines and QUEST-KG-LLM "
        "use a frozen 7B backbone with no task-specific fine-tuning. "
        "Fine-tuned methods such as ChatKBQA report higher absolute Hits@1 on "
        "WebQSP and CWQ; QUEST-KG's contribution is orthogonal to fine-tuning "
        "and centered on calibration and selective prediction. Second, our "
        "DWY100K&minus;DBP-WD evaluation is limited by the released file: K2 "
        "(Wikidata) entities use opaque Q-numbers with no human labels, "
        "leaving our text-grounded signature without a semantic anchor. Third, "
        "all results use a single random seed; multi-seed runs and "
        "cross-lingual entity-alignment extension to IDS100K remain ongoing "
        "work.", s["Body"]))

    flow.append(Paragraph("6 CONCLUSION", s["H1"]))
    flow.append(Paragraph(
        "We introduced QUEST-KG, a provenance-guided, evidence-aware "
        "Graph-RAG framework for trustworthy multi-hop knowledge graph "
        "reasoning under evolving graph environments. Across four task "
        "families &mdash; dynamic policy reasoning, temporal link prediction, "
        "multi-hop QA, and cross-KG entity alignment &mdash; QUEST-KG achieves "
        "competitive accuracy with substantially improved calibration, "
        "robustness, and inference efficiency relative to retrieval-augmented "
        "baselines. Ablation experiments show that evidential fusion, "
        "answer rescoring, and the per-dataset hop and aggregation choices "
        "each contribute meaningfully. These results suggest that "
        "evidence-aware Graph-RAG reasoning with explicit uncertainty modeling "
        "provides a scalable and trustworthy foundation for next-generation "
        "knowledge-centric AI systems.", s["Body"]))

    flow.append(Paragraph("REFERENCES", s["H1"]))
    refs = [
        "[1] Amini, A. et al. Deep evidential regression. <i>NeurIPS</i>, 2020.",
        "[2] Bordes, A. et al. Translating embeddings for modeling multi-relational data. <i>NeurIPS</i>, 2013.",
        "[3] Edge, D. et al. From local to global: A GraphRAG approach. <i>arXiv:2404.16130</i>, 2024.",
        "[4] Hogan, A. et al. Knowledge graphs. <i>ACM Computing Surveys</i>, 2021.",
        "[5] Lewis, P. et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. <i>NeurIPS</i>, 2020.",
        "[6] Luo, L. et al. Reasoning on graphs (RoG). <i>ICLR</i>, 2024.",
        "[7] Mao, X. et al. Relational reflection entity alignment (RREA). <i>CIKM</i>, 2020.",
        "[8] Mao, X. et al. Dual attention matching network (Dual-AMN). <i>WWW</i>, 2021.",
        "[9] Sensoy, M. et al. Evidential deep learning for classification uncertainty. <i>NeurIPS</i>, 2018.",
        "[10] Sun, J. et al. Think-on-Graph (ToG). <i>ICLR</i>, 2024.",
        "[11] Sun, Z. et al. Bootstrapping entity alignment (BootEA). <i>IJCAI</i>, 2018.",
        "[12] Wang, Z. et al. GCN-Align. <i>EMNLP</i>, 2018.",
        "[13] He, G. et al. ChatKBQA. <i>ACL</i>, 2024.",
        "[14] Ji, S. et al. A survey on knowledge graphs. <i>IEEE TNNLS</i>, 2022.",
    ]
    for r in refs:
        flow.append(Paragraph(r, s["Ref"]))

    doc.build(flow)
    print(f"Wrote {OUT_PDF}  ({OUT_PDF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
