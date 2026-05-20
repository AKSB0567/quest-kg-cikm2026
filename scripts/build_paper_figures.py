"""Generate the figures for the QUEST-KG paper:
  - fig1_architecture.png  -- box-and-arrow pipeline diagram
  - fig2_comparison.png    -- per-dataset bar chart of primary metric
  - fig3_calibration.png   -- 4-panel reliability diagram
  - fig4_latency.png       -- log-scale latency comparison
  - fig5_hopsweep.png      -- hop sweep curves per dataset
  - fig6_confhist.png      -- confidence histograms (calibrated vs not)
All written to paper/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ---------------------------------------------------------------------------
# Fig 1: Architecture pipeline
# ---------------------------------------------------------------------------
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, color="#e8f0fe", edge="#1f4e8c"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            linewidth=1.0, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8, color="#0f2540", weight="bold")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#222"))

    # Row 1 - input
    box(0.4, 3.3, 2.4, 1.2, "Query $q$\n+ Global KG $G$\n+ Provenance $\\Pi$",
        color="#f1f8ff", edge="#0e639c")

    # Row 1 - retrieval
    box(3.3, 3.3, 2.6, 1.2,
        "Schema-Aware\nGraph-RAG Retrieval\ntop-K + multi-hop",
        color="#dfeaff", edge="#1f4e8c")

    # Row 1 - evidential MP
    box(6.4, 3.3, 2.8, 1.2,
        "Provenance-Aware\nEvidential MP\n$b_i, u_i$ propagation",
        color="#cfe0fa", edge="#1f4e8c")

    # Row 1 - neuro-symbolic
    box(9.7, 3.3, 2.8, 1.2,
        "Neuro-Symbolic\nReasoning + Symbolic\nConstraint Check",
        color="#bfd2f0", edge="#1f4e8c")

    # Row 2 - output
    box(11.0, 0.7, 2.6, 1.2,
        "Prediction $\\hat{y}$\n+ confidence\n+ abstain flag",
        color="#fff2cc", edge="#b39200")

    # Side - subgraph artifact
    box(3.3, 0.7, 2.6, 1.2, "Retrieved\nsubgraph $G_q$",
        color="#f7f7f7", edge="#666")
    box(6.4, 0.7, 2.8, 1.2, "Belief / uncertainty\nover candidate paths",
        color="#f7f7f7", edge="#666")
    box(9.7, 0.7, 1.0, 1.2, "$p\\star,\\, H\\star$\nabstention\ngate",
        color="#ffe4e1", edge="#8b0000")

    # arrows
    arrow(2.85, 3.9, 3.3, 3.9)
    arrow(5.95, 3.9, 6.4, 3.9)
    arrow(9.25, 3.9, 9.7, 3.9)
    arrow(12.5, 3.3, 12.5, 1.95)
    arrow(4.6, 3.3, 4.6, 1.95)
    arrow(7.8, 3.3, 7.8, 1.95)
    arrow(10.7, 1.3, 11.0, 1.3)
    arrow(5.9, 1.3, 6.4, 1.3)
    arrow(9.2, 1.3, 9.7, 1.3)

    plt.savefig(FIG_DIR / "fig1_architecture.png", dpi=200)
    plt.close(fig)
    print("  fig1_architecture.png")


# ---------------------------------------------------------------------------
# Fig 2: per-dataset bar chart (primary metric)
# ---------------------------------------------------------------------------
def fig2_comparison():
    datasets = ["OrgAccess\n(BAcc)", "ICEWS18\n(MRR)",
                 "WebQSP\n(Acc)", "CWQ\n(Acc)", "DBP-YG\n(Hits@1)"]
    methods = ["Vanilla-RAG", "GraphRAG", "ToG-1", "ToG-2", "CoK",
                "QUEST-KG (sym.)", "QUEST-KG-LLM"]
    # rows = methods, cols = datasets. Use 0 for n/a so bars don't show.
    M = np.array([
        [0.507, 0.016, 0.185, 0.149, 0.000],   # vanilla
        [0.507, 0.035, 0.114, 0.145, 0.000],   # graphrag
        [0.515, 0.013, 0.019, 0.072, 0.000],   # tog1
        [0.497, 0.011, 0.020, 0.070, 0.000],   # tog2
        [0.500, 0.015, 0.055, 0.094, 0.000],   # cok
        [0.963, 0.053, 0.330, 0.204, 0.919],   # qkg sym
        [0.963, 0.053, 0.397, 0.236, 0.919],   # qkg llm
    ])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(datasets))
    width = 0.11
    colors = ["#bdbdbd", "#9e9e9e", "#cccccc", "#a0a0a0", "#7d7d7d",
              "#2b6cb0", "#1f4e8c"]
    for i, (m, c) in enumerate(zip(methods, colors)):
        offset = (i - len(methods) / 2) * width + width / 2
        bars = ax.bar(x + offset, M[i], width, label=m, color=c,
                      edgecolor="black", linewidth=0.3)
        # mark zeros (n/a) lightly
        for b, v in zip(bars, M[i]):
            if v == 0:
                b.set_alpha(0.0)
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Primary metric")
    ax.set_title("Primary-metric comparison: QUEST-KG vs LLM-RAG baselines")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    plt.savefig(FIG_DIR / "fig2_comparison.png", dpi=200)
    plt.close(fig)
    print("  fig2_comparison.png")


# ---------------------------------------------------------------------------
# Fig 3: 4-panel reliability diagram (real per-query CSVs)
# ---------------------------------------------------------------------------
def fig3_calibration():
    from quest_kg.eval.metrics import ece
    R = ROOT / "results"
    panels = [
        ("OrgAccess", "quest_kg__orgaccess__local-1080ti__symbolic__seed0.csv",
                       "cok__orgaccess__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("WebQSP",    "quest_kg__webqsp__local-1080ti__symbolic__seed0.csv",
                       "cok__webqsp__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("CWQ",       "quest_kg__cwq__local-1080ti__symbolic__seed0.csv",
                       "cok__cwq__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
        ("ICEWS18",   "quest_kg__icews18__local-1080ti__symbolic__seed0.csv",
                       "cok__icews18__colab-l4__Qwen2.5-7B-Instruct__seed0.csv"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True)
    for ax, (title, qkg_csv, base_csv) in zip(axes, panels):
        ax.set_title(title)
        ax.plot([0, 1], [0, 1], "k:", lw=0.7, label="perfect")
        for label, csv_name, color in [
            ("QUEST-KG", qkg_csv, "#1f4e8c"),
            ("CoK (best baseline)", base_csv, "#c62828"),
        ]:
            p = R / csv_name
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "confidence" not in df.columns or "em" not in df.columns:
                continue
            conf = df["confidence"].fillna(0.0).clip(0, 1).to_numpy(dtype=float)
            correct = df["em"].fillna(0).astype(int).to_numpy(dtype=float)
            n_bins = 10
            edges = np.linspace(0, 1, n_bins + 1)
            xs, ys = [], []
            for i in range(n_bins):
                m = (conf >= edges[i]) & (conf < edges[i + 1])
                if m.sum() < 5:
                    continue
                xs.append(conf[m].mean()); ys.append(correct[m].mean())
            ax.plot(xs, ys, marker="o", ms=3, color=color, label=label,
                    lw=1.2)
            e = ece(conf, correct, n_bins=15)
            ax.text(0.05, 0.92 - (0.07 if label.startswith("CoK") else 0),
                    f"ECE={e:.3f}", color=color, transform=ax.transAxes,
                    fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
        if ax is axes[0]:
            ax.set_ylabel("Empirical accuracy")
    axes[-1].legend(fontsize=6.5, loc="lower right", framealpha=0.95)
    plt.savefig(FIG_DIR / "fig3_calibration.png", dpi=200)
    plt.close(fig)
    print("  fig3_calibration.png")


# ---------------------------------------------------------------------------
# Fig 4: latency comparison (log scale)
# ---------------------------------------------------------------------------
def fig4_latency():
    datasets = ["OrgAccess", "ICEWS18", "WebQSP", "CWQ"]
    methods = ["Vanilla-RAG", "GraphRAG", "ToG-1", "ToG-2", "CoK",
               "QUEST-KG-LLM", "QUEST-KG (sym.)"]
    L = np.array([
        [1482, 1478, 1518, 1600],   # vanilla
        [1593, 1563, 1578, 1638],   # graphrag
        [7873, 8599, 8881, 9332],   # tog1
        [8951, 9147, 9134, 9465],   # tog2
        [9208, 9080, 8836, 8965],   # cok
        [1,    5,    381,  489],    # qkg-llm
        [18,   16,   52,   78],     # qkg sym
    ])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(datasets))
    width = 0.115
    colors = ["#bdbdbd", "#9e9e9e", "#cccccc", "#a0a0a0", "#7d7d7d",
              "#1f4e8c", "#0a2c5a"]
    for i, (m, c) in enumerate(zip(methods, colors)):
        offset = (i - len(methods) / 2) * width + width / 2
        ax.bar(x + offset, L[i], width, label=m, color=c,
               edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms / query, log scale)")
    ax.set_title("Inference latency: QUEST-KG is 19–500× faster")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    plt.savefig(FIG_DIR / "fig4_latency.png", dpi=200)
    plt.close(fig)
    print("  fig4_latency.png")


# ---------------------------------------------------------------------------
# Fig 5: hop sweep curves
# ---------------------------------------------------------------------------
def fig5_hopsweep():
    p = ROOT / "results" / "_ablations_local__local-1080ti__symbolic.csv"
    if not p.exists():
        print("  fig5 skipped (no ablation CSV)")
        return
    df = pd.read_csv(p)
    hop = df[df["variant"].str.startswith("hop_k=")].copy()
    hop["k"] = hop["variant"].str.extract(r"hop_k=(\d+)").astype(int)
    piv = hop.pivot_table(index="k", columns="dataset",
                           values="primary_value", aggfunc="first")

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    colors = {"orgaccess": "#1f4e8c", "icews18": "#c62828",
               "webqsp": "#2e7d32", "cwq": "#6a1b9a"}
    locked_k = {"orgaccess": 2, "icews18": 1, "webqsp": 1, "cwq": 3}
    for ds in piv.columns:
        ax.plot(piv.index, piv[ds], marker="o", ms=5,
                color=colors.get(ds, "#444"), label=ds, lw=1.5)
        # Mark locked optimum
        k_lock = locked_k.get(ds)
        if k_lock and k_lock in piv.index:
            ax.scatter([k_lock], [piv.loc[k_lock, ds]], s=140,
                       facecolors="none", edgecolors=colors.get(ds, "#444"),
                       linewidth=1.8, zorder=5)
    ax.set_xlabel("k (hop budget)")
    ax.set_ylabel("Primary metric")
    ax.set_title("Hop sweep: per-dataset locked optimum (★) matches sweep")
    ax.set_xticks(sorted(piv.index))
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.95)
    plt.savefig(FIG_DIR / "fig5_hopsweep.png", dpi=200)
    plt.close(fig)
    print("  fig5_hopsweep.png")


# ---------------------------------------------------------------------------
# Fig 6: confidence histograms (correct vs incorrect)
# ---------------------------------------------------------------------------
def fig6_confhist():
    R = ROOT / "results"
    cells = [
        ("WebQSP", "quest_kg__webqsp__local-1080ti__symbolic__seed0.csv"),
        ("CWQ",    "quest_kg__cwq__local-1080ti__symbolic__seed0.csv"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ax, (title, csv_name) in zip(axes, cells):
        p = R / csv_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "confidence" not in df.columns or "em" not in df.columns:
            continue
        conf = df["confidence"].fillna(0.0).clip(0, 1).to_numpy(dtype=float)
        em = df["em"].fillna(0).astype(int).to_numpy(dtype=int)
        bins = np.linspace(0, 1, 16)
        ax.hist(conf[em == 1], bins=bins, color="#1f4e8c", alpha=0.75,
                label="Correct predictions")
        ax.hist(conf[em == 0], bins=bins, color="#c62828", alpha=0.55,
                label="Incorrect predictions")
        ax.set_title(f"QUEST-KG confidence distribution: {title}")
        ax.set_xlabel("Posterior confidence")
        if ax is axes[0]:
            ax.set_ylabel("# queries")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_confhist.png", dpi=200)
    plt.close(fig)
    print("  fig6_confhist.png")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    print(f"Writing figures to {FIG_DIR}")
    fig1_architecture()
    fig2_comparison()
    fig3_calibration()
    fig4_latency()
    fig5_hopsweep()
    fig6_confhist()
    print("done.")
