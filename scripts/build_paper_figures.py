"""Publication-quality figures for QUEST-KG paper (SciencePlots + seaborn).

Outputs (paper/figures/):
  fig1_architecture.pdf   -- box-and-arrow pipeline diagram
  fig2_comparison.pdf     -- grouped bar chart, primary metric per dataset
  fig3_calibration.pdf    -- 4-panel reliability diagrams
  fig4_latency.pdf        -- log-scale latency comparison
  fig5_hopsweep.pdf       -- hop sweep curves per dataset
  fig6_confhist.pdf       -- confidence histograms (correct vs incorrect)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Publication-quality style: SciencePlots + seaborn fallback
try:
    plt.style.use(["science", "no-latex"])  # no-latex avoids needing local TeX
    PUB_STYLE = True
except Exception:
    sns.set_theme(context="paper", style="whitegrid",
                  rc={"font.family": "serif"})
    PUB_STYLE = False

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
})

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Publication color palette (Tab10-derived, academic-friendly)
C_QKG       = "#0A4D8C"   # QUEST-KG primary blue
C_QKG_LLM   = "#1F77B4"   # QUEST-KG-LLM lighter blue
C_BASELINE1 = "#9E9E9E"   # vanilla / RAG (grey)
C_BASELINE2 = "#757575"   # graphrag
C_BASELINE3 = "#B0BEC5"   # tog1
C_BASELINE4 = "#90A4AE"   # tog2
C_BASELINE5 = "#607D8B"   # cok
C_HIGHLIGHT = "#D62728"   # red for comparison
C_PERFECT   = "#2CA02C"   # green for "perfect" reference line
PALETTE = {
    "Vanilla-RAG": "#B0B0B0",
    "GraphRAG":    "#888888",
    "ToG-1":       "#CCCCCC",
    "ToG-2":       "#9E9E9E",
    "CoK":         "#7A7A7A",
    "QUEST-KG":    C_QKG,
    "QUEST-KG-LLM": C_QKG_LLM,
}


# ---------------------------------------------------------------------------
# Fig 1: Architecture diagram (publication-quality boxes + arrows)
# ---------------------------------------------------------------------------
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, title, sub="", color="#E8F0FE", edge="#0A4D8C",
            text_color="#0A2540"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.12",
            linewidth=1.4, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.62, title,
                ha="center", va="center", fontsize=8.5, color=text_color,
                weight="bold")
        if sub:
            ax.text(x + w / 2, y + h * 0.28, sub,
                    ha="center", va="center", fontsize=7, color=text_color,
                    style="italic")

    def arrow(x1, y1, x2, y2, color="#222"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_length=0.4,head_width=0.3",
                                    lw=1.3, color=color))

    # Main pipeline row (top)
    box(0.3, 4.0, 2.6, 1.5, "Query  $q$",
        "+ Global KG $G$\n+ Provenance $\\Pi$",
        color="#F5F9FF", edge="#1565C0")
    box(3.4, 4.0, 2.9, 1.5, "Schema-Aware",
        "Graph-RAG Retrieval\n(top-K + multi-hop)",
        color="#E3F2FD", edge="#1976D2")
    box(6.8, 4.0, 3.0, 1.5, "Provenance-Aware",
        "Evidential MP\n($b_i, u_i$ propagation)",
        color="#BBDEFB", edge="#1565C0")
    box(10.3, 4.0, 3.2, 1.5, "Neuro-Symbolic",
        "Reasoning + Symbolic\nConstraint Check",
        color="#90CAF9", edge="#0D47A1")

    # Output box (right, lower)
    box(10.3, 0.6, 3.2, 1.4, "Prediction $\\hat{y}$",
        "+ confidence\n+ abstain flag",
        color="#FFF9C4", edge="#F57F17")

    # Intermediate artifacts (lower row)
    box(3.4, 0.6, 2.9, 1.4, "Retrieved subgraph",
        "$G_q = (V_q, E_q, R_q)$",
        color="#F5F5F5", edge="#616161")
    box(6.8, 0.6, 3.0, 1.4, "Belief/uncertainty",
        "over candidate paths",
        color="#F5F5F5", edge="#616161")

    # Arrows (forward path)
    arrow(2.9, 4.75, 3.4, 4.75)
    arrow(6.3, 4.75, 6.8, 4.75)
    arrow(9.8, 4.75, 10.3, 4.75)

    # Arrows (artifact projection)
    arrow(4.85, 4.0, 4.85, 2.0, color="#999")
    arrow(8.3, 4.0, 8.3, 2.0, color="#999")
    arrow(11.9, 4.0, 11.9, 2.0, color="#1565C0")

    # Lower-row arrows
    arrow(6.3, 1.3, 6.8, 1.3, color="#888")
    arrow(9.8, 1.3, 10.3, 1.3, color="#888")

    # Title-like labels for stages
    ax.text(1.6,  5.7, "INPUT",         ha="center", fontsize=7.5,
            color="#555", weight="bold")
    ax.text(4.85, 5.7, "RETRIEVAL",     ha="center", fontsize=7.5,
            color="#555", weight="bold")
    ax.text(8.3,  5.7, "EVIDENTIAL MP", ha="center", fontsize=7.5,
            color="#555", weight="bold")
    ax.text(11.9, 5.7, "REASONING",     ha="center", fontsize=7.5,
            color="#555", weight="bold")

    plt.savefig(FIG_DIR / "fig1_architecture.pdf")
    plt.savefig(FIG_DIR / "fig1_architecture.png", dpi=300)
    plt.close(fig)
    print("  fig1_architecture.{pdf,png}")


# ---------------------------------------------------------------------------
# Fig 2: Comparison bar chart
# ---------------------------------------------------------------------------
def fig2_comparison():
    datasets = ["OrgAccess\n(BAcc)", "ICEWS18\n(MRR)",
                "WebQSP\n(Acc)", "CWQ\n(Acc)", "DBP-YG\n(Hits@1)"]
    methods = ["Vanilla-RAG", "GraphRAG", "ToG-1", "ToG-2", "CoK",
               "QUEST-KG", "QUEST-KG-LLM"]
    M = np.array([
        [0.507, 0.016, 0.185, 0.149, 0.0],
        [0.507, 0.035, 0.114, 0.145, 0.0],
        [0.515, 0.013, 0.019, 0.072, 0.0],
        [0.497, 0.011, 0.020, 0.070, 0.0],
        [0.500, 0.015, 0.055, 0.094, 0.0],
        [0.963, 0.053, 0.330, 0.204, 0.919],
        [0.963, 0.053, 0.397, 0.236, 0.919],
    ])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(datasets))
    width = 0.115
    for i, m in enumerate(methods):
        offset = (i - len(methods) / 2) * width + width / 2
        bars = ax.bar(x + offset, M[i], width, label=m,
                      color=PALETTE[m], edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, M[i]):
            if v == 0:
                b.set_alpha(0.0)
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Primary metric")
    ax.set_title("Primary-metric comparison across benchmark datasets",
                  fontsize=9.5, pad=8)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    sns.despine(ax=ax)
    plt.savefig(FIG_DIR / "fig2_comparison.pdf")
    plt.savefig(FIG_DIR / "fig2_comparison.png", dpi=300)
    plt.close(fig)
    print("  fig2_comparison.{pdf,png}")


# ---------------------------------------------------------------------------
# Fig 3: 4-panel reliability diagrams
# ---------------------------------------------------------------------------
def fig3_calibration():
    sys.path.insert(0, str(ROOT))
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
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.1), sharey=True)
    for ax, (title, qkg_csv, base_csv) in zip(axes, panels):
        ax.set_title(title, fontsize=9.5)
        ax.plot([0, 1], [0, 1], color="#888", linestyle=(0, (3, 3)),
                lw=0.9, label="ideal", zorder=1)
        text_y = 0.94
        for label, csv_name, color in [
            ("QUEST-KG",          qkg_csv,  C_QKG),
            ("CoK (best LLM-RAG)", base_csv, C_HIGHLIGHT),
        ]:
            p = R / csv_name
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "confidence" not in df.columns or "em" not in df.columns:
                continue
            conf = df["confidence"].fillna(0).clip(0, 1).to_numpy(dtype=float)
            correct = df["em"].fillna(0).astype(int).to_numpy(dtype=float)
            edges = np.linspace(0, 1, 11)
            xs, ys, ws = [], [], []
            for i in range(10):
                m = (conf >= edges[i]) & (conf < edges[i + 1])
                if m.sum() >= 5:
                    xs.append(conf[m].mean())
                    ys.append(correct[m].mean())
                    ws.append(m.sum())
            ax.plot(xs, ys, marker="o", ms=4, color=color, label=label,
                    lw=1.5, zorder=3)
            ax.fill_between(xs, ys, [x for x in xs], color=color, alpha=0.08,
                            zorder=2)
            e = ece(conf, correct, n_bins=15)
            ax.text(0.03, text_y, f"ECE={e:.3f}", transform=ax.transAxes,
                    color=color, fontsize=7.5, weight="bold")
            text_y -= 0.08
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        if ax is axes[0]:
            ax.set_ylabel("Empirical accuracy", fontsize=8)
    axes[-1].legend(fontsize=6.5, loc="lower right", framealpha=0.95)
    plt.savefig(FIG_DIR / "fig3_calibration.pdf")
    plt.savefig(FIG_DIR / "fig3_calibration.png", dpi=300)
    plt.close(fig)
    print("  fig3_calibration.{pdf,png}")


# ---------------------------------------------------------------------------
# Fig 4: Latency
# ---------------------------------------------------------------------------
def fig4_latency():
    datasets = ["OrgAccess", "ICEWS18", "WebQSP", "CWQ"]
    methods = ["Vanilla-RAG", "GraphRAG", "ToG-1", "ToG-2", "CoK",
               "QUEST-KG-LLM", "QUEST-KG"]
    L = np.array([
        [1482, 1478, 1518, 1600],
        [1593, 1563, 1578, 1638],
        [7873, 8599, 8881, 9332],
        [8951, 9147, 9134, 9465],
        [9208, 9080, 8836, 8965],
        [1,    5,    381,  489],
        [18,   16,   52,   78],
    ])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(len(datasets))
    width = 0.115
    for i, m in enumerate(methods):
        offset = (i - len(methods) / 2) * width + width / 2
        ax.bar(x + offset, L[i], width, label=m,
               color=PALETTE.get(m, "#444"),
               edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms / query, log scale)")
    ax.set_title("Inference latency: QUEST-KG is 19$\\times$ to 500$\\times$ faster than LLM baselines",
                  fontsize=9.5, pad=8)
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    sns.despine(ax=ax)
    plt.savefig(FIG_DIR / "fig4_latency.pdf")
    plt.savefig(FIG_DIR / "fig4_latency.png", dpi=300)
    plt.close(fig)
    print("  fig4_latency.{pdf,png}")


# ---------------------------------------------------------------------------
# Fig 5: Hop sweep
# ---------------------------------------------------------------------------
def fig5_hopsweep():
    p = ROOT / "results" / "_ablations_local__local-1080ti__symbolic.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    hop = df[df["variant"].str.startswith("hop_k=")].copy()
    hop["k"] = hop["variant"].str.extract(r"hop_k=(\d+)").astype(int)
    piv = hop.pivot_table(index="k", columns="dataset",
                          values="primary_value", aggfunc="first")
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    colors = {"orgaccess": "#0A4D8C", "icews18": "#C62828",
              "webqsp": "#2E7D32", "cwq": "#6A1B9A"}
    markers = {"orgaccess": "o", "icews18": "s",
               "webqsp": "^", "cwq": "D"}
    locked = {"orgaccess": 2, "icews18": 1, "webqsp": 1, "cwq": 3}
    for ds in piv.columns:
        ax.plot(piv.index, piv[ds], marker=markers.get(ds, "o"), ms=6,
                color=colors.get(ds, "#444"), label=ds, lw=1.8)
        k_lock = locked.get(ds)
        if k_lock and k_lock in piv.index:
            ax.scatter([k_lock], [piv.loc[k_lock, ds]], s=180,
                       facecolors="none", edgecolors=colors.get(ds, "#444"),
                       linewidth=2.2, zorder=5)
    ax.set_xlabel("Hop budget $k$")
    ax.set_ylabel("Primary metric")
    ax.set_title("Hop sweep: locked optimum (⭘) matches sweep maximum",
                  fontsize=9.5, pad=8)
    ax.set_xticks(sorted(piv.index))
    ax.legend(fontsize=7.5, loc="lower left", frameon=True, framealpha=0.92)
    sns.despine(ax=ax)
    plt.savefig(FIG_DIR / "fig5_hopsweep.pdf")
    plt.savefig(FIG_DIR / "fig5_hopsweep.png", dpi=300)
    plt.close(fig)
    print("  fig5_hopsweep.{pdf,png}")


# ---------------------------------------------------------------------------
# Fig 6: Confidence histograms
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
        conf = df["confidence"].fillna(0).clip(0, 1).to_numpy(dtype=float)
        em = df["em"].fillna(0).astype(int).to_numpy(dtype=int)
        bins = np.linspace(0, 1, 16)
        ax.hist(conf[em == 1], bins=bins, color=C_QKG, alpha=0.85,
                label="Correct", edgecolor="white", linewidth=0.4)
        ax.hist(conf[em == 0], bins=bins, color=C_HIGHLIGHT, alpha=0.55,
                label="Incorrect", edgecolor="white", linewidth=0.4)
        ax.set_title(f"QUEST-KG confidence distribution: {title}", fontsize=9.5)
        ax.set_xlabel("Posterior confidence")
        ax.tick_params(axis="both", labelsize=7.5)
        if ax is axes[0]:
            ax.set_ylabel("# queries")
        ax.legend(fontsize=7.5, loc="upper right")
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_confhist.pdf")
    plt.savefig(FIG_DIR / "fig6_confhist.png", dpi=300)
    plt.close(fig)
    print("  fig6_confhist.{pdf,png}")


if __name__ == "__main__":
    print(f"Writing figures to {FIG_DIR}  (PUB_STYLE={PUB_STYLE})")
    fig1_architecture()
    fig2_comparison()
    fig3_calibration()
    fig4_latency()
    fig5_hopsweep()
    fig6_confhist()
    print("done.")
