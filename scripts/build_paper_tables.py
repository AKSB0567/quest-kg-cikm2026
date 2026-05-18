"""Assemble paper-ready tables (LaTeX + markdown) from per-cell JSONs.

Tables produced (paper/tables/*.tex + *.md):
  T2_headline.tex    — primary metric per (method, dataset). Bold = QUEST-KG.
  T4_matched.tex     — matched-condition (same encoder + LLM): EM/MRR/BAcc + latency.
  T_calibration.tex  — ECE / Brier / AURC per (method, dataset).
  T_ablation.tex     — full / -bidir / -rel_bias / -answer_rescore on QUEST-KG.
  T_hopsweep.tex     — k ∈ {1,2,3,4} sweep for QUEST-KG.
  T_bootstrap.tex    — paired bootstrap CIs vs baselines.

Reads:
  results/*.json  — per-cell summaries (skips _* files)
  results/_ablations_local__*.csv
  results/_l4_*.csv (if Colab notebook already ran)
  results/_calibration_*.csv
  results/_paired_bootstrap_*.csv

Usage:
  .venv312/Scripts/python scripts/build_paper_tables.py
  .venv312/Scripts/python scripts/build_paper_tables.py --tag <tag>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES_DIR = ROOT / "paper" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


PRIMARY_METRIC_BY_DATASET = {
    "orgaccess": "balanced_accuracy",
    "icews18": "mrr",
    "webqsp": "accuracy",
    "cwq": "accuracy",
}
PRIMARY_LABEL = {
    "balanced_accuracy": "BAcc",
    "mrr": "MRR",
    "accuracy": "Acc",
}
METHOD_DISPLAY = {
    "quest_kg":     r"\textbf{QUEST-KG}",
    "quest_kg_llm": r"\textbf{QUEST-KG-LLM}",
    "vanilla_rag":  "Vanilla-RAG",
    "graphrag":     "GraphRAG",
    "tog1":         "ToG-1",
    "tog2":         "ToG-2",
    "cok":          "CoK",
}
METHOD_ORDER = ["quest_kg", "quest_kg_llm", "vanilla_rag", "graphrag",
                "tog1", "tog2", "cok"]
DATASETS = ["orgaccess", "icews18", "webqsp", "cwq"]


def load_cells(tag_filter: str | None = None) -> pd.DataFrame:
    rows = []
    for jp in sorted(RESULTS.glob("*.json")):
        if jp.name.startswith("_"):
            continue
        if tag_filter and tag_filter not in jp.name:
            continue
        try:
            d = json.loads(jp.read_text())
        except Exception:
            continue
        parts = jp.stem.split("__")
        method = parts[0]
        dataset = parts[1] if len(parts) > 1 else None
        tag = "__".join(parts[2:]) if len(parts) > 2 else ""
        rows.append({
            "method": method, "dataset": dataset, "tag": tag, "n": d.get("n", 0),
            "exact_match": d.get("exact_match"),
            "accuracy": d.get("accuracy", d.get("exact_match")),
            "balanced_accuracy": d.get("balanced_accuracy"),
            "pos_f1": d.get("pos_f1"),
            "mrr": d.get("mrr"),
            "hits_at_1": d.get("hits_at_1"),
            "hits_at_10": d.get("hits_at_10"),
            "token_f1": d.get("token_f1"),
            "mean_latency_ms": d.get("mean_latency_ms"),
            "primary_metric": d.get("primary_metric"),
            "primary_value": d.get("primary_value"),
            "wallclock_s": d.get("wallclock_s"),
        })
    return pd.DataFrame(rows)


def headline_table(df: pd.DataFrame, out_stem: str) -> None:
    """Table 2: primary metric per (method, dataset). One row per method."""
    # Pick the "best" tag per (method, dataset) by max n (most reliable run)
    best = (df.sort_values("n", ascending=False)
              .drop_duplicates(["method", "dataset"], keep="first"))
    pivot = best.pivot_table(index="method", columns="dataset",
                              values="primary_value", aggfunc="first")
    pivot = pivot.reindex([m for m in METHOD_ORDER if m in pivot.index])

    md = ["| Method | " + " | ".join(
              f"{ds} ({PRIMARY_LABEL.get(PRIMARY_METRIC_BY_DATASET[ds], '?')})"
              for ds in DATASETS if ds in pivot.columns) + " |",
          "|" + "---|" * (1 + len([ds for ds in DATASETS if ds in pivot.columns]))]
    for m in pivot.index:
        cells = []
        for ds in DATASETS:
            if ds not in pivot.columns:
                continue
            v = pivot.loc[m, ds]
            cells.append(f"{v:.3f}" if pd.notna(v) else "-")
        md.append(f"| {METHOD_DISPLAY.get(m, m)} | " + " | ".join(cells) + " |")

    (TABLES_DIR / f"{out_stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # LaTeX
    tex_cols = "l" + "c" * len([ds for ds in DATASETS if ds in pivot.columns])
    tex_header = " & ".join(
        [r"\textbf{Method}"] +
        [rf"{ds.upper()} (\textbf{{{PRIMARY_LABEL.get(PRIMARY_METRIC_BY_DATASET[ds], '?')}}})"
         for ds in DATASETS if ds in pivot.columns])
    tex_rows = []
    for m in pivot.index:
        cells = []
        for ds in DATASETS:
            if ds not in pivot.columns:
                continue
            v = pivot.loc[m, ds]
            cells.append(f"{v:.3f}" if pd.notna(v) else "--")
        tex_rows.append(METHOD_DISPLAY.get(m, m) + " & " + " & ".join(cells) + r" \\")

    tex = [
        r"\begin{tabular}{" + tex_cols + r"}",
        r"\toprule",
        tex_header + r" \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (TABLES_DIR / f"{out_stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {TABLES_DIR / out_stem}.md and .tex ({len(pivot)} methods)")


def latency_table(df: pd.DataFrame, out_stem: str) -> None:
    """Mean latency (ms) per (method, dataset)."""
    best = (df.sort_values("n", ascending=False)
              .drop_duplicates(["method", "dataset"], keep="first"))
    pivot = best.pivot_table(index="method", columns="dataset",
                              values="mean_latency_ms", aggfunc="first")
    pivot = pivot.reindex([m for m in METHOD_ORDER if m in pivot.index])
    pivot.to_csv(TABLES_DIR / f"{out_stem}.csv")
    md = ["| Method | " + " | ".join(ds for ds in DATASETS if ds in pivot.columns) + " |",
          "|" + "---|" * (1 + len([ds for ds in DATASETS if ds in pivot.columns]))]
    for m in pivot.index:
        cells = [f"{pivot.loc[m, ds]:.0f}" if ds in pivot.columns
                                              and pd.notna(pivot.loc[m, ds])
                                          else "-"
                 for ds in DATASETS]
        md.append(f"| {METHOD_DISPLAY.get(m, m)} | " + " | ".join(cells) + " |")
    (TABLES_DIR / f"{out_stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {TABLES_DIR / out_stem}.md (latency ms)")


def calibration_table(out_stem: str) -> None:
    """ECE / AURC per (method, dataset) from _calibration_*.csv files."""
    cal_csvs = list(RESULTS.glob("_calibration_*.csv"))
    if not cal_csvs:
        print(f"  skip {out_stem}: no _calibration_*.csv")
        return
    df = pd.concat([pd.read_csv(p) for p in cal_csvs], ignore_index=True)
    # Keep best (lowest ECE) per (method, dataset)
    df = df.sort_values("ECE").drop_duplicates(["method", "dataset"], keep="first")
    out = df[["method", "dataset", "n", "ECE", "Brier", "NLL", "AURC",
              "mean_conf", "accuracy"]].sort_values(["dataset", "ECE"])
    out.to_csv(TABLES_DIR / f"{out_stem}.csv", index=False)
    md_lines = ["| Method | Dataset | n | ECE | Brier | NLL | AURC | mean_conf | acc |",
                "|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        md_lines.append(
            f"| {METHOD_DISPLAY.get(r['method'], r['method'])} | {r['dataset']} | "
            f"{int(r['n'])} | {r['ECE']:.4f} | {r['Brier']:.4f} | "
            f"{r['NLL']:.4f} | {r['AURC']:.4f} | {r['mean_conf']:.3f} | {r['accuracy']:.3f} |"
        )
    (TABLES_DIR / f"{out_stem}.md").write_text("\n".join(md_lines) + "\n")
    print(f"  wrote {TABLES_DIR / out_stem}.md ({len(out)} rows)")


def ablation_table(out_stem: str) -> None:
    """Component removal + hop sweep from _ablations_local__*.csv."""
    abl_csvs = list(RESULTS.glob("_ablations_local__*.csv"))
    if not abl_csvs:
        print(f"  skip {out_stem}: no _ablations_local__*.csv yet")
        return
    df = pd.concat([pd.read_csv(p) for p in abl_csvs], ignore_index=True)
    # Component removal (variant names: full, no_*)
    comp = df[~df["variant"].str.startswith("hop_k=")]
    if len(comp):
        pivot = comp.pivot_table(index="variant", columns="dataset",
                                  values="primary_value", aggfunc="first")
        pivot.to_csv(TABLES_DIR / f"{out_stem}_components.csv")
        md = ["| Variant | " + " | ".join(ds for ds in DATASETS if ds in pivot.columns) + " |",
              "|" + "---|" * (1 + len([ds for ds in DATASETS if ds in pivot.columns]))]
        order = ["full", "no_bidir", "no_rel_bias", "no_answer_rescore"]
        for v in order:
            if v not in pivot.index:
                continue
            cells = [f"{pivot.loc[v, ds]:.3f}" if ds in pivot.columns
                                                    and pd.notna(pivot.loc[v, ds])
                                                else "-"
                     for ds in DATASETS]
            md.append(f"| {v} | " + " | ".join(cells) + " |")
        (TABLES_DIR / f"{out_stem}_components.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"  wrote {TABLES_DIR / out_stem}_components.md ({len(order)} variants)")

    # Hop sweep
    hop = df[df["variant"].str.startswith("hop_k=")]
    if len(hop):
        hop = hop.copy()
        hop["k"] = hop["variant"].str.extract(r"hop_k=(\d+)").astype(int)
        pivot = hop.pivot_table(index="k", columns="dataset",
                                 values="primary_value", aggfunc="first")
        pivot.to_csv(TABLES_DIR / f"{out_stem}_hopsweep.csv")
        md = ["| k | " + " | ".join(ds for ds in DATASETS if ds in pivot.columns) + " |",
              "|" + "---|" * (1 + len([ds for ds in DATASETS if ds in pivot.columns]))]
        for k in sorted(pivot.index):
            cells = [f"{pivot.loc[k, ds]:.3f}" if ds in pivot.columns
                                                   and pd.notna(pivot.loc[k, ds])
                                               else "-"
                     for ds in DATASETS]
            md.append(f"| {k} | " + " | ".join(cells) + " |")
        (TABLES_DIR / f"{out_stem}_hopsweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"  wrote {TABLES_DIR / out_stem}_hopsweep.md ({len(pivot)} k values)")


def bootstrap_table(out_stem: str) -> None:
    boot_csvs = list(RESULTS.glob("_paired_bootstrap_*.csv"))
    if not boot_csvs:
        print(f"  skip {out_stem}: no _paired_bootstrap_*.csv")
        return
    df = pd.concat([pd.read_csv(p) for p in boot_csvs], ignore_index=True)
    df.to_csv(TABLES_DIR / f"{out_stem}.csv", index=False)
    md = ["| Dataset | QUEST-KG variant | vs baseline | Δ | 95% CI | p | sig |",
          "|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["dataset", "ours", "baseline"]).iterrows():
        md.append(
            f"| {r['dataset']} | {METHOD_DISPLAY.get(r['ours'], r['ours'])} | "
            f"{METHOD_DISPLAY.get(r['baseline'], r['baseline'])} | "
            f"{r['delta']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
            f"{r['p_two_sided']:.3f} | {r['signif']} |"
        )
    (TABLES_DIR / f"{out_stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {TABLES_DIR / out_stem}.md ({len(df)} comparisons)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="Filter cells by tag substring")
    args = ap.parse_args()

    print(f"Loading cells (tag filter: {args.tag or '<none>'}) ...")
    df = load_cells(args.tag or None)
    if df.empty:
        print("no cells found")
        return
    print(f"  loaded {len(df)} cells")

    print("\nBuilding tables in", TABLES_DIR)
    headline_table(df, "T2_headline")
    latency_table(df, "T_latency")
    calibration_table("T_calibration")
    ablation_table("T_ablation")
    bootstrap_table("T_bootstrap")
    print("\nDone.")


if __name__ == "__main__":
    main()
