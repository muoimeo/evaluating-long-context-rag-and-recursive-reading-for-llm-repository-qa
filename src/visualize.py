"""
visualize.py — Claim-Driven Research Visualizations

Reads scored JSON results and generates exactly 9 charts,
each serving one specific research claim.

Usage:
  python src/visualize.py --scored results/scored/ --output results/plots/
  python src/visualize.py --scored results/scored/ --analysis results/analysis/ --output results/plots/

Charts generated:
  1. avg_scores_table.png         — Overall accuracy/citation comparison (claim: overall ranking)
  2. latency_boxplot.png          — Latency distribution (claim: RLM costs 3x latency)
  3. accuracy_by_dataset.png      — Accuracy per dataset (claim: RAG strong on retrieval)
  4. citation_by_dataset.png      — Citation F1 per dataset (claim: LongCtx fails universally)
  5. accuracy_by_reasoning_type.png — Acc per reasoning type (claim: RLM wins multi-hop)
  6. token_cost_breakdown.png     — Input/output/calls (claim: RLM cost is call-driven)
  7. error_breakdown.png          — Error types stacked (claim: each method fails differently)
  8. pareto_frontier.png          — Accuracy vs Latency (claim: RAG practical, RLM quality)
  9. citation_layer_comparison.png — File precision/recall/line-iou (claim: citation 2-layer view)
"""
import os
import sys
import json
import glob
import argparse
from typing import Dict, List, Any

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import seaborn as sns
except ImportError:
    print("Missing libraries. Run: pip install pandas matplotlib seaborn numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

METHOD_LABELS = {"A": "Long-Context", "B": "RAG", "C": "Recursive (RLM)"}
METHOD_ORDER  = ["A", "B", "C"]
PALETTE       = {"A": "#E76F51", "B": "#2A9D8F", "C": "#457B9D"}

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 150, "font.family": "sans-serif"})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scored(scored_dir: str) -> pd.DataFrame:
    records = []
    for path in glob.glob(os.path.join(scored_dir, "scored_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records.extend(data)
    if not records:
        raise FileNotFoundError(f"No scored_*.json found in {scored_dir}")
    df = pd.DataFrame(records)
    df["Method"] = df["method"].map(lambda x: METHOD_LABELS.get(x, x))
    df["method_order"] = df["method"].map({"A": 0, "B": 1, "C": 2})
    df = df.sort_values("method_order")
    return df


def load_analysis(analysis_dir: str) -> Dict:
    if not analysis_dir or not os.path.exists(analysis_dir):
        return {}
    files = sorted(glob.glob(os.path.join(analysis_dir, "analysis_*.json")))
    if not files:
        return {}
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def savefig(fig, path: str):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def method_colors(methods):
    return [PALETTE.get(m, "#888") for m in methods]


# ---------------------------------------------------------------------------
# 9 Charts
# ---------------------------------------------------------------------------

def chart1_avg_scores(df: pd.DataFrame, output_dir: str):
    """Chart 1: Grouped bar — Overall accuracy and citation scores."""
    metrics = ["accuracy_primary", "citation_file_f1", "citation_support_score"]
    labels  = ["Accuracy (Primary)", "Citation F1 (File)", "Citation Support Score"]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(METHOD_ORDER))
    width = 0.25

    for i, (m, label) in enumerate(zip(metrics, labels)):
        means = [df[df["method"] == mth][m].mean() for mth in METHOD_ORDER]
        bars = ax.bar(x + (i - 1) * width, means, width, label=label, alpha=0.87)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0 – 1)")
    ax.set_title("Overall Accuracy and Citation Scores by Method", pad=15)
    ax.legend(loc="upper right")
    savefig(fig, os.path.join(output_dir, "1_avg_scores_comparison.png"))


def chart2_latency_boxplot(df: pd.DataFrame, output_dir: str):
    """Chart 2: Latency boxplot with median + p95 annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = [METHOD_LABELS[m] for m in METHOD_ORDER]
    colors = [PALETTE[m] for m in METHOD_ORDER]

    bplot = ax.boxplot(
        [df[df["method"] == m]["latency_sec"].dropna().values for m in METHOD_ORDER],
        labels=order, patch_artist=True, showfliers=True, whis=[5, 95]
    )
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Annotate median and p95
    for i, m in enumerate(METHOD_ORDER, start=1):
        vals = df[df["method"] == m]["latency_sec"].dropna().values
        if len(vals):
            med = float(np.median(vals))
            p95 = float(np.percentile(vals, 95))
            ax.text(i, med + 0.5, f"med={med:.1f}s", ha="center", fontsize=8, color="navy")
            ax.text(i, p95 + 0.5, f"p95={p95:.1f}s", ha="center", fontsize=8, color="darkred")

    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Response Latency Distribution by Method", pad=15)
    savefig(fig, os.path.join(output_dir, "2_latency_boxplot.png"))


def chart3_accuracy_by_dataset(df: pd.DataFrame, output_dir: str):
    """Chart 3: Accuracy by dataset category."""
    fig, ax = plt.subplots(figsize=(12, 6))
    datasets = sorted(df["dataset"].dropna().unique())
    x = np.arange(len(datasets))
    width = 0.25

    for i, m in enumerate(METHOD_ORDER):
        sub = df[df["method"] == m]
        means = [sub[sub["dataset"] == d]["accuracy_primary"].mean() for d in datasets]
        bars = ax.bar(x + (i - 1) * width, means, width,
                      label=METHOD_LABELS[m], color=PALETTE[m], alpha=0.85)
        for bar, v in zip(bars, means):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in datasets])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (Primary)")
    ax.set_title("Accuracy by Dataset Category", pad=15)
    ax.legend()
    savefig(fig, os.path.join(output_dir, "3_accuracy_by_dataset.png"))


def chart4_citation_by_dataset(df: pd.DataFrame, output_dir: str):
    """Chart 4: Citation F1 by dataset — shows Long-Context fails universally."""
    fig, ax = plt.subplots(figsize=(12, 6))
    datasets = sorted(df["dataset"].dropna().unique())
    x = np.arange(len(datasets))
    width = 0.25

    for i, m in enumerate(METHOD_ORDER):
        sub = df[df["method"] == m]
        means = [sub[sub["dataset"] == d]["citation_file_f1"].mean() for d in datasets]
        bars = ax.bar(x + (i - 1) * width, means, width,
                      label=METHOD_LABELS[m], color=PALETTE[m], alpha=0.85)
        for bar, v in zip(bars, means):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in datasets])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Citation File F1")
    ax.set_title("Citation F1 by Dataset Category\n(Long-Context fails at grounding universally)", pad=15)
    ax.legend()
    savefig(fig, os.path.join(output_dir, "4_citation_f1_by_dataset.png"))


def chart5_accuracy_by_reasoning_type(df: pd.DataFrame, output_dir: str):
    """Chart 5: Accuracy by reasoning type — shows RLM advantage on complex types."""
    col = "expected_reasoning_type"
    if col not in df.columns or df[col].isna().all():
        print("  Skipping chart5: expected_reasoning_type not in data")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    rtypes = sorted(df[col].dropna().unique())
    x = np.arange(len(rtypes))
    width = 0.25

    for i, m in enumerate(METHOD_ORDER):
        sub = df[df["method"] == m]
        means = [sub[sub[col] == rt]["accuracy_primary"].mean() for rt in rtypes]
        bars = ax.bar(x + (i - 1) * width, means, width,
                      label=METHOD_LABELS[m], color=PALETTE[m], alpha=0.85)
        for bar, v in zip(bars, means):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([rt.replace("_", "\n") for rt in rtypes], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (Primary)")
    ax.set_title("Accuracy by Reasoning Type\n(RLM advantage grows with complexity)", pad=15)
    ax.legend()
    savefig(fig, os.path.join(output_dir, "5_accuracy_by_reasoning_type.png"))


def chart6_token_cost(df: pd.DataFrame, output_dir: str):
    """Chart 6: Token usage + num_calls — shows RLM cost is call-driven."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: token breakdown
    ax = axes[0]
    x = np.arange(len(METHOD_ORDER))
    inp = [df[df["method"] == m]["input_tokens"].mean() for m in METHOD_ORDER]
    out = [df[df["method"] == m]["output_tokens"].mean() for m in METHOD_ORDER]
    bars1 = ax.bar(x, inp, label="Input Tokens", color="#457B9D", alpha=0.85)
    bars2 = ax.bar(x, out, bottom=inp, label="Output Tokens", color="#E9C46A", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Tokens (mean)")
    ax.set_title("Token Usage per Query")
    ax.legend()

    # Right: model calls
    ax2 = axes[1]
    calls = [df[df["method"] == m]["model_calls"].mean() for m in METHOD_ORDER]
    colors = [PALETTE[m] for m in METHOD_ORDER]
    bars = ax2.bar([METHOD_LABELS[m] for m in METHOD_ORDER], calls, color=colors, alpha=0.85)
    for bar, v in zip(bars, calls):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Mean LLM Calls per Query")
    ax2.set_title("LLM Calls per Query\n(RLM cost is call-driven)")

    fig.suptitle("Token Cost Breakdown by Method", fontsize=14, y=1.02)
    plt.tight_layout()
    savefig(fig, os.path.join(output_dir, "6_token_cost_breakdown.png"))


def chart7_error_breakdown(analysis: Dict, output_dir: str):
    """Chart 7: Stacked bar — error types per method."""
    if not analysis:
        print("  Skipping chart7: no analysis data. Run analyze.py first.")
        return

    error_cats = ["retrieval_miss", "insufficient_evidence", "reasoning_failure",
                  "citation_failure", "over_generation", "trace_inefficiency", "success"]
    cat_colors  = ["#E63946", "#F4A261", "#E76F51", "#F4D35E", "#EE9B00", "#94D2BD", "#52B788"]

    methods_in = [m for m in METHOD_ORDER if m in analysis]
    if not methods_in:
        print("  Skipping chart7: no method data in analysis.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods_in))
    bottoms = np.zeros(len(methods_in))

    for cat, color in zip(error_cats, cat_colors):
        vals = []
        for m in methods_in:
            eb = analysis.get(m, {}).get("error_breakdown", {})
            n = sum(analysis.get(m, {}).get("error_breakdown", {}).values()) or 1
            vals.append(eb.get(cat, 0) / n * 100)
        ax.bar(x, vals, bottom=bottoms, label=cat.replace("_", " ").title(),
               color=color, alpha=0.87)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods_in])
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of Samples")
    ax.set_title("Error Type Breakdown by Method\n(Each method fails differently)", pad=15)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    savefig(fig, os.path.join(output_dir, "7_error_breakdown.png"))


def chart8_pareto_frontier(df: pd.DataFrame, output_dir: str):
    """Chart 8: Accuracy vs Latency scatter — Pareto efficiency frontier."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        sc = ax.scatter(
            sub["latency_sec"], sub["accuracy_primary"],
            c=[PALETTE[m]] * len(sub),
            s=sub["citation_file_f1"] * 300 + 40,
            alpha=0.65, edgecolors="white", linewidths=0.5,
            label=None
        )

    # Method-level means as large labeled points
    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        mx, my = sub["latency_sec"].mean(), sub["accuracy_primary"].mean()
        ax.scatter(mx, my, c=PALETTE[m], s=350, marker="D",
                   edgecolors="black", linewidths=1.5, zorder=5)
        ax.annotate(METHOD_LABELS[m], (mx, my), textcoords="offset points",
                    xytext=(8, 5), fontsize=10, fontweight="bold", color=PALETTE[m])

    ax.set_xlabel("Mean Latency (seconds)")
    ax.set_ylabel("Accuracy (Primary)")
    ax.set_title("Accuracy vs Latency Trade-off\n(Bubble size ∝ Citation F1)", pad=15)
    # Legend for method colors
    patches = [mpatches.Patch(color=PALETTE[m], label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    ax.legend(handles=patches, title="Method")
    savefig(fig, os.path.join(output_dir, "8_pareto_frontier.png"))


def chart9_citation_layers(df: pd.DataFrame, output_dir: str):
    """Chart 9: Citation metric layers — file precision/recall vs line IoU."""
    metrics = ["citation_file_precision", "citation_file_recall", "citation_file_f1", "line_iou", "citation_support_score"]
    labels  = ["File Precision", "File Recall", "File F1", "Line IoU", "Support Score"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics))
    width = 0.25

    for i, m in enumerate(METHOD_ORDER):
        sub = df[df["method"] == m]
        means = [sub[col].mean() for col in metrics]
        bars = ax.bar(x + (i - 1) * width, means, width,
                      label=METHOD_LABELS[m], color=PALETTE[m], alpha=0.85)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Citation Quality — Two-Layer View\n(File-level → Evidence Localization)", pad=15)
    # Separator line between layers
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.6)
    ax.text(1, 0.97, "Layer 1: File Coverage", ha="center", fontsize=9, color="gray")
    ax.text(3.5, 0.97, "Layer 2: Localization", ha="center", fontsize=9, color="gray")
    ax.legend()
    savefig(fig, os.path.join(output_dir, "9_citation_layer_comparison.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate 9 claim-driven research charts")
    parser.add_argument("--scored", type=str, default="results/scored",
                        help="Directory with scored_*.json files")
    parser.add_argument("--analysis", type=str, default="results/analysis",
                        help="Directory with analysis_*.json (for error breakdown chart)")
    parser.add_argument("--output", type=str, default="results/plots")
    args = parser.parse_args()

    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    os.makedirs(args.output, exist_ok=True)

    print("Loading scored data...")
    df = load_scored(args.scored)
    print(f"Loaded {len(df)} records across {df['method'].nunique()} methods.")

    analysis = load_analysis(args.analysis)

    print("\nGenerating 9 charts...")
    chart1_avg_scores(df, args.output)
    chart2_latency_boxplot(df, args.output)
    chart3_accuracy_by_dataset(df, args.output)
    chart4_citation_by_dataset(df, args.output)
    chart5_accuracy_by_reasoning_type(df, args.output)
    chart6_token_cost(df, args.output)
    chart7_error_breakdown(analysis, args.output)
    chart8_pareto_frontier(df, args.output)
    chart9_citation_layers(df, args.output)

    print(f"\nAll 9 charts saved to {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
