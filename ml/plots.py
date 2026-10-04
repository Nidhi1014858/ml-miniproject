"""
ml/plots.py  -  Generates diagnostic visualization charts for LinkWise.
Creates tree.png, confusion.png, and importance.png styled to match
the website's dark theme.
"""

import json
import os
from pathlib import Path
import sys

# Add project root to sys.path so config can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file generation
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.tree import plot_tree


# Friendly display names for features in charts
FEATURE_LABELS = {
    "data_type_Fault alert": "Fault alert type",
    "data_type_Housekeeping": "Housekeeping type",
    "data_type_SSTV image": "SSTV image type",
    "data_type_Voice/Data": "Voice/Data type",
    "battery_pct": "Battery level (%)",
    "link_quality": "Link quality",
    "sat_mode": "Satellite mode",
    "size_kb": "Packet size (KB)",
    "pass_time_min": "Pass time left",
}

# Theme palette matching style.css
COLOR_BG = "#111a2e"        # Card background (--bg-card)
COLOR_TEXT = "#f1f5f9"      # Main light text (--text-main)
COLOR_MUTED = "#94a3b8"     # Secondary text (--text-muted)
COLOR_ACCENT = "#38bdf8"    # Accent cyan (--accent-cyan)
COLOR_ACCENT_DARK = "#0284c7"  # Deep blue (--accent-blue)
COLOR_BORDER = "#22304d"    # Card border highlight


def clean_feature_name(name: str) -> str:
    """Returns a clean, readable name for a feature."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    return name.replace("data_type_", "").replace("_", " ").capitalize()


def generate_tree_plot(model, feature_names, output_path: str):
    """
    Renders the full decision tree with readable names, filled colors,
    and rounded boxes on the dark theme.
    """
    readable_names = [clean_feature_name(f) for f in feature_names]

    fig, ax = plt.subplots(figsize=(26, 13), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    plot_tree(
        model,
        feature_names=readable_names,
        class_names=config.PRIORITIES,
        filled=True,
        rounded=True,
        precision=1,
        fontsize=8,
        ax=ax,
    )

    # Style connector lines to be visible against dark background
    for line in ax.lines:
        line.set_color(COLOR_MUTED)
        line.set_linewidth(1.2)

    ax.set_title(
        "Decision Tree Rules",
        fontsize=18,
        color=COLOR_TEXT,
        pad=20,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        facecolor=COLOR_BG,
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_confusion_matrix_plot(matrix_data, labels, output_path: str):
    """
    Plots the confusion matrix showing counts for each cell.
    Rows = Actual priority, Columns = Predicted priority.
    """
    cm = np.array(matrix_data)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # Gradient colormap from dark card surface to vibrant accent
    cmap = LinearSegmentedColormap.from_list(
        "linkwise_cyan",
        ["#15223c", "#0284c7", COLOR_ACCENT],
    )

    im = ax.imshow(cm, cmap=cmap, interpolation="nearest")

    # Add numeric count inside every cell
    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            # Use dark text on bright cells, light text on dark cells
            text_color = "#090d16" if val > (max_val * 0.55) else COLOR_TEXT
            ax.text(
                j, i,
                f"{val}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold",
            )

    # Configure axes with readable labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color=COLOR_TEXT, fontsize=11, fontweight="600")
    ax.set_yticklabels(labels, color=COLOR_TEXT, fontsize=11, fontweight="600")

    ax.set_xlabel("Predicted priority", color=COLOR_TEXT, fontsize=12, labelpad=12, fontweight="600")
    ax.set_ylabel("Actual priority", color=COLOR_TEXT, fontsize=12, labelpad=12, fontweight="600")
    ax.set_title("Confusion Matrix", color=COLOR_TEXT, fontsize=15, pad=16, fontweight="bold")

    # Clean borders
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
        spine.set_linewidth(1.2)

    ax.tick_params(colors=COLOR_MUTED)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=COLOR_MUTED)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COLOR_MUTED)
    cbar.outline.set_edgecolor(COLOR_BORDER)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        facecolor=COLOR_BG,
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_feature_importance_plot(importance_data, output_path: str):
    """
    Plots a horizontal bar chart of feature importances sorted biggest first.
    """
    # Sort biggest first
    sorted_items = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)

    # For horizontal bar charts, matplotlib plots from bottom to top,
    # so we reverse the list so the highest appears at the top.
    names = [clean_feature_name(k) for k, v in reversed(sorted_items)]
    values = [v for k, v in reversed(sorted_items)]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    y_positions = range(len(names))
    bars = ax.barh(
        y_positions,
        values,
        height=0.62,
        color=COLOR_ACCENT,
        edgecolor=COLOR_ACCENT_DARK,
        linewidth=1,
    )

    # Add numeric percentage label on each bar
    for bar in bars:
        width = bar.get_width()
        if width > 0.001:
            ax.text(
                width + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{width * 100:.1f}%",
                va="center",
                fontsize=9.5,
                color=COLOR_TEXT,
                fontweight="600",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, color=COLOR_TEXT, fontsize=10)
    ax.set_xlabel("Importance Score", color=COLOR_TEXT, fontsize=11, labelpad=10, fontweight="600")
    ax.set_title("Feature Importance", color=COLOR_TEXT, fontsize=15, pad=16, fontweight="bold")

    # Style spines and ticks
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLOR_BORDER)
    ax.spines["bottom"].set_color(COLOR_BORDER)

    ax.tick_params(colors=COLOR_MUTED)
    ax.grid(axis="x", linestyle="--", alpha=0.18, color=COLOR_MUTED)

    # Give room for text labels
    max_val = max(values) if values else 0.3
    ax.set_xlim(0, max_val * 1.25)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        facecolor=COLOR_BG,
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    print("=" * 65)
    print("LinkWise - Generating Diagnostic Visualizations")
    print("=" * 65)

    # Ensure output plots directory exists
    plots_dir = Path(config.PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model and feature order
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_PATH}. Run ml/train.py first.")
    model_data = joblib.load(config.MODEL_PATH)
    model = model_data["model"]
    feature_names = model_data["feature_names"]

    # 2. Load metrics
    if not os.path.exists(config.METRICS_PATH):
        raise FileNotFoundError(f"Metrics file not found at {config.METRICS_PATH}. Run ml/train.py first.")
    with open(config.METRICS_PATH, "r") as f:
        metrics = json.load(f)

    # File paths
    tree_path = str(plots_dir / "tree.png")
    confusion_path = str(plots_dir / "confusion.png")
    importance_path = str(plots_dir / "importance.png")

    # Generate plots
    print("Generating decision tree visualization...")
    generate_tree_plot(model, feature_names, tree_path)

    print("Generating confusion matrix plot...")
    generate_confusion_matrix_plot(
        metrics["confusion_matrix"],
        metrics.get("labels", config.PRIORITIES),
        confusion_path,
    )

    print("Generating feature importance bar chart...")
    generate_feature_importance_plot(
        metrics["feature_importance"],
        importance_path,
    )

    print("\nSaved files:")
    print(f"  [OK] {tree_path}")
    print(f"  [OK] {confusion_path}")
    print(f"  [OK] {importance_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
