# src/evaluation/plots.py
"""
Reusable plotting utilities for evaluation and explainability.

Includes:
- Grouped bar plots comparing metric scores across dataset variants.
- Violin plots showing score distributions.
All save paths are resolved using src.utils.resolve_path for project consistency.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from src.utils import resolve_path


def _make_figure_dir(variant: str) -> Path:
    """Create and return timestamped plot directory for a given dataset variant."""
    timestamp = datetime.now().strftime("%Y%m%d")
    fig_dir = resolve_path(f"results/plots/{variant}_{timestamp}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_grouped_bars(metric_long, available_metrics, variant: str):
    """
    Create grouped bar plots comparing metric scores across dataset variants.

    Args:
        metric_long (pd.DataFrame): Long-format metrics DataFrame (Classifier, Metric, Score, Variant).
        available_metrics (list): List of metric names to plot.
        variant (str): Dataset variant (e.g., 'w2v_optimized_radiology').
    Returns:
        Path: Directory where figures were saved.
    """
    palette = {
        'original_baseline_smote': '#2ca02c',
        'w2v_radiology_baseline_smote': '#d62728',
        'original_baseline': '#1f77b4',
        'w2v_radiology_baseline': '#ff7f0e',
    }
    hue_order = [
        'original_baseline_smote',
        'w2v_radiology_baseline_smote',
        'original_baseline',
        'w2v_radiology_baseline',
    ]

    fig_root = _make_figure_dir(variant)

    for metric in available_metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=metric_long.query("Metric == @metric"),
            x='Classifier', y='Score', hue='Variant',
            hue_order=hue_order, palette=palette
        )
        plt.title(f'{metric} Comparison Across Datasets', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric)
        plt.xlabel('Classifier')
        plt.legend(title='Variant', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path = fig_root / f"{metric.lower()}_comparison_bar.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved → {save_path}")

    return fig_root


def plot_violin_distributions(metric_long, available_metrics, variant: str):
    """
    Create violin plots showing distribution of metric scores across dataset variants.

    Args:
        metric_long (pd.DataFrame): Long-format metrics DataFrame (Classifier, Metric, Score, Variant).
        available_metrics (list): List of metric names to plot.
        variant (str): Dataset variant (e.g., 'w2v_optimized_radiology').
    Returns:
        Path: Directory where figures were saved.
    """
    palette = {
        'original_baseline_smote': '#2ca02c',
        'w2v_radiology_baseline_smote': '#d62728',
        'original_baseline': '#1f77b4',
        'w2v_radiology_baseline': '#ff7f0e',
    }
    hue_order = [
        'original_baseline_smote',
        'w2v_radiology_baseline_smote',
        'original_baseline',
        'w2v_radiology_baseline',
    ]
    variant_labels = {
        'original_baseline_smote': 'Original + SMOTE',
        'w2v_radiology_baseline_smote': 'W2V + SMOTE',
        'original_baseline': 'Original',
        'w2v_radiology_baseline': 'W2V',
    }

    fig_root = _make_figure_dir(variant)

    for metric in available_metrics:
        plt.figure(figsize=(9, 6))
        ax = sns.violinplot(
            data=metric_long.query("Metric == @metric"),
            x='Variant', y='Score', hue='Variant',
            order=hue_order, hue_order=hue_order,
            palette=palette, inner='quartile', legend=False
        )
        ax.set_xticklabels([variant_labels[v] for v in hue_order])
        plt.title(f'{metric} Score Distribution Across Variants', fontsize=16)
        plt.ylabel(metric)
        plt.xlabel('')
        plt.tight_layout()

        save_path = fig_root / f"{metric.lower()}_comparison_violin.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🎻 Saved → {save_path}")

    return fig_root
