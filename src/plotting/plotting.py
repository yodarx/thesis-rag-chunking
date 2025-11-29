import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METRICS_TO_PLOT: list[str] = [
    "mrr",
    "map",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "f1_score_at_k",
]

METRIC_DISPLAY_NAMES: dict[str, str] = {
    "mrr": "Mean Reciprocal Rank (MRR)",
    "map": "Mean Average Precision (MAP)",
    "ndcg_at_k": "Normalized Discounted Cumulative Gain (NDCG@5)",
    "precision_at_k": "Precision@5",
    "recall_at_k": "Recall@5",
    "f1_score_at_k": "F1-Score@5",
}


def visualize_and_save_results(summary_df: pd.DataFrame, output_dir: str, timestamp: str) -> None:
    """
    Generiert und speichert Plots für alle wichtigen Retrieval-Metriken.
    """
    print(f"\nGenerating and saving plots in directory: {output_dir}")
    sns.set_theme(style="whitegrid")

    for metric in METRICS_TO_PLOT:
        if metric not in summary_df.columns:
            print(f"Warning: Metric '{metric}' not found in summary. Skipping plot.")
            continue

        display_name: str = METRIC_DISPLAY_NAMES.get(metric, metric)

        _create_and_save_bar_plot(
            df=summary_df,
            metric=metric,
            display_name=display_name,
            output_dir=output_dir,
            timestamp=timestamp,
        )

    print("\nAll plots were successfully saved.")


def _create_and_save_bar_plot(
    df: pd.DataFrame, metric: str, display_name: str, output_dir: str, timestamp: str
) -> None:
    """Erstellt und speichert ein Balkendiagramm für eine Metrik."""
    plt.figure(figsize=(12, 7))
    bar_plot = sns.barplot(
        data=df.sort_values(metric, ascending=False),
        x=metric,
        y="experiment",
        hue="experiment",
        palette="viridis",
        legend=False,
    )
    plt.title(f"Performance Comparison: {display_name}", fontsize=16, pad=20)
    plt.xlabel(display_name, fontsize=12)
    plt.ylabel("Chunking Strategy", fontsize=12)

    for container in bar_plot.containers:
        bar_plot.bar_label(container, fmt="%.2f", label_type="edge")

    plt.tight_layout()
    plot_path: str = os.path.join(output_dir, f"{timestamp}_{metric}_barplot.png")
    plt.savefig(plot_path)
    plt.close()
