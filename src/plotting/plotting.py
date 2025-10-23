import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Konfiguration ---

# Metriken, die geplottet werden sollen
METRICS_TO_PLOT: list[str] = [
    "mrr",
    "map",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "f1_score_at_k",
]

# Anzeigenamen für Titel und Achsenbeschriftungen
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "mrr": "Mean Reciprocal Rank (MRR)",
    "map": "Mean Average Precision (MAP)",
    "ndcg_at_k": "Normalized Discounted Cumulative Gain (NDCG@5)",
    "precision_at_k": "Precision@5",
    "recall_at_k": "Recall@5",
    "f1_score_at_k": "F1-Score@5",
}


# --- Private Plotting-Helfer ---


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
        bar_plot.bar_label(container, fmt="%.4f", padding=3)

    filename = os.path.join(output_dir, f"{timestamp}_bar_{metric}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def _create_and_save_scatter_plot(
    df: pd.DataFrame, metric: str, display_name: str, output_dir: str, timestamp: str
) -> None:
    """Erstellt und speichert ein Streudiagramm (Kosten-Nutzen) für eine Metrik."""
    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=df, x="chunking_time_s", y=metric, hue="experiment", s=200, palette="deep")
    plt.title(f"Cost-Benefit Analysis: {display_name} vs. Compute Time", fontsize=16, pad=20)
    plt.xlabel("Average Chunking Time per Document (seconds)", fontsize=12)
    plt.ylabel(display_name, fontsize=12)
    plt.legend(title="Strategies", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    filename = os.path.join(output_dir, f"{timestamp}_scatter_{metric}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# --- Hauptfunktion ---


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

        display_name = METRIC_DISPLAY_NAMES.get(metric, metric)

        _create_and_save_bar_plot(
            df=summary_df,
            metric=metric,
            display_name=display_name,
            output_dir=output_dir,
            timestamp=timestamp,
        )

        _create_and_save_scatter_plot(
            df=summary_df,
            metric=metric,
            display_name=display_name,
            output_dir=output_dir,
            timestamp=timestamp,
        )

    print("\nAll plots were successfully saved.")
