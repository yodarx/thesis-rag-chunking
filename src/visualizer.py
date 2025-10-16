import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def visualize_and_save_results(summary_df: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Generates and saves plots with timestamp in the filename.
    """
    print(f"\nGeneriere und speichere Plots im Ordner: {output_dir}")
    sns.set_theme(style="whitegrid")

    metrics_to_plot = ['mrr', 'precision_at_k', 'recall']
    metric_display_names = {
        'mrr': 'Mean Reciprocal Rank (MRR)',
        'precision_at_k': 'Precision@5',
        'recall': 'Recall'
    }

    for metric in metrics_to_plot:
        display_name = metric_display_names.get(metric, metric)

        # Bar Chart
        plt.figure(figsize=(10, 6))

        # --- KORRIGIERTE ZEILE ---
        # We explicitly tell seaborn to color (`hue`) based on the 'experiment' column
        # and hide the redundant legend.
        bar_plot = sns.barplot(
            data=summary_df.sort_values(metric, ascending=False),
            x=metric, y='experiment', hue='experiment', palette='viridis', legend=False
        )
        # --- ENDE DER KORREKTUR ---

        plt.title(f'Performance Comparison: {display_name}', fontsize=16)
        plt.xlabel(display_name, fontsize=12)
        plt.ylabel('Experiment', fontsize=12)
        for container in bar_plot.containers:
            bar_plot.bar_label(container, fmt='%.4f')

        filename_bar = os.path.join(output_dir, f'{timestamp}_performance_comparison_{metric}.png')
        plt.savefig(filename_bar, bbox_inches='tight')
        plt.close()

        # Scatter Plot (no changes needed here)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=summary_df,
            x='chunking_time_s', y=metric, hue='experiment', s=150
        )
        plt.title(f'Cost-Benefit Analysis: {display_name} vs. Compute Time', fontsize=16)
        plt.xlabel('Average Chunking Time per Document (s)', fontsize=12)
        plt.ylabel(display_name, fontsize=12)
        plt.legend(title='Experiments')
        plt.grid(True)

        filename_scatter = os.path.join(output_dir, f'{timestamp}_cost_benefit_{metric}.png')
        plt.savefig(filename_scatter, bbox_inches='tight')
        plt.close()

    print(f"\nAlle Plots wurden erfolgreich gespeichert.")
