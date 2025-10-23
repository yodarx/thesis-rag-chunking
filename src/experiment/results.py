import os
from typing import Any

import pandas as pd


class ResultsHandler:
    def __init__(self, output_dir: str, timestamp: str):
        self.output_dir = output_dir
        self.timestamp = timestamp
        self.all_results: list[dict[str, Any]] = []

    def add_result_record(
        self,
        data_point: dict[str, Any],
        experiment_name: str,
        chunking_time: float,
        num_chunks: int,
        metrics: dict[str, float],
    ) -> None:
        record = {
            "sample_id": data_point["sample_id"],
            "experiment": experiment_name,
            "chunking_time_s": chunking_time,
            "num_chunks": num_chunks,
            **metrics,
        }
        self.all_results.append(record)

    def save_detailed_results(self) -> pd.DataFrame:
        if not self.all_results:
            return pd.DataFrame()

        results_df = pd.DataFrame(self.all_results)
        filepath = os.path.join(self.output_dir, f"{self.timestamp}_detailed_results.csv")
        results_df.to_csv(filepath, index=False)
        return results_df

    def create_and_save_summary(self, detailed_df: pd.DataFrame) -> pd.DataFrame:
        if detailed_df.empty:
            return pd.DataFrame()

        summary_df = (
            detailed_df.groupby("experiment")
            .agg(
                {
                    "mrr": "mean",
                    "map": "mean",
                    "ndcg_at_k": "mean",
                    "precision_at_k": "mean",
                    "recall_at_k": "mean",
                    "f1_score_at_k": "mean",
                    "chunking_time_s": "mean",
                    "num_chunks": "mean",
                }
            )
            .reset_index()
        )

        filepath = os.path.join(self.output_dir, f"{self.timestamp}_summary_results.csv")
        summary_df.to_csv(filepath, index=False)
        return summary_df

    def display_summary(self, summary_df: pd.DataFrame) -> None:
        print("\n--- Aggregated Results ---")
        print(summary_df.to_string())
