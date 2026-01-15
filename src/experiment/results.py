import os
from typing import Any

import pandas as pd


class ResultsHandler:
    def __init__(self, output_dir: str, timestamp: str) -> None:
        self.output_dir: str = output_dir
        self.timestamp: str = timestamp
        self.all_results: list[dict[str, Any]] = []

    def add_result_record(
        self,
        data_point: dict[str, Any],
        experiment_name: str,
        chunking_time: float,
        retrieval_time: float,
        num_chunks: int,
        metrics: dict[str, float],
    ) -> None:
        record: dict[str, Any] = {
            "sample_id": data_point.get("sample_id", data_point.get("id", "unknown")),
            "question": data_point.get("question", ""),
            "experiment": experiment_name,
            "chunking_time_s": chunking_time,
            "retrieval_time_s": retrieval_time,
            "num_chunks": num_chunks,
            **metrics,
        }
        self.all_results.append(record)

    def save_detailed_results(self) -> pd.DataFrame:
        if not self.all_results:
            return pd.DataFrame()

        results_df: pd.DataFrame = pd.DataFrame(self.all_results)
        filepath: str = self._get_detailed_results_path()
        results_df.to_csv(filepath, index=False)
        return results_df

    def _get_detailed_results_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.timestamp}_detailed_results.csv")

    def create_and_save_summary(self, detailed_df: pd.DataFrame) -> pd.DataFrame:
        if detailed_df.empty:
            return pd.DataFrame()

        summary_df: pd.DataFrame = (
            detailed_df.groupby("experiment")
            .agg(
                {
                    "mrr": "mean",
                    "map": "mean",
                    "ndcg_at_k": "mean",
                    "precision_at_k": "mean",
                    "recall_at_k": "mean",
                    "f1_score_at_k": "mean",
                }
            )
            .reset_index()
        )

        summary_path: str = self._get_summary_path()
        summary_df.to_csv(summary_path, index=False)
        return summary_df

    def _get_summary_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.timestamp}_summary.csv")

    def display_summary(self, summary_df: pd.DataFrame) -> None:
        print("\n--- Aggregated Results ---")
        print(summary_df.to_string())
