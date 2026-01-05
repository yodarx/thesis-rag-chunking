import os
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.evaluation import evaluation
from src.vectorizer.vectorizer import Vectorizer

from .results import ResultsHandler
from .retriever import FaissRetriever


def create_index_name(experiment_name: str, model_name: str) -> str:
    """Creates a descriptive name for the index directory."""
    # Sanitize model name for use in file paths
    sanitized_model_name: str = model_name.replace("/", "_")
    return f"{experiment_name}_{sanitized_model_name}"


class ExperimentRunner:
    def __init__(
        self,
        experiments: list[dict[str, Any]],
        dataset: list[dict[str, Any]],
        vectorizer: Vectorizer,
        retriever: FaissRetriever,
        results_handler: ResultsHandler,
        top_k: int,
        embedding_model_name: str,
        difficulty: str | None = None,
    ) -> None:
        self.experiments: list[dict[str, Any]] = experiments
        self.vectorizer: Vectorizer = vectorizer
        self.retriever: FaissRetriever = retriever
        self.results_handler: ResultsHandler = results_handler
        self.top_k: int = top_k
        self.embedding_model_name: str = embedding_model_name

        if difficulty:
            print(f"Filtering dataset for difficulty: {difficulty}")
            self.dataset: list[dict[str, Any]] = [
                d for d in dataset if d.get("difficulty") == difficulty
            ]
            print(f"Filtered dataset size: {len(self.dataset)} (Original: {len(dataset)})")
        else:
            self.dataset: list[dict[str, Any]] = dataset

    def _get_index_paths(self, experiment_name: str) -> tuple[str, str]:
        index_folder_name: str = create_index_name(experiment_name, self.embedding_model_name)
        index_dir: str = os.path.join("data", "indices", index_folder_name)
        index_path: str = os.path.join(index_dir, "index.faiss")
        chunks_path: str = os.path.join(index_dir, "chunks.json")
        return index_path, chunks_path

    def run_all(self) -> pd.DataFrame:
        print(f"Starting experiments with {len(self.dataset)} data points.")

        # The outer loop should be experiments, as we load an index per experiment
        for experiment in self.experiments:
            exp_name: str = experiment["name"]
            print(f"\nProcessing experiment: {exp_name}")

            index_path, chunks_path = self._get_index_paths(exp_name)

            if not os.path.exists(index_path) or not os.path.exists(chunks_path):
                print(f"Warning: Index for experiment '{exp_name}' not found. Skipping.")
                print(f"  - Looked for: {index_path}")
                continue

            self.retriever.load_index(index_path, chunks_path)

            for data_point in tqdm(self.dataset, desc=f"Evaluating {exp_name}"):
                # Retrieve relevant chunks for the question (ONCE - expensive operation)
                retrieved_chunks: list[str] = self.retriever.retrieve(
                    data_point["question"], self.top_k
                )

                # print(f"\nQuestion: {data_point['question']}")
                # print(f"Retrieved Chunks: {retrieved_chunks}")

                log_matches: bool = experiment.get("log_matches", False)

                # --- H2: Retrieve once, measure multiple k-values ---
                # Standard metrics at configured k
                metrics: dict[str, float] = evaluation.calculate_metrics(
                    retrieved_chunks=retrieved_chunks,
                    gold_passages=data_point["gold_passages"],
                    k=self.top_k,
                    question=data_point["question"],
                    log_matches=log_matches,
                )

                # Additional: Calculate metrics for different k-values (cheap re-computations)
                # This allows measuring the impact of retrieval depth in a single run
                k_levels: list[int] = [1, 3, 5, 10, 20]

                for k_val in k_levels:
                    # Only calculate if we have enough chunks retrieved
                    if k_val <= self.top_k:
                        sub_metrics: dict[str, float] = evaluation.calculate_metrics(
                            retrieved_chunks=retrieved_chunks,
                            gold_passages=data_point["gold_passages"],
                            k=k_val,
                        )

                        # Store with @k suffix for clarity (e.g., "ndcg@1", "recall@5")
                        metrics[f"ndcg@{k_val}"] = sub_metrics["ndcg_at_k"]
                        metrics[f"recall@{k_val}"] = sub_metrics["recall_at_k"]
                        metrics[f"precision@{k_val}"] = sub_metrics["precision_at_k"]
                        metrics[f"f1@{k_val}"] = sub_metrics["f1_score_at_k"]
                        metrics[f"mrr@{k_val}"] = sub_metrics["mrr"]
                        metrics[f"map@{k_val}"] = sub_metrics["map"]

                # Chunking time is 0, num_chunks is from the loaded index
                self.results_handler.add_result_record(
                    data_point,
                    exp_name,
                    chunking_time=0,
                    num_chunks=self.retriever.index.ntotal,
                    metrics=metrics,
                )

        print("\nAll experiments finished. Saving results...")

        detailed_df: pd.DataFrame = self.results_handler.save_detailed_results()
        if detailed_df.empty:
            print("Warning: No results were generated.")
            return pd.DataFrame()

        summary_df: pd.DataFrame = self.results_handler.create_and_save_summary(detailed_df)

        self.results_handler.display_summary(summary_df)
        return summary_df
