import os
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.evaluation import evaluation
from src.vectorizer.vectorizer import Vectorizer
from .results import ResultsHandler
from .retriever import FaissRetriever


def create_index_name(experiment_name: str, model_name: str) -> str:
    """Creates a descriptive name for the index directory."""
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

    def _get_index_paths(self, experiment: dict[str, Any]) -> tuple[str, str]:
        experiment_name: str = experiment["name"]

        index_folder_name: str = create_index_name(experiment_name, self.embedding_model_name)
        index_dir: str = os.path.join("data", "indices", index_folder_name)
        index_path: str = os.path.join(index_dir, "index.faiss")

        chunks_path: str = os.path.join("data", "chunks", experiment_name, "chunks_SORTED.json")
        chunks_path = os.path.abspath(chunks_path)

        return index_path, chunks_path

    def run_all(self) -> pd.DataFrame:
        total_experiments = len(self.experiments)
        total_dataset_size = len(self.dataset)
        BATCH_SIZE = 64  # GPU Batch Size

        print(f"🚀 Starting Benchmark Pipeline")
        print(f"==================================================")
        print(f"Total Experiments : {total_experiments}")
        print(f"Dataset Size      : {total_dataset_size} questions")
        print(f"Batch Size        : {BATCH_SIZE}")
        print(f"Top-K             : {self.top_k}")
        print(f"==================================================\n")

        for i_exp, experiment in enumerate(self.experiments, 1):
            exp_name: str = experiment["name"]

            print(f"▶ Experiment {i_exp}/{total_experiments}: {exp_name}")

            index_path, chunks_path = self._get_index_paths(experiment)

            if not os.path.exists(index_path):
                print(f"   ⚠️  Index not found: {index_path}. Skipping.")
                continue
            if not os.path.exists(chunks_path):
                print(f"   ⚠️  Chunks not found: {chunks_path}. Skipping.")
                continue

            # Index laden
            print(f"   Loading Index...", end="\r")
            self.retriever.load_index(index_path, chunks_path)
            print(f"   Index Loaded ({self.retriever.index.ntotal} documents). Starting Retrieval...")

            # Zeitmessung für das gesamte Experiment
            exp_start_time = time.time()

            # --- PROGRESS BAR SETUP ---
            # Wir nutzen 'total' als Anzahl der Fragen, nicht Batches. Das ist lesbarer.
            with tqdm(total=total_dataset_size, unit="q", desc=f"   Processing", ncols=100) as pbar:

                for i in range(0, total_dataset_size, BATCH_SIZE):
                    # 1. Batch erstellen
                    batch_data = self.dataset[i: i + BATCH_SIZE]
                    batch_questions = [d["question"] for d in batch_data]
                    current_batch_size = len(batch_questions)

                    # 2. Batch Retrieval (GPU)
                    batch_start_time = time.time()
                    try:
                        batch_retrieved_chunks = self.retriever.retrieve_batch(batch_questions, self.top_k)
                    except AttributeError:
                        # Fallback
                        batch_retrieved_chunks = [self.retriever.retrieve(q, self.top_k) for q in batch_questions]
                    batch_duration = time.time() - batch_start_time
                    avg_retrieval_time = batch_duration / current_batch_size if current_batch_size > 0 else 0

                    # 3. Metriken & Speichern (CPU)
                    for j, data_point in enumerate(batch_data):
                        retrieved_chunks = batch_retrieved_chunks[j]
                        log_matches: bool = experiment.get("log_matches", False)

                        # Haupt-Metrik
                        metrics: dict[str, float] = evaluation.calculate_metrics(
                            retrieved_chunks=retrieved_chunks,
                            gold_passages=data_point["gold_passages"],
                            k=self.top_k,
                            question=data_point["question"],
                            log_matches=log_matches,
                        )

                        # Zusätzliche k-Werte (für Plots später wichtig)
                        k_levels: list[int] = [1, 3, 5, 10, 20]
                        for k_val in k_levels:
                            if k_val <= self.top_k:
                                sub_metrics = evaluation.calculate_metrics(
                                    retrieved_chunks=retrieved_chunks,
                                    gold_passages=data_point["gold_passages"],
                                    k=k_val,
                                )
                                metrics[f"ndcg@{k_val}"] = sub_metrics["ndcg_at_k"]
                                metrics[f"recall@{k_val}"] = sub_metrics["recall_at_k"]
                                metrics[f"precision@{k_val}"] = sub_metrics["precision_at_k"]
                                metrics[f"f1@{k_val}"] = sub_metrics["f1_score_at_k"]
                                metrics[f"mrr@{k_val}"] = sub_metrics["mrr"]
                                metrics[f"map@{k_val}"] = sub_metrics["map"]

                        self.results_handler.add_result_record(
                            data_point,
                            exp_name,
                            chunking_time=0,
                            retrieval_time=avg_retrieval_time,
                            num_chunks=self.retriever.index.ntotal,
                            metrics=metrics,
                        )

                    # Update Progress Bar um die Anzahl der verarbeiteten Fragen
                    pbar.update(current_batch_size)

            # Zusammenfassung nach dem Experiment
            duration = time.time() - exp_start_time
            speed = total_dataset_size / duration if duration > 0 else 0
            print(f"   ✅ Finished in {duration:.2f}s ({speed:.1f} q/s)\n")

        print("==================================================")
        print("All experiments finished. Generating Report...")

        detailed_df: pd.DataFrame = self.results_handler.save_detailed_results()
        if detailed_df.empty:
            print("Warning: No results were generated.")
            return pd.DataFrame()

        summary_df: pd.DataFrame = self.results_handler.create_and_save_summary(detailed_df)

        self.results_handler.display_summary(summary_df)

        return summary_df

