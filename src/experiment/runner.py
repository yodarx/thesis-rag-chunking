import json
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
    sanitized_model_name = model_name.replace("/", "_")
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
    ):
        self.experiments = experiments
        self.dataset = dataset
        self.vectorizer = vectorizer
        self.retriever = retriever
        self.results_handler = results_handler
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name

    def _process_single_experiment(
        self, data_point: dict[str, Any], experiment: dict[str, Any]
    ) -> None:
        exp_name = experiment["name"]

        # Instead of chunking, we now load the pre-built index
        index_folder_name = create_index_name(exp_name, self.embedding_model_name)
        index_dir = os.path.join("indices", index_folder_name)

        index_path = os.path.join(index_dir, "index.faiss")
        chunks_path = os.path.join(index_dir, "chunks.json")
        metadata_path = os.path.join(index_dir, "metadata.json")

        if not all(os.path.exists(p) for p in [index_path, chunks_path, metadata_path]):
            print(f"Warning: Index for experiment '{exp_name}' not found. Skipping.")
            print(f"Looked in: {index_dir}")
            return

        # Load the index and chunks into the retriever
        self.retriever.load_index(index_path, chunks_path)

        with open(metadata_path, encoding="utf-8") as f:
            json.load(f)

        # Retrieve relevant chunks for the question
        retrieved_chunks = self.retriever.retrieve(data_point["question"], self.top_k)

        # We need to find which document the retrieved chunks belong to,
        # but for this evaluation, we assume the retrieval is across all docs,
        # which is what we want.

        log_matches = experiment.get("log_matches", False)
        metrics = evaluation.calculate_metrics(
            retrieved_chunks=retrieved_chunks,
            gold_passages=data_point["gold_passages"],
            k=self.top_k,
            question=data_point["question"],
            log_matches=log_matches,
        )

        # Chunking time is now 0 since it's pre-processed
        self.results_handler.add_result_record(
            data_point,
            exp_name,
            chunking_time=0,
            num_chunks=self.retriever.index.ntotal,
            metrics=metrics,
        )

    def run_all(self) -> pd.DataFrame:
        print(f"Starting experiments with {len(self.dataset)} data points.")

        # The outer loop should be experiments, as we load an index per experiment
        for experiment in self.experiments:
            exp_name = experiment["name"]
            print(f"\nProcessing experiment: {exp_name}")

            # Load the index for this experiment once
            index_folder_name = create_index_name(exp_name, self.embedding_model_name)
            index_dir = os.path.join("indices", index_folder_name)
            index_path = os.path.join(index_dir, "index.faiss")
            chunks_path = os.path.join(index_dir, "chunks.json")

            if not os.path.exists(index_path) or not os.path.exists(chunks_path):
                print(f"Warning: Index for experiment '{exp_name}' not found. Skipping.")
                print(f"  - Looked for: {index_path}")
                continue

            self.retriever.load_index(index_path, chunks_path)

            for data_point in tqdm(self.dataset, desc=f"Evaluating {exp_name}"):
                # Retrieve relevant chunks for the question
                retrieved_chunks = self.retriever.retrieve(data_point["question"], self.top_k)

                log_matches = experiment.get("log_matches", False)
                metrics = evaluation.calculate_metrics(
                    retrieved_chunks=retrieved_chunks,
                    gold_passages=data_point["gold_passages"],
                    k=self.top_k,
                    question=data_point["question"],
                    log_matches=log_matches,
                )

                # Chunking time is 0, num_chunks is from the loaded index
                self.results_handler.add_result_record(
                    data_point,
                    exp_name,
                    chunking_time=0,
                    num_chunks=self.retriever.index.ntotal,
                    metrics=metrics,
                )

        print("\nAll experiments finished. Saving results...")

        detailed_df = self.results_handler.save_detailed_results()
        if detailed_df.empty:
            print("Warning: No results were generated.")
            return pd.DataFrame()

        summary_df = self.results_handler.create_and_save_summary(detailed_df)

        self.results_handler.display_summary(summary_df)
        return summary_df
