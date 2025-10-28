import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation import evaluation
from src.vectorizer.vectorizer import Vectorizer

from .results import ResultsHandler
from .retriever import FaissRetriever


class ExperimentRunner:
    def __init__(
        self,
        experiments: list[dict[str, Any]],
        dataset: list[dict[str, Any]],
        vectorizer: Vectorizer,
        retriever: FaissRetriever,
        results_handler: ResultsHandler,
        top_k: int,
    ):
        self.experiments = experiments
        self.dataset = dataset
        self.vectorizer = vectorizer
        self.retriever = retriever
        self.results_handler = results_handler
        # self.cacher = cacher (entfernt)
        self.top_k = top_k

    def _prepare_chunk_parameters(self, experiment: dict[str, Any]) -> dict[str, Any]:
        chunk_params = experiment["params"].copy()

        if experiment["function"].__name__ == "chunk_semantic":
            chunk_params["vectorizer"] = self.vectorizer
        return chunk_params

    def _execute_chunking(
        self, document: str, chunk_func: Callable, chunk_params: dict[str, Any]
    ) -> (list[str], float):
        start_time = time.time()
        chunks = chunk_func(document, **chunk_params)
        chunking_time = time.time() - start_time
        return chunks, chunking_time

    def _process_single_experiment(
        self, data_point: dict[str, Any], experiment: dict[str, Any]
    ) -> None:
        exp_name = experiment["name"]
        chunk_func = experiment["function"]
        chunk_params = self._prepare_chunk_parameters(experiment)

        chunks, chunking_time = self._execute_chunking(
            data_point["document_text"], chunk_func, chunk_params
        )

        if not chunks:
            print(f"Warning: No chunks created for {exp_name} on {data_point['sample_id']}")
            return

        chunk_embeddings_list = self.vectorizer.embed_documents(chunks)
        chunk_embeddings = np.array(chunk_embeddings_list, dtype="float32")
        indices = self.retriever.search(data_point["question"], chunk_embeddings, self.top_k)

        metrics = evaluation.calculate_metrics(
            retrieved_chunks=[chunks[i] for i in indices],
            gold_passages=data_point["gold_passages"],
            k=self.top_k,
            question=data_point["question"],
        )

        self.results_handler.add_result_record(
            data_point, exp_name, chunking_time, len(chunks), metrics
        )

    def run_all(self) -> pd.DataFrame:
        print(f"Starting experiments with {len(self.dataset)} data points.")

        for data_point in tqdm(self.dataset, desc="Processing Data Points"):
            for experiment in self.experiments:
                self._process_single_experiment(data_point, experiment)

        print("\nAll experiments finished. Saving results...")

        detailed_df = self.results_handler.save_detailed_results()
        if detailed_df.empty:
            print("Warning: No results were generated.")
            return pd.DataFrame()

        summary_df = self.results_handler.create_and_save_summary(detailed_df)

        self.results_handler.display_summary(summary_df)
        return summary_df
