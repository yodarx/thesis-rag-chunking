import os
import time
import re
from typing import Any
import difflib  # For fuzzy matching

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

    def _normalize(self, text: str) -> str:
        """Normalizes text for robust matching (lowercase, single spaces)."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip().lower()

    def _extract_text(self, chunk: Any) -> str:
        """Safely extracts text string from chunk, handling dicts, objects, or raw strings."""
        if isinstance(chunk, str):
            return chunk
        elif isinstance(chunk, dict):
            return chunk.get("page_content", chunk.get("content", ""))
        elif hasattr(chunk, "page_content"):
            return chunk.page_content
        else:
            return str(chunk)

    def run_all(self) -> pd.DataFrame:
        total_experiments = len(self.experiments)
        total_dataset_size = len(self.dataset)
        BATCH_SIZE = 64

        print(f"🚀 Starting Benchmark Pipeline")
        print(f"==================================================")
        print(f"Total Experiments : {total_experiments}")
        print(f"Dataset Size      : {total_dataset_size} questions")
        print(f"Batch Size        : {BATCH_SIZE}")
        print(f"Top-K             : {self.top_k}")
        print(f"==================================================\n")

        # Counters to ensure we get a variety of examples without spamming
        debug_counters = {}
        MAX_DEBUG_PER_TYPE = 2  # Limits examples per failure type PER strategy

        for i_exp, experiment in enumerate(self.experiments, 1):
            exp_name: str = experiment["name"]

            # Init counters for this strategy
            if exp_name not in debug_counters:
                debug_counters[exp_name] = {
                    "fragmentation": 0,
                    "near_miss": 0,
                    "lost_in_middle": 0,
                    "partial_hop": 0
                }

            print(f"▶ Experiment {i_exp}/{total_experiments}: {exp_name}")

            index_path, chunks_path = self._get_index_paths(experiment)

            if not os.path.exists(index_path):
                print(f"   ⚠️  Index not found: {index_path}. Skipping.")
                continue
            if not os.path.exists(chunks_path):
                print(f"   ⚠️  Chunks not found: {chunks_path}. Skipping.")
                continue

            # Load Index
            print(f"   Loading Index...", end="\r")
            self.retriever.load_index(index_path, chunks_path)
            print(f"   Index Loaded ({self.retriever.index.ntotal} documents). Starting Retrieval...")

            exp_start_time = time.time()

            # --- PROGRESS BAR SETUP ---
            with tqdm(total=total_dataset_size, unit="q", desc=f"   Processing", ncols=100) as pbar:

                for i in range(0, total_dataset_size, BATCH_SIZE):
                    # 1. Create Batch
                    batch_data = self.dataset[i: i + BATCH_SIZE]
                    batch_questions = [d["question"] for d in batch_data]
                    current_batch_size = len(batch_questions)

                    # 2. Batch Retrieval
                    batch_start_time = time.time()
                    try:
                        batch_retrieved_chunks = self.retriever.retrieve_batch(batch_questions, self.top_k)
                    except AttributeError:
                        batch_retrieved_chunks = [self.retriever.retrieve(q, self.top_k) for q in batch_questions]

                    batch_duration = time.time() - batch_start_time
                    avg_retrieval_time = batch_duration / current_batch_size if current_batch_size > 0 else 0

                    # 3. Metrics & Forensic Debugging
                    for j, data_point in enumerate(batch_data):
                        retrieved_chunks = batch_retrieved_chunks[j]
                        log_matches: bool = experiment.get("log_matches", False)

                        # Standard Metrics
                        metrics: dict[str, float] = evaluation.calculate_metrics(
                            retrieved_chunks=retrieved_chunks,
                            gold_passages=data_point["gold_passages"],
                            k=self.top_k,
                            question=data_point["question"],
                            log_matches=log_matches,
                        )

                        # --- 🕵️‍♂️ FORENSIC ANALYSIS ----------------------------

                        # Use helper to safely get text list
                        retrieved_texts = [self._extract_text(c) for c in retrieved_chunks]

                        # 1. CHECK FOR "LOST IN THE MIDDLE" (Ranking Failure)
                        # We use .get() to avoid KeyErrors
                        recall_k = metrics.get("recall_at_k", 0.0)  # Main recall

                        # Note: recall_at_20 / recall_at_5 might not exist yet if k=10
                        # So we skip this specific check unless we calculated them.
                        # Instead, we rely on the loops below for detailed metrics.
                        # BUT for the main debug, we check the PRIMARY recall.

                        # 2. CHECK FOR PARTIAL MULTI-HOP FAILURE
                        total_gold = len(data_point["gold_passages"])
                        if total_gold > 1:
                            found_count = 0
                            for gold in data_point["gold_passages"]:
                                norm_gold = self._normalize(gold)
                                if any(norm_gold in self._normalize(t) for t in retrieved_texts[:10]):
                                    found_count += 1

                            if 0 < found_count < total_gold:
                                if debug_counters[exp_name]["partial_hop"] < MAX_DEBUG_PER_TYPE:
                                    tqdm.write(f"\n🧩 [CASE STUDY] PARTIAL MULTI-HOP (Reasoning Fail)")
                                    tqdm.write(f"   Strategy: {exp_name}")
                                    tqdm.write(f"   QID: {data_point.get('qa_id', 'N/A')}")
                                    tqdm.write(f"   Question: {data_point['question']}")
                                    tqdm.write(f"   Found: {found_count}/{total_gold} required text segments.")
                                    tqdm.write(f"   Diagnosis: Context disconnected. Reasoning chain broken.")
                                    debug_counters[exp_name]["partial_hop"] += 1
                                    tqdm.write("-" * 50)

                        # 3. & 4. CHECK FOR FRAGMENTATION & NEAR MISS (If Recall=0)
                        # FIX: Check 'recall_at_k' which is always present from calculate_metrics
                        if recall_k == 0.0:
                            for gold in data_point["gold_passages"]:
                                norm_gold = self._normalize(gold)
                                if not norm_gold: continue

                                # FRAGMENTATION CHECK
                                combined_top_5 = " ".join([self._normalize(t) for t in retrieved_texts[:5]])
                                if norm_gold in combined_top_5:
                                    if debug_counters[exp_name]["fragmentation"] < MAX_DEBUG_PER_TYPE:
                                        tqdm.write(f"\n🔥 [CASE STUDY] FRAGMENTATION DETECTED")
                                        tqdm.write(f"   Strategy: {exp_name}")
                                        tqdm.write(f"   QID: {data_point.get('qa_id', 'N/A')}")
                                        tqdm.write(f"   Gold: '{gold[:60]}...'")
                                        tqdm.write(f"   Diagnosis: Answer split across Top 5 chunks.")
                                        debug_counters[exp_name]["fragmentation"] += 1
                                        tqdm.write("-" * 50)
                                    break

                                    # NEAR MISS CHECK
                                else:
                                    found_near = False
                                    for idx, chunk_text in enumerate(retrieved_texts[:3]):
                                        norm_chunk = self._normalize(chunk_text)
                                        s = difflib.SequenceMatcher(None, norm_gold, norm_chunk)
                                        match = s.find_longest_match(0, len(norm_gold), 0, len(norm_chunk))

                                        if match.size > len(norm_gold) * 0.8:
                                            if debug_counters[exp_name]["near_miss"] < MAX_DEBUG_PER_TYPE:
                                                tqdm.write(f"\n⚠️ [CASE STUDY] NEAR MISS (Strict Metric)")
                                                tqdm.write(f"   Strategy: {exp_name}")
                                                tqdm.write(f"   QID: {data_point.get('qa_id', 'N/A')}")
                                                tqdm.write(f"   Overlap: {int(match.size / len(norm_gold) * 100)}%")
                                                debug_counters[exp_name]["near_miss"] += 1
                                                tqdm.write("-" * 50)
                                            found_near = True
                                            break
                                    if found_near: break

                        # --- END FORENSIC ANALYSIS ---------------------------

                        # Calculate additional K-levels
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

                    pbar.update(current_batch_size)

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