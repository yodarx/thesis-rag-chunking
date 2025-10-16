import json
import time
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import os
from datetime import datetime
import hashlib

from src.chunking_strategies import (
    chunk_fixed_size,
    chunk_by_sentence,
    chunk_recursive,
    chunk_semantic
)
from src.vectorizer import Vectorizer
from src.evaluation import calculate_metrics
from src.visualizer import visualize_and_save_results


class ExperimentRunner:
    def __init__(self):
        self.experiments = self._define_experiments()
        self.dataset_limit = 5
        self.embedding_cache_dir = 'cache/embeddings'
        self._ensure_cache_directory_exists()

    def _define_experiments(self):
        return [
            {'name': 'fixed_size_25_5', 'function': chunk_fixed_size,
             'params': {'chunk_size': 25, 'chunk_overlap': 5}},
            {'name': 'fixed_size_10_5', 'function': chunk_fixed_size,
             'params': {'chunk_size': 10, 'chunk_overlap': 5}},
            {'name': 'sentence_s3', 'function': chunk_by_sentence, 'params': {'sentences_per_chunk': 3}},
            {'name': 'sentence_s5', 'function': chunk_by_sentence, 'params': {'sentences_per_chunk': 5}},
            {'name': 'recursive_512_50', 'function': chunk_recursive,
             'params': {'chunk_size': 512, 'chunk_overlap': 50}},
            {'name': 'semantic_t0.85', 'function': chunk_semantic, 'params': {'similarity_threshold': 0.85}},
            {'name': 'semantic_t0.9', 'function': chunk_semantic, 'params': {'similarity_threshold': 0.9}},
        ]

    def _ensure_cache_directory_exists(self):
        os.makedirs(self.embedding_cache_dir, exist_ok=True)

    def _create_output_directory(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join('results', timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ergebnisse für diesen Lauf werden in '{output_dir}' gespeichert.")
        return output_dir, timestamp

    def _load_dataset(self):
        input_file_path = 'data/processed/asqa_preprocessed_plain.jsonl'
        with open(input_file_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
        if self.dataset_limit:
            all_data = all_data[:self.dataset_limit]
        return all_data

    def _prepare_chunk_parameters(self, experiment, vectorizer):
        chunk_params = experiment['params'].copy()
        if experiment['function'] == chunk_semantic:
            chunk_params['vectorizer'] = vectorizer
        return chunk_params

    def _execute_chunking(self, document, chunk_func, chunk_params):
        start_time = time.time()
        chunks = chunk_func(document, **chunk_params)
        chunking_time = time.time() - start_time
        return chunks, chunking_time

    def _generate_cache_key(self, sample_id, experiment_name):
        cache_key = f"{sample_id}_{experiment_name}"
        cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        return os.path.join(self.embedding_cache_dir, f"{cache_hash}.npy")

    def _get_or_create_embeddings(self, chunks, cache_file, vectorizer):
        if os.path.exists(cache_file):
            return np.load(cache_file)
        else:
            chunk_embeddings_list = vectorizer.embed_documents(chunks)
            chunk_embeddings = np.array(chunk_embeddings_list, dtype='float32')
            np.save(cache_file, chunk_embeddings)
            return chunk_embeddings

    def _perform_similarity_search(self, chunk_embeddings, question, vectorizer, top_k=5):
        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(chunk_embeddings)
        question_embedding_np = np.array(vectorizer.embed_documents([question]), dtype='float32')
        _, indices = index.search(question_embedding_np, top_k)
        return indices[0]

    def _create_result_record(self, data_point, experiment_name, chunking_time, chunks, metrics):
        return {
            'sample_id': data_point['sample_id'],
            'experiment': experiment_name,
            'chunking_time_s': chunking_time,
            'num_chunks': len(chunks),
            'mrr': metrics['mrr'],
            'precision_at_k': metrics['precision_at_k'],
            'recall': metrics['recall']
        }

    def _process_single_data_point(self, data_point, vectorizer):
        document = data_point['document_text']
        question = data_point['question']
        gold_passages = data_point['gold_passages']
        results = []

        for experiment in self.experiments:
            exp_name = experiment['name']
            chunk_func = experiment['function']
            chunk_params = self._prepare_chunk_parameters(experiment, vectorizer)

            chunks, chunking_time = self._execute_chunking(document, chunk_func, chunk_params)

            if not chunks:
                continue

            cache_file = self._generate_cache_key(data_point['sample_id'], exp_name)
            chunk_embeddings = self._get_or_create_embeddings(chunks, cache_file, vectorizer)

            top_k = 5
            indices = self._perform_similarity_search(chunk_embeddings, question, vectorizer, top_k)
            retrieved_chunks = [chunks[i] for i in indices]
            metrics = calculate_metrics(retrieved_chunks, gold_passages, top_k, question)

            result = self._create_result_record(data_point, exp_name, chunking_time, chunks, metrics)
            results.append(result)

        return results

    def _save_detailed_results(self, all_results, output_dir, timestamp):
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, f'{timestamp}_detailed_results.csv'), index=False)
        return results_df

    def _create_and_save_summary(self, results_df, output_dir, timestamp):
        summary_df = results_df.groupby('experiment').agg({
            'mrr': 'mean',
            'precision_at_k': 'mean',
            'recall': 'mean',
            'chunking_time_s': 'mean',
            'num_chunks': 'mean'
        }).reset_index()

        summary_df.to_csv(os.path.join(output_dir, f'{timestamp}_summary_results.csv'), index=False)
        return summary_df

    def _display_results(self, summary_df):
        print("\n--- Aggregated Results ---")
        print(summary_df.to_string())

    def run_experiments(self):
        output_dir, timestamp = self._create_output_directory()

        print("Initializing Vectorizer...")
        vectorizer = Vectorizer()

        all_data = self._load_dataset()
        print(f"Starting experiments with {len(all_data)} data points.")

        all_results = []
        for data_point in tqdm(all_data, desc="Processing Data Points", ncols=100):
            results = self._process_single_data_point(data_point, vectorizer)
            all_results.extend(results)

        print("\nAll experiments finished. Saving results...")
        results_df = self._save_detailed_results(all_results, output_dir, timestamp)
        summary_df = self._create_and_save_summary(results_df, output_dir, timestamp)

        self._display_results(summary_df)
        visualize_and_save_results(summary_df, output_dir, timestamp)


def main():
    runner = ExperimentRunner()
    runner.run_experiments()


if __name__ == "__main__":
    main()
