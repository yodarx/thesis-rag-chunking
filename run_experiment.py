import json
import time
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import os
from datetime import datetime
import hashlib

# Importiere alle Chunking-Funktionen
from src.chunking_strategies import (
    chunk_fixed_size,
    chunk_by_sentence,
    chunk_recursive,
    chunk_semantic
)
from src.vectorizer import Vectorizer
from src.evaluation import calculate_metrics
from src.visualizer import visualize_and_save_results

# --- EXPERIMENT CONFIGURATION ---
EXPERIMENTS = [
    # Baseline
    {'name': 'fixed_size_1000_100', 'function': chunk_fixed_size, 'params': {'chunk_size': 1000, 'chunk_overlap': 100}},
    {'name': 'fixed_size_512_50', 'function': chunk_fixed_size, 'params': {'chunk_size': 512, 'chunk_overlap': 50}},
    # Sentence-based
    {'name': 'sentence_s3', 'function': chunk_by_sentence, 'params': {'sentences_per_chunk': 3}},
    {'name': 'sentence_s5', 'function': chunk_by_sentence, 'params': {'sentences_per_chunk': 5}},
    # Recursive
    {'name': 'recursive_512_50', 'function': chunk_recursive, 'params': {'chunk_size': 512, 'chunk_overlap': 50}},
    # Semantic
    {'name': 'semantic_t0.85', 'function': chunk_semantic, 'params': {'similarity_threshold': 0.85}},
    {'name': 'semantic_t0.9', 'function': chunk_semantic, 'params': {'similarity_threshold': 0.9}},
]
DATASET_LIMIT = 2
VERBOSE_MODE = False
DATASET_METHODS = ['plain', 'markdown']
DATASET_LIMIT = 20  # Setze auf None für den kompletten Datensatz
EMBEDDING_CACHE_DIR = 'cache/embeddings'
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)


def main():
    base_output_dir, timestamp = setup_result_dir()
    print(f"Base result directory: '{base_output_dir}'")

    print("Initializing Vectorizer...")
    output_dir = os.path.join('results', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ergebnisse für diesen Lauf werden in '{output_dir}' gespeichert.")
    for method in DATASET_METHODS:
        print(f"\n--- Running dataset method: {method} ---")
        method_output_dir = os.path.join(base_output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)
    input_file_path = 'data/processed/asqa_preprocessed.jsonl'
    with open(input_file_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    if DATASET_LIMIT: all_data = all_data[:DATASET_LIMIT]
    results_df = save_raw_results(all_results, output_dir, timestamp)
    print(f"Starting experiments with {len(all_data)} data points.")
    all_results = []
    visualize_and_save_results(summary_df, output_dir, timestamp)
    for data_point in tqdm(all_data, desc="Processing Data Points", ncols=100):
        document = data_point['document_text'];
        question = data_point['question'];
        gold_passages = data_point['gold_passages']
        'chunking_time_s': 'mean', 'num_chunks': 'mean'
        for experiment in EXPERIMENTS:
            exp_name = experiment['name'];
            chunk_func = experiment['function'];
            chunk_params = experiment['params'].copy()

            start_time = time.time()
            # Spezielle Behandlung für Semantic Chunking
            if chunk_func == chunk_semantic:
                chunk_params['vectorizer'] = vectorizer
    return results_df
            chunks = chunk_func(document, **chunk_params)
            chunking_time = time.time() - start_time
def load_dataset(method='plain'):
            if not chunks: continue

            # Caching-Logik für Embeddings
            cache_key = f"{data_point['sample_id']}_{exp_name}";
            cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()
            cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"{cache_hash}.npy")
    os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(cache_file):
                chunk_embeddings = np.load(cache_file)
            else:
                chunk_embeddings_list = vectorizer.embed_documents(chunks)
                chunk_embeddings = np.array(chunk_embeddings_list, dtype='float32')
                np.save(cache_file, chunk_embeddings)
        'document': data_point['document_text'],
            # ... Rest der Pipeline (Indexing, Search, Evaluation) bleibt gleich
            index = faiss.IndexFlatL2(chunk_embeddings.shape[1]);
            index.add(chunk_embeddings)
            question_embedding_np = np.array(vectorizer.embed_documents([question]),
                                             dtype='float32')  # Caching für Fragen der Einfachheit halber weggelassen
            top_k = 5;
            _, indices = index.search(question_embedding_np, top_k)
            retrieved_chunks = [chunks[i] for i in indices[0]]
            metrics = calculate_metrics(retrieved_chunks, gold_passages, top_k)
    retrieved_chunks = search_relevant_chunks(chunk_embeddings, chunks,
            all_results.append({
                'sample_id': data_point['sample_id'], 'experiment': exp_name,
                'chunking_time_s': chunking_time, 'num_chunks': len(chunks),
                'mrr': metrics['mrr'], 'precision_at_k': metrics['precision_at_k'], 'recall': metrics['recall']
            })
        'name': experiment['name'],
    print("\nAll experiments finished. Saving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, f'{timestamp}_detailed_results.csv'), index=False)

    summary_df = results_df.groupby('experiment').agg({
        'mrr': 'mean', 'precision_at_k': 'mean', 'recall': 'mean',
        'chunking_time_s': 'mean', 'num_chunks': 'mean'
    }).reset_index()

    summary_df.to_csv(os.path.join(output_dir, f'{timestamp}_summary_results.csv'), index=False)
def perform_chunking(document, experiment_config):
    print("\n--- Aggregated Results ---")
    print(summary_df.to_string())
    chunking_time = time.time() - start_time
    visualize_and_save_results(summary_df, output_dir, timestamp)
    return chunks


def get_or_create_embeddings(chunks, sample_id, experiment_name, vectorizer):
    cache_file_path = generate_cache_file_path(sample_id, experiment_name)

    if os.path.exists(cache_file_path):
        return np.load(cache_file_path)

    return create_and_cache_embeddings(chunks, cache_file_path, vectorizer)


def generate_cache_file_path(sample_id, experiment_name):
    cache_key = f"{sample_id}_{experiment_name}"
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    return os.path.join(EMBEDDING_CACHE_DIR, f"{cache_hash}.npy")


def create_and_cache_embeddings(chunks, cache_file_path, vectorizer):
    chunk_embeddings_list = vectorizer.embed_documents(chunks)
    chunk_embeddings = np.array(chunk_embeddings_list, dtype='float32')
    np.save(cache_file_path, chunk_embeddings)
    return chunk_embeddings


def search_relevant_chunks(chunk_embeddings, chunks, question, vectorizer):
    index = create_search_index(chunk_embeddings)
    question_embedding = vectorize_question(question, vectorizer)

    top_k = 5
    _, indices = index.search(question_embedding, top_k)
    return [chunks[i] for i in indices[0]]


def create_search_index(chunk_embeddings):
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    return index


def vectorize_question(question, vectorizer):
    return np.array(vectorizer.embed_documents([question]), dtype='float32')


def create_experiment_result(sample_id, experiment_config, num_chunks, metrics):
    return {
        'sample_id': sample_id,
        'experiment': experiment_config['name'],
        'chunking_time_s': experiment_config.get('chunking_time', 0),
        'num_chunks': num_chunks,
        'mrr': metrics['mrr'],
        'precision_at_k': metrics['precision_at_k'],
        'recall': metrics['recall']
    }


if __name__ == "__main__":
    main()
