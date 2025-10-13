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
DATASET_LIMIT = 20  # Setze auf None für den kompletten Datensatz

# --- Caching Konfiguration ---
EMBEDDING_CACHE_DIR = 'cache/embeddings'
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)


# --- MAIN EXECUTION ---
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join('results', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ergebnisse für diesen Lauf werden in '{output_dir}' gespeichert.")

    print("Initializing Vectorizer...")
    vectorizer = Vectorizer()

    input_file_path = 'data/processed/asqa_preprocessed.jsonl'
    with open(input_file_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    if DATASET_LIMIT: all_data = all_data[:DATASET_LIMIT]

    print(f"Starting experiments with {len(all_data)} data points.")
    all_results = []

    for data_point in tqdm(all_data, desc="Processing Data Points", ncols=100):
        document = data_point['document_text'];
        question = data_point['question'];
        gold_passages = data_point['gold_passages']

        for experiment in EXPERIMENTS:
            exp_name = experiment['name'];
            chunk_func = experiment['function'];
            chunk_params = experiment['params'].copy()

            start_time = time.time()
            # Spezielle Behandlung für Semantic Chunking
            if chunk_func == chunk_semantic:
                chunk_params['vectorizer'] = vectorizer

            chunks = chunk_func(document, **chunk_params)
            chunking_time = time.time() - start_time

            if not chunks: continue

            # Caching-Logik für Embeddings
            cache_key = f"{data_point['sample_id']}_{exp_name}";
            cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()
            cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"{cache_hash}.npy")

            if os.path.exists(cache_file):
                chunk_embeddings = np.load(cache_file)
            else:
                chunk_embeddings_list = vectorizer.embed_documents(chunks)
                chunk_embeddings = np.array(chunk_embeddings_list, dtype='float32')
                np.save(cache_file, chunk_embeddings)

            # ... Rest der Pipeline (Indexing, Search, Evaluation) bleibt gleich
            index = faiss.IndexFlatL2(chunk_embeddings.shape[1]);
            index.add(chunk_embeddings)
            question_embedding_np = np.array(vectorizer.embed_documents([question]),
                                             dtype='float32')  # Caching für Fragen der Einfachheit halber weggelassen
            top_k = 5;
            _, indices = index.search(question_embedding_np, top_k)
            retrieved_chunks = [chunks[i] for i in indices[0]]
            metrics = calculate_metrics(retrieved_chunks, gold_passages, top_k)

            all_results.append({
                'sample_id': data_point['sample_id'], 'experiment': exp_name,
                'chunking_time_s': chunking_time, 'num_chunks': len(chunks),
                'mrr': metrics['mrr'], 'precision_at_k': metrics['precision_at_k'], 'recall': metrics['recall']
            })

    print("\nAll experiments finished. Saving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, f'{timestamp}_detailed_results.csv'), index=False)

    summary_df = results_df.groupby('experiment').agg({
        'mrr': 'mean', 'precision_at_k': 'mean', 'recall': 'mean',
        'chunking_time_s': 'mean', 'num_chunks': 'mean'
    }).reset_index()

    summary_df.to_csv(os.path.join(output_dir, f'{timestamp}_summary_results.csv'), index=False)

    print("\n--- Aggregated Results ---")
    print(summary_df.to_string())

    visualize_and_save_results(summary_df, output_dir, timestamp)


if __name__ == "__main__":
    main()