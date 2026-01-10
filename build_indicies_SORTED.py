import argparse
import json
import os
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import faiss
import torch
from tqdm import tqdm

# Deine Imports
from src.chunking.chunk_fixed import chunk_fixed_size
from src.chunking.chunk_recursive import chunk_recursive
from src.chunking.chunk_semantic import chunk_semantic
from src.chunking.chunk_sentence import chunk_by_sentence
from src.experiment.data_loader import load_asqa_dataset
from src.vectorizer.vectorizer import Vectorizer


# --- Helper Functions ---
def get_chunking_function(name: str) -> Callable[..., list[str]]:
    chunk_functions: dict[str, Callable[..., list[str]]] = {
        "chunk_fixed_size": chunk_fixed_size,
        "chunk_by_sentence": chunk_by_sentence,
        "chunk_recursive": chunk_recursive,
        "chunk_semantic": chunk_semantic,
    }
    if name not in chunk_functions:
        raise ValueError(f"Unknown chunking function: {name}")
    return chunk_functions[name]


def create_index_name(experiment_name: str, model_name: str) -> str:
    sanitized_model_name: str = model_name.replace("/", "_")
    return f"{experiment_name}_{sanitized_model_name}"


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return json.load(f)


def get_optimal_batch_size(model_name: str) -> int:
    """
    Ermittelt die ideale Batch Size für eine Nvidia L4 (24GB VRAM) bei Nutzung von FP16.
    ACHTUNG: Bei sortierten Daten (Smart Batching) sind die Batches "dichter" (weniger Padding).
    Wir müssen etwas konservativer sein als bei zufälligen Daten.
    """
    name_lower = model_name.lower()

    if "minilm" in name_lower:
        return 8192

    if "bge-base" in name_lower:
        # Zufällig gingen 3000. Sortiert (dicht) sind 2048-2560 sicherer.
        return 2048

    if "bge-large" in name_lower:
        # Große Modelle sind bei vollen 512-Token-Batches sehr speicherhungrig.
        return 192

    # Default Fallback
    return 1024


def save_index(
        index: faiss.Index,
        index_dir: str,
        build_time: float,
        num_chunks: int,
        linked_cache_filename: str
) -> None:
    os.makedirs(index_dir, exist_ok=True)
    print(f"Saving FAISS index (FP16) to {index_dir}...")
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    metadata: dict[str, Any] = {
        "indexing_compute_duration_seconds": build_time,
        "num_chunks": num_chunks,
        "linked_cache_file": linked_cache_filename,  # WICHTIG: Verweis auf die SORTIERTE Datei
        "timestamp": datetime.now().isoformat(),
        "faiss_ntotal": index.ntotal,
        "index_type": "IndexScalarQuantizer_FP16",
        "optimization": "global_sort_fp16"
    }

    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# --- THREADED INDEX BUILDER ---
def build_faiss_index(chunks: list[str], vectorizer: Vectorizer, gpu_batch_size: int) -> faiss.Index | None:
    if not chunks:
        return None

    print(f"Initializing FAISS (Chunks: {len(chunks)} | BatchSize: {gpu_batch_size})...")

    # 1. Dimension ermitteln
    sample_emb = vectorizer.embed_documents(chunks[:1], batch_size=1)
    dimension = sample_emb.shape[1]

    # 2. Index erstellen: ScalarQuantizer FP16 (Spart 50% RAM)
    index = faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)

    # 3. Setup Threads
    faiss.omp_set_num_threads(32)  # Nutze alle CPU Cores für den Index

    # 4. Threading Pipeline
    # Wir entkoppeln GPU (Vectorizing) und CPU (Index Add)
    result_queue = queue.Queue(maxsize=5)

    def index_worker():
        while True:
            embeddings = result_queue.get()
            if embeddings is None:
                result_queue.task_done()
                break
            index.add(embeddings)
            result_queue.task_done()

    worker_thread = threading.Thread(target=index_worker, daemon=True)
    worker_thread.start()

    # 5. Main Loop (GPU)
    # Wir laden Daten in großen Blöcken, damit der RAM nicht explodiert
    block_process_size = 100_000
    total_chunks = len(chunks)

    pbar = tqdm(total=total_chunks, desc="🚀 Sorted Indexing", unit="chunk")
    start_time = time.time()

    try:
        for i in range(0, total_chunks, block_process_size):
            end_idx = min(i + block_process_size, total_chunks)
            batch_text = chunks[i:end_idx]

            # GPU Arbeit
            embeddings = vectorizer.embed_documents(batch_text, batch_size=gpu_batch_size)

            # Übergabe an CPU Worker
            result_queue.put(embeddings)
            pbar.update(len(batch_text))

    except KeyboardInterrupt:
        print("\nInterrupted! Stopping threads...")
        result_queue.put(None)
        worker_thread.join()
        return None

    result_queue.put(None)
    worker_thread.join()
    pbar.close()

    duration = time.time() - start_time
    print(f"Index built. Speed: {total_chunks / duration:.1f} chunks/sec")
    return index


# --- EXPERIMENT LOOP ---
def process_experiment(
        experiment: dict[str, Any],
        config: dict[str, Any],
        dataset: list[dict[str, Any]],
        vectorizer: Vectorizer,
        output_dir: str,
        cache_dir: str,
        manual_batch_size: int,  # Kann durch CLI überschrieben werden
) -> None:
    experiment_name = experiment["name"]
    id_string = f"{experiment['name']}_{experiment['function']}"

    cache_filename = f"{experiment_name}_{id_string}_chunks.json"
    meta_filename = f"{experiment_name}_{id_string}_metadata.json"
    cache_path = os.path.join(cache_dir, cache_filename)
    meta_path = os.path.join(cache_dir, meta_filename)

    index_dir: str = os.path.join(
        output_dir, create_index_name(experiment_name, config["embedding_model"])
    )

    if os.path.exists(os.path.join(index_dir, "index.faiss")):
        print(f"Index already exists at {index_dir}. Skipping.")
        return

    # --- 1. Chunking Part (Load or Generate) ---
    chunks: list[str] = []
    if os.path.exists(cache_path):
        print(f"Cache hit! Loading chunks from {cache_path}...")
        with open(cache_path, encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        print(f"Cache miss. Starting chunking for {experiment_name}...")
        chunk_start_time = time.time()
        chunk_func = get_chunking_function(experiment["function"])

        for data_point in tqdm(dataset, desc="Chunking Docs"):
            text = data_point.get("document_text", "")
            params = experiment["params"]
            if experiment["function"] == "chunk_semantic":
                params = params.copy()
                params["chunking_embeddings"] = vectorizer
            chunks.extend(chunk_func(text, **params))

        os.makedirs(cache_dir, exist_ok=True)
        # Speichern der unsortierten Original-Chunks (zur Sicherheit)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        # Save Chunk Metadata
        chunk_metadata = {
            "experiment_name": experiment_name,
            "chunking_function": experiment["function"],
            "parameters": experiment["params"],
            "num_chunks": len(chunks),
            "chunking_duration_seconds": time.time() - chunk_start_time,
            "created_at": datetime.now().isoformat()
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunk_metadata, f, indent=2)

    print(f"⚡ Sorting {len(chunks)} chunks by length to minimize padding...")
    chunks.sort(key=len)

    sorted_cache_filename = cache_filename.replace(".json", "_SORTED.json")
    sorted_cache_path = os.path.join(cache_dir, sorted_cache_filename)

    print(f"💾 Saving sorted chunks map to {sorted_cache_path}...")
    with open(sorted_cache_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    # --- 3. Batch Size Logic ---
    final_batch_size = manual_batch_size
    if final_batch_size <= 0:
        final_batch_size = get_optimal_batch_size(config["embedding_model"])
        print(f"⚡ Auto-Optimized Batch Size for L4 (Sorted/Dense): {final_batch_size}")

    # --- 4. Indexing Part ---
    start_index_time = time.time()
    index = build_faiss_index(chunks, vectorizer, gpu_batch_size=final_batch_size)

    if index is not None:
        # Wir übergeben hier den neuen sorted_cache_filename!
        save_index(index, index_dir, time.time() - start_index_time, len(chunks), sorted_cache_filename)


def main(config_path: str, output_dir: str | None, cache_dir: str | None, batch_size: int) -> None:
    config = load_config(config_path)

    input_filepath = config.get("input_file")
    if output_dir is None: output_dir = config.get("output_dir", "data/indices")
    if cache_dir is None: cache_dir = "data/chunks"

    dataset = load_asqa_dataset(input_filepath, config.get("limit"))
    vectorizer = Vectorizer.from_model_name(config["embedding_model"])

    for experiment in config["experiments"]:
        process_experiment(experiment, config, dataset, vectorizer, output_dir, cache_dir, batch_size)


def cli_entry() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="data/chunks")
    # Default ist jetzt -1, damit wir wissen, ob der User was angegeben hat oder nicht
    parser.add_argument("--batch-size", type=int, default=-1,
                        help="Set specific batch size or leave empty for auto-optimization")
    args = parser.parse_args()

    main(args.config, args.output_dir, args.cache_dir, args.batch_size)


if __name__ == "__main__":
    print(f"--- Execution started at: {datetime.now()} ---")
    torch.set_float32_matmul_precision('medium')
    cli_entry()
    print(f"--- Execution finished at: {datetime.now()} ---")
