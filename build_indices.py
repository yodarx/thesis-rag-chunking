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

# Deine Imports (wie gehabt)
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
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {e}") from e


def save_index(
        index: faiss.Index,  # Typ ist jetzt allgemeiner Index
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
        "linked_cache_file": linked_cache_filename,
        "timestamp": datetime.now().isoformat(),
        "faiss_ntotal": index.ntotal,
        "index_type": "IndexScalarQuantizer_FP16"  # Vermerk für später
    }

    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved. Refer to {linked_cache_filename} for text chunks.")


# --- THREADED INDEX BUILDER (FP16 VERSION) ---
def build_faiss_index(chunks: list[str], vectorizer: Vectorizer, gpu_batch_size: int = 4096) -> faiss.Index | None:
    if not chunks:
        print("Warning: No chunks provided for indexing.")
        return None

    print(f"Initializing FAISS index (Total chunks: {len(chunks)})...")

    # 1. Dimension ermitteln
    sample_emb = vectorizer.embed_documents(chunks[:1], batch_size=1)
    dimension = sample_emb.shape[1]

    # 2. Index erstellen: ScalarQuantizer mit FP16
    # Das ist der Schlüssel: Es speichert Vektoren als 16-Bit Floats statt 32-Bit.
    # Keine Qualitätseinbußen für Retrieval, aber 50% RAM gespart.
    print(f"Creating IndexScalarQuantizer (QT_fp16) for dimension {dimension}...")
    index = faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)

    # 3. Setup CPU Threads
    faiss.omp_set_num_threads(28)

    # 4. Threading Pipeline
    result_queue = queue.Queue(maxsize=10)

    def index_worker():
        while True:
            embeddings = result_queue.get()
            if embeddings is None:
                result_queue.task_done()
                break
            index.add(embeddings)  # Funktioniert automatisch mit FP16
            result_queue.task_done()

    worker_thread = threading.Thread(target=index_worker, daemon=True)
    worker_thread.start()

    # 5. Main Loop (GPU)
    block_process_size = 100_000
    total_chunks = len(chunks)

    pbar = tqdm(total=total_chunks, desc="Vectorizing & Indexing", unit="chunk")

    try:
        for i in range(0, total_chunks, block_process_size):
            end_idx = min(i + block_process_size, total_chunks)
            batch_text = chunks[i:end_idx]

            embeddings = vectorizer.embed_documents(batch_text, batch_size=gpu_batch_size)
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

    print(f"Index built. Final size: {index.ntotal}")
    return index


# --- EXPERIMENT LOOP (Unverändert) ---
def process_experiment(
        experiment: dict[str, Any],
        config: dict[str, Any],
        dataset: list[dict[str, Any]],
        vectorizer: Vectorizer,
        output_dir: str,
        cache_dir: str,
        batch_size: int = 4096,
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

    chunks: list[str] = []

    if os.path.exists(cache_path):
        print(f"Cache hit! Loading chunks from {cache_path}...")
        with open(cache_path, encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks.")
    else:
        print(f"Cache miss. Starting chunking for {experiment_name}...")
        chunk_start_time = time.time()

        chunk_func: Callable[..., list[str]] = get_chunking_function(experiment["function"])
        for data_point in tqdm(dataset, desc="Chunking Docs"):
            text: str = data_point.get("document_text", "")
            params: dict[str, Any] = experiment["params"]
            if experiment["function"] == "chunk_semantic":
                params = params.copy()
                params["chunking_embeddings"] = vectorizer
            chunks.extend(chunk_func(text, **params))

        chunk_duration = time.time() - chunk_start_time

        os.makedirs(cache_dir, exist_ok=True)
        print(f"Saving chunks to cache: {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f)

        chunk_metadata = {
            "experiment_name": experiment_name,
            "chunking_function": experiment["function"],
            "parameters": experiment["params"],
            "num_chunks": len(chunks),
            "chunking_duration_seconds": chunk_duration,
            "created_at": datetime.now().isoformat()
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunk_metadata, f, indent=2)

    start_index_time = time.time()

    index = build_faiss_index(chunks, vectorizer, gpu_batch_size=batch_size)

    build_time = time.time() - start_index_time

    if index is not None:
        save_index(index, index_dir, build_time, len(chunks), cache_filename)


def main(config_path: str, output_dir: str | None = None, cache_dir: str | None = None, batch_size: int = 4096) -> None:
    config: dict[str, Any] = load_config(config_path)
    input_filepath: str | None = config.get("input_file")

    if output_dir is None:
        output_dir = config.get("output_dir", "data/indices")
    if cache_dir is None:
        cache_dir = "data/chunks"

    dataset: list[dict[str, Any]] = load_asqa_dataset(input_filepath, config.get("limit"))
    vectorizer: Vectorizer = Vectorizer.from_model_name(config["embedding_model"])

    for experiment in config["experiments"]:
        process_experiment(experiment, config, dataset, vectorizer, output_dir, cache_dir, batch_size)


def cli_entry() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="data/chunks")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    main(args.config, args.output_dir, args.cache_dir, args.batch_size)


if __name__ == "__main__":
    print(f"--- Execution started at: {datetime.now()} ---")
    torch.set_float32_matmul_precision('medium')
    cli_entry()
    print(f"--- Execution finished at: {datetime.now()} ---")
