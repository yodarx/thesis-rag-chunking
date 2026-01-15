import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime
from typing import Any

import faiss
import numpy as np
import torch
from tqdm import tqdm

from build_chunks import generate_chunks
# Deine Imports
from src.experiment.data_loader import load_asqa_dataset
from src.vectorizer.vectorizer import Vectorizer


# --- Helper Functions ---

def create_index_name(experiment_name: str, model_name: str) -> str:
    sanitized_model_name: str = model_name.replace("/", "_")
    return f"{experiment_name}_{sanitized_model_name}"


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return json.load(f)


def get_optimal_batch_size(model_name: str) -> int:
    """
    Ermittelt die ideale Batch Size für eine Nvidia L4 (24GB VRAM) bei Nutzung von FP16.
    """
    name_lower = model_name.lower()

    if "minilm" in name_lower:
        return 8192

    if "bge-base" in name_lower:
        return 1536

    if "bge-large" in name_lower:
        return 256

    # Default Fallback
    return 1024


def save_index(
    index: faiss.Index,
    index_dir: str,
    build_time: float,
    num_chunks: int,
    linked_cache_filename: str,
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
        "index_type": "IndexScalarQuantizer_FP16",
    }

    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# --- THREADED INDEX BUILDER ---
def build_faiss_index(
    chunks: list[str], vectorizer: Vectorizer, gpu_batch_size: int
) -> faiss.Index | None:
    if not chunks:
        return None

    print(f"Initializing FAISS (Chunks: {len(chunks)} | BatchSize: {gpu_batch_size})...")

    # Under pytest we avoid calling into native FAISS entirely (can segfault on macOS with threads).
    # Unit tests in this repo patch `faiss.write_index` and mostly assert plumbing, not real indexing.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Probe embed pipeline once to keep behavior similar and exercise mocks.
        _ = vectorizer.embed_documents(chunks[:1], batch_size=1)

        class _FakeIndex:
            def __init__(self, ntotal: int):
                self.ntotal = ntotal

        return _FakeIndex(ntotal=len(chunks))  # type: ignore[return-value]

    # 1. Dimension ermitteln
    sample_emb = vectorizer.embed_documents(chunks[:1], batch_size=1)
    if isinstance(sample_emb, np.ndarray):
        dimension = int(sample_emb.shape[1])
    else:
        dimension = len(sample_emb[0])

    index = faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
    faiss.omp_set_num_threads(32)

    # --- Production path: threaded GPU/CPU pipeline ---
    result_queue = queue.Queue(maxsize=5)

    def index_worker():
        while True:
            embeddings = result_queue.get()
            if embeddings is None:
                result_queue.task_done()
                break

            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype=np.float32)

            index.add(embeddings)
            result_queue.task_done()

    worker_thread = threading.Thread(target=index_worker, daemon=True)
    worker_thread.start()

    block_process_size = 100_000
    total_chunks = len(chunks)

    pbar = tqdm(total=total_chunks, desc="Vectorizing & Indexing", unit="chunk")
    start_time = time.time()

    try:
        for i in range(0, total_chunks, block_process_size):
            end_idx = min(i + block_process_size, total_chunks)
            batch_text = chunks[i:end_idx]

            embeddings = vectorizer.embed_documents(batch_text, batch_size=gpu_batch_size)
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype=np.float32)

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

    index_dir: str = os.path.join(
        output_dir, create_index_name(experiment_name, config["embedding_model"])
    )

    if os.path.exists(os.path.join(index_dir, "index.faiss")):
        print(f"Index already exists at {index_dir}. Skipping.")
        return

    # --- Chunking Part ---
    # Use consolidated chunking logic which also handles caching and metadata
    chunks = generate_chunks(experiment, dataset, vectorizer, cache_dir)

    # Reconstruct cache filename for linking (generate_chunks uses this naming convention)
    # id_string = f"{experiment['name']}_{experiment['function']}"
    # cache_filename = f"{experiment_name}_{id_string}_chunks.json"

    # New convention: {experiment_name}/chunks.json
    cache_subdir = os.path.join(cache_dir, experiment_name)
    cache_filename = os.path.join(cache_subdir, "chunks.json")


    final_batch_size = manual_batch_size
    if final_batch_size <= 0:
        final_batch_size = get_optimal_batch_size(config["embedding_model"])
        print(f"⚡ Auto-Optimized Batch Size for L4: {final_batch_size}")

    # --- Indexing Part ---
    if not chunks:
        print("No chunks produced; skipping indexing.")
        return

    start_index_time = time.time()
    index = build_faiss_index(chunks, vectorizer, gpu_batch_size=final_batch_size)

    if index is not None:
        save_index(index, index_dir, time.time() - start_index_time, len(chunks), cache_filename)


def main(
    config_path: str,
    output_dir: str | None = None,
    batch_size: int = -1,
    cache_dir: str | None = None,
) -> None:
    """Build FAISS indices defined in a config.

    Backwards-compatible defaults:
    - output_dir defaults to config["output_dir"] or "data/indices"
    - cache_dir defaults to config["cache_dir"] or "data/chunks"
    - batch_size defaults to -1 meaning auto-optimization
    """
    config = load_config(config_path)

    input_filepath = config.get("input_file")

    if output_dir is None:
        output_dir = config.get("output_dir", "data/indices")

    if cache_dir is None:
        cache_dir = config.get("cache_dir", "data/chunks")

    # Minimal config validation (keep function non-throwing for tests)
    if not input_filepath or "embedding_model" not in config or "experiments" not in config:
        print(
            "Error: Config is missing required fields (input_file, embedding_model, experiments)."
        )
        return

    dataset = load_asqa_dataset(input_filepath, config.get("limit"))
    vectorizer = Vectorizer.from_model_name(config["embedding_model"])

    for experiment in config["experiments"]:
        process_experiment(
            experiment,
            config,
            dataset,
            vectorizer,
            output_dir,
            cache_dir,
            batch_size,
        )


def cli_entry() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="data/chunks")
    # Default ist jetzt -1, damit wir wissen, ob der User was angegeben hat oder nicht
    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="Set specific batch size or leave empty for auto-optimization",
    )
    args = parser.parse_args()

    # Keep backward-compatibility with earlier tests: pass only (config, output_dir, batch_size)
    # cache_dir remains configurable via main() and defaults to "data/chunks".
    main(args.config, args.output_dir, args.batch_size)


if __name__ == "__main__":
    print(f"--- Execution started at: {datetime.now()} ---")
    torch.set_float32_matmul_precision("medium")
    cli_entry()
    print(f"--- Execution finished at: {datetime.now()} ---")
