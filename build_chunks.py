import argparse
import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import torch
from tqdm import tqdm

from src.chunking.chunk_fixed import chunk_fixed_size
from src.chunking.chunk_recursive import chunk_recursive
from src.chunking.chunk_semantic import chunk_semantic
from src.chunking.chunk_sentence import chunk_by_sentence
from src.experiment.data_loader import load_asqa_dataset
from src.vectorizer.vectorizer import Vectorizer


def get_chunking_function(name: str) -> Callable[..., list[str]]:
    chunk_functions = {
        "chunk_fixed_size": chunk_fixed_size,
        "chunk_by_sentence": chunk_by_sentence,
        "chunk_recursive": chunk_recursive,
        "chunk_semantic": chunk_semantic,
    }
    return chunk_functions[name]


def generate_chunks(
        experiment: dict[str, Any],
        dataset: list[dict[str, Any]],
        vectorizer: Vectorizer | None,
        cache_dir: str
) -> list[str]:
    """
    Generates chunks for a dataset based on experiment configuration,
    with caching and optimized batch processing for semantic chunking.
    """
    name = experiment["name"]
    func_name = experiment["function"]
    params = experiment.get("params", {})

    # Resolve the function
    chunk_func = get_chunking_function(func_name)

    # 1. Setup Paths
    exp_dir = os.path.join(cache_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    cache_path = os.path.join(exp_dir, "chunks.json")
    metadata_path = os.path.join(exp_dir, "metadata.json")
    sorted_cache_path = os.path.join(exp_dir, "chunks_SORTED.json")

    # 2. Check Cache
    if os.path.exists(cache_path):
        print(f"[{name}] Loading chunks from cache: {cache_path}")
        with open(cache_path) as f:
            chunks = json.load(f)

        # Ensure sorted version exists (self-healing cache)
        if not os.path.exists(sorted_cache_path):
            print(f"[{name}] Sorted chunks missing. Generating...")
            sorted_chunks = sorted(chunks, key=len)
            with open(sorted_cache_path, "w") as f:
                json.dump(sorted_chunks, f)

        return chunks

    # 3. Prepare Parameters
    call_params = params.copy()
    if func_name == "chunk_semantic":
        if vectorizer is None:
            raise ValueError(f"Experiment '{name}' requires a loaded Vectorizer.")
        call_params["chunking_embeddings"] = vectorizer

    # 4. Processing Loop (Batched)
    chunks = []
    start_time = time.time()

    # BATCH CONFIGURATION
    BATCH_SIZE = 16 if func_name == "chunk_semantic" else 1
    total_docs = len(dataset)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Iterate in batches
    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc=f"Chunking ({name})"):
        # Slice batch
        batch_slice = dataset[i: i + BATCH_SIZE]
        batch_texts = [d.get("document_text", "") for d in batch_slice]

        if not batch_texts:
            continue

        if func_name == "chunk_semantic":
            batch_chunks = chunk_func(batch_texts, **call_params)
            chunks.extend(batch_chunks)
        else:
            for text in batch_texts:
                if text:  # Skip empty strings
                    chunks.extend(chunk_func(text, **call_params))

    end_time = time.time()
    duration = end_time - start_time

    # 5. Statistics & Saving
    total_chunks = len(chunks)
    chunks_per_second = total_chunks / duration if duration > 0 else 0
    total_chars = sum(len(c) for c in chunks)
    avg_chars_per_chunk = total_chars / total_chunks if total_chunks > 0 else 0

    metadata = {
        "experiment_name": name,
        "function": func_name,
        "params": params,
        "source_documents_count": len(dataset),
        "total_chunks": total_chunks,
        "total_characters": total_chars,
        "avg_chars_per_chunk": avg_chars_per_chunk,
        "processing_time_seconds": duration,
        "chunks_per_second": chunks_per_second,
        "timestamp": datetime.now().isoformat(),
        "device": Vectorizer.get_device() if func_name == "chunk_semantic" else "cpu",
    }

    # Save Standard
    with open(cache_path, "w") as f:
        json.dump(chunks, f)

    # Save Metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save Sorted
    print(f"[{name}] Saving sorted chunks to {sorted_cache_path}...")
    sorted_chunks = sorted(chunks, key=len)
    with open(sorted_cache_path, "w") as f:
        json.dump(sorted_chunks, f)

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Generate chunks based on experiment config")
    parser.add_argument("--config", required=True, help="Path to experiment config json")
    parser.add_argument("--cache-dir", default="data/chunks", help="Directory to store generated chunks")

    # Re-added device arg to ensure you can control it if needed
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda, cpu, mps)")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Load shared resources
    dataset = load_asqa_dataset(config.get("input_file"), config.get("limit"))

    vec = None
    needs_vectorizer = any(exp["function"] == "chunk_semantic" for exp in config["experiments"])

    if needs_vectorizer or config.get("embedding_model"):
        if config.get("embedding_model"):
            print(f"Initializing Vectorizer on device: {args.device}")
            vec = Vectorizer.from_model_name(config["embedding_model"])
        elif needs_vectorizer:
            raise ValueError("Semantic chunking requires 'embedding_model' in config")

    for exp in config["experiments"]:
        generate_chunks(exp, dataset, vec, args.cache_dir)


if __name__ == "__main__":
    main()
