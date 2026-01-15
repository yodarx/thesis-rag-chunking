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
        exp_config: dict[str, Any],
        dataset: list[dict[str, Any]],
        vectorizer: Vectorizer,
        cache_dir: str
) -> list[str]:
    """
    Loads chunks from cache or generates them if not present.
    """
    name = exp_config["name"]
    func_name = exp_config["function"]

    # Create a directory for the experiment chunks
    exp_dir = os.path.join(cache_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    cache_name = "chunks.json"
    metadata_name = "metadata.json"
    cache_path = os.path.join(exp_dir, cache_name)
    metadata_path = os.path.join(exp_dir, metadata_name)

    if os.path.exists(cache_path):
        print(f"[{name}] Lade Chunks aus Cache...")
        with open(cache_path) as f:
            chunks = json.load(f)

        # Ensure sorted chunks exist
        sorted_cache_path = os.path.join(exp_dir, "chunks_SORTED.json")
        if not os.path.exists(sorted_cache_path):
            print(f"[{name}] Sorted chunks missing. Generating sorted file...")
            sorted_chunks = sorted(chunks, key=len)
            with open(sorted_cache_path, "w") as f:
                json.dump(sorted_chunks, f)

        return chunks

    print(f"[{name}] ⚠️ Cache Miss! Generiere Chunks...")
    chunk_func = get_chunking_function(func_name)
    params = exp_config.get("params", {})

    # Copy params and inject vectorizer if needed
    call_params = params.copy()
    if func_name == "chunk_semantic":
        call_params["chunking_embeddings"] = vectorizer
    chunks = []
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i, d in enumerate(tqdm(dataset, desc=f"Chunking ({name})")):
        text = d.get("document_text", "")
        chunks.extend(chunk_func(text, **call_params))

        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    end_time = time.time()
    duration = end_time - start_time

    # Calculate statistics
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
    }

    with open(cache_path, "w") as f:
        json.dump(chunks, f)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save sorted chunks
    sorted_cache_path = os.path.join(exp_dir, "chunks_SORTED.json")
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
            # Ensure your Vectorizer class accepts device!
            vec = Vectorizer.from_model_name(config["embedding_model"], device=args.device)
        elif needs_vectorizer:
            raise ValueError("Semantic chunking requires 'embedding_model' in config")

    for exp in config["experiments"]:
        generate_chunks(exp, dataset, vec, args.cache_dir)


if __name__ == "__main__":
    main()
