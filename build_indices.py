# build_indices.py
import argparse
import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

from src.chunking.chunk_fixed import chunk_fixed_size
from src.chunking.chunk_recursive import chunk_recursive
from src.chunking.chunk_semantic import chunk_semantic
from src.chunking.chunk_sentence import chunk_by_sentence
from src.experiment.data_loader import load_asqa_dataset
from src.vectorizer.vectorizer import Vectorizer


def get_chunking_function(name: str) -> Callable[..., list[str]]:
    """Maps a string name to a chunking function."""
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
    """Creates a descriptive name for the index directory."""
    sanitized_model_name: str = model_name.replace("/", "_")
    return f"{experiment_name}_{sanitized_model_name}"


def load_config(config_path: str) -> dict[str, Any]:
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {e}") from e


def build_faiss_index(chunks: list[str], vectorizer: Vectorizer) -> faiss.IndexFlatL2 | None:
    if not chunks:
        print("Warning: No chunks provided for indexing. Skipping index creation.")
        return None
    embeddings: np.ndarray = np.array(vectorizer.embed_documents(chunks)).astype("float32")
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        print("Warning: Embeddings are empty or malformed. Skipping index creation.")
        return None
    dimension: int = embeddings.shape[1]
    index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatL2, index_dir: str, chunks: list[str], build_time: float) -> None:
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    # Save metadata with build time
    metadata: dict[str, Any] = {
        "build_time_seconds": build_time,
        "num_chunks": len(chunks),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def process_experiment(
    experiment: dict[str, Any],
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    vectorizer: Vectorizer,
    output_dir: str,
) -> None:
    chunk_func: Callable[..., list[str]] = get_chunking_function(experiment["function"])
    index_dir: str = os.path.join(
        output_dir, create_index_name(experiment["name"], config["embedding_model"])
    )
    index_path = os.path.join(index_dir, "index.faiss")
    chunks_path = os.path.join(index_dir, "chunks.json")
    metadata_path = os.path.join(index_dir, "metadata.json")
    if os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(metadata_path):
        print(f"Index already exists for {experiment['name']} at {index_dir}. Skipping.")
        return

    start_time: float = time.time()
    chunks: list[str] = []
    for data_point in tqdm(dataset, desc=f"Chunking {experiment['name']}"):
        text: str = data_point.get("document_text", "")
        params: dict[str, Any] = experiment["params"]
        if experiment["function"] == "chunk_semantic":
            params = params.copy()
            params["chunking_embeddings"] = vectorizer
        chunks.extend(chunk_func(text, **params))

    index: faiss.IndexFlatL2 = build_faiss_index(chunks, vectorizer)
    if index is not None:
        build_time: float = time.time() - start_time
        save_index(index, index_dir, chunks, build_time)


def main(config_path: str, output_dir: str | None = None) -> None:
    config: dict[str, Any] = load_config(config_path)
    input_filepath: str | None = config.get("input_file")
    if "embedding_model" not in config:
        print("Error: 'embedding_model' missing from config. Skipping.")
        return
    # Use argument if provided, otherwise fall back to config, then default
    if output_dir is None:
        output_dir = config.get("output_dir", "data/indices")
    dataset: list[dict[str, Any]] = load_asqa_dataset(input_filepath, config.get("limit"))
    vectorizer: Vectorizer = Vectorizer.from_model_name(config["embedding_model"])
    for experiment in config["experiments"]:
        process_experiment(experiment, config, dataset, vectorizer, output_dir)


def cli_entry() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indices for chunking experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for indices (overrides config setting).")
    args = parser.parse_args()
    main(args.config, args.output_dir)


if __name__ == "__main__":
    print(f"--- Execution started at: {datetime.now()} ---")
    cli_entry()
    print(f"--- Execution finished at: {datetime.now()} ---")
