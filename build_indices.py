# build_indices.py
import argparse
import json
import os
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


def get_chunking_function(name: str):
    """Maps a string name to a chunking function."""
    chunk_functions = {
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
    sanitized_model_name = model_name.replace("/", "_")
    return f"{experiment_name}_{sanitized_model_name}"


def main(config_path: str) -> None:
    """
    Builds and saves FAISS indices for all chunking strategies defined in a config file.
    """
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    input_filepath = config.get("input_file")
    if not input_filepath:
        print("Config missing 'input_file'. Exiting.")
        return
    limit = config.get("limit")
    embedding_model_name = config.get("embedding_model")
    experiments = config.get("experiments", [])
    if not embedding_model_name or not experiments:
        print("Config missing 'embedding_model' or 'experiments'. Exiting.")
        return

    # --- Load Dataset ---
    print(f"Loading dataset from '{input_filepath}'...")
    dataset = load_asqa_dataset(input_filepath, limit=limit)
    if not dataset:
        print("Dataset could not be loaded. Exiting.")
        return

    # --- Initialize Vectorizer ---
    print(f"Initializing vectorizer with model '{embedding_model_name}'...")
    vectorizer = Vectorizer.from_model_name(model_name=embedding_model_name)

    # --- Base directory for all indices ---
    base_index_dir = "indices"
    os.makedirs(base_index_dir, exist_ok=True)
    print(f"Indices will be saved in '{base_index_dir}/'")

    # --- Process each chunking experiment ---
    for experiment in experiments:
        exp_name = experiment.get("name")
        chunk_func_name = experiment.get("function")
        chunk_params = experiment.get("params", {}).copy()
        if not exp_name or not chunk_func_name:
            print("Experiment missing 'name' or 'function'. Skipping.")
            continue
        try:
            chunk_function = get_chunking_function(chunk_func_name)
        except ValueError as e:
            print(f"{e}. Skipping experiment '{exp_name}'.")
            continue

        index_folder_name = create_index_name(exp_name, embedding_model_name)
        output_dir = os.path.join(base_index_dir, index_folder_name)

        # --- Check if index already exists ---
        if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "index.faiss")):
            print(f"\nSkipping experiment '{exp_name}': Index already exists at '{output_dir}'")
            continue

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing experiment: '{exp_name}'")
        print(f"  - Index directory: '{output_dir}'")

        # Special handling for semantic chunking which requires a Vectorizer instance
        if (
            chunk_func_name == "chunk_semantic"
            and "chunking_embeddings" in chunk_params
            and isinstance(chunk_params["chunking_embeddings"], str)
        ):
            model_name = chunk_params["chunking_embeddings"]
            print(f"  - Initializing semantic chunking model: {model_name}")
            chunk_params["chunking_embeddings"] = Vectorizer.from_model_name(model_name)

        all_chunks: list[str] = []
        chunk_metadata: list[dict[str, Any]] = []

        print("  - Generating chunks for all documents...")
        for doc in tqdm(dataset, desc="Chunking"):
            doc_id = doc.get("sample_id")
            full_context = doc.get("document_text")
            if not doc_id or not full_context:
                continue
            try:
                chunks = chunk_function(full_context, **chunk_params)
            except Exception as e:
                print(f"    Error chunking doc {doc_id}: {e}")
                continue
            for i, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                chunk_metadata.append({"doc_id": doc_id, "chunk_index": i})

        if not all_chunks:
            print(f"  - No chunks were generated for '{exp_name}'. Skipping.")
            continue

        # --- Create Embeddings and Build Index in Batches to save memory ---
        print(f"  - Creating {len(all_chunks)} embeddings and building FAISS index...")

        try:
            first_embedding = vectorizer.embed_documents([all_chunks[0]])
            dimension = np.array(first_embedding).shape[1]
        except Exception as e:
            print(f"  - Error creating first embedding: {e}. Skipping experiment.")
            continue

        nlist = 1024  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        print(f"  - Training index on a sample of {min(len(all_chunks), 256 * nlist)} vectors...")
        num_train_vectors = min(len(all_chunks), 256 * nlist)
        train_indices = np.random.choice(len(all_chunks), num_train_vectors, replace=False)
        train_chunks = [all_chunks[i] for i in train_indices]
        try:
            train_embeddings = vectorizer.embed_documents(train_chunks)
            train_embeddings_np = np.array(train_embeddings, dtype="float32")
            index.train(train_embeddings_np)
        except Exception as e:
            print(f"  - Error training index: {e}. Skipping experiment.")
            continue
        print("  - Index training complete.")

        batch_size = 16384
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing Batches"):
            batch_chunks = all_chunks[i : i + batch_size]
            if not batch_chunks:
                continue
            try:
                batch_embeddings = vectorizer.embed_documents(batch_chunks)
                batch_embeddings_np = np.array(batch_embeddings, dtype="float32")
                index.add(batch_embeddings_np)
            except Exception as e:
                print(f"    Error indexing batch {i // batch_size}: {e}")
                continue

        print(f"  - FAISS index built with {index.ntotal} vectors.")

        index_path = os.path.join(output_dir, "index.faiss")
        try:
            faiss.write_index(index, index_path)
            print(f"  - FAISS index saved to '{index_path}'")
        except Exception as e:
            print(f"  - Error saving index: {e}")

        chunks_path = os.path.join(output_dir, "chunks.json")
        try:
            with open(chunks_path, "w") as f:
                json.dump(all_chunks, f)
        except Exception as e:
            print(f"  - Error saving chunks: {e}")

        metadata_path = os.path.join(output_dir, "metadata.json")
        try:
            with open(metadata_path, "w") as f:
                json.dump(chunk_metadata, f)
            print("  - Chunks and metadata saved.")
        except Exception as e:
            print(f"  - Error saving metadata: {e}")

    print("\nAll indices built successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS indices for RAG experiments.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment config JSON file (e.g., configs/base_experiment.json).",
    )
    args = parser.parse_args()

    print(f"--- Execution started at: {datetime.now()} ---")
    main(config_path=args.config)
    print(f"--- Execution finished at: {datetime.now()} ---")
