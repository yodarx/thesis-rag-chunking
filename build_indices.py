# build_indices.py
import argparse
import json
import os
import sys
from datetime import datetime

import faiss
import numpy as np
from tqdm import tqdm

from src.chunking.chunk_fixed import chunk_fixed_size
from src.chunking.chunk_recursive import chunk_recursive
from src.chunking.chunk_semantic import chunk_semantic
from src.chunking.chunk_sentence import chunk_by_sentence
from src.experiment.data_loader import load_asqa_dataset
from src.vectorizer.vectorizer import Vectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def get_chunking_function(name):
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
    # Sanitize model name for use in file paths
    sanitized_model_name = model_name.replace("/", "_")
    return f"{experiment_name}_{sanitized_model_name}"


def main(config_path: str):
    """
    Builds and saves FAISS indices for all chunking strategies defined in a config file.
    """
    with open(config_path) as f:
        config = json.load(f)

    input_filepath = config["input_file"]
    limit = config.get("limit")
    embedding_model_name = config["embedding_model"]
    experiments = config["experiments"]

    # --- Load Dataset ---
    print(f"Loading dataset from '{input_filepath}'...")
    dataset = load_asqa_dataset(input_filepath, limit=limit)
    if not dataset:
        print("Dataset could not be loaded. Exiting.")
        sys.exit(1)

    # --- Initialize Vectorizer ---
    print(f"Initializing vectorizer with model '{embedding_model_name}'...")
    vectorizer = Vectorizer.from_model_name(model_name=embedding_model_name)

    # --- Base directory for all indices ---
    base_index_dir = "indices"
    os.makedirs(base_index_dir, exist_ok=True)
    print(f"Indices will be saved in '{base_index_dir}/'")

    # --- Process each chunking experiment ---
    for experiment in experiments:
        exp_name = experiment["name"]
        chunk_func_name = experiment["function"]
        chunk_params = experiment["params"].copy()
        chunk_function = get_chunking_function(chunk_func_name)

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
        if chunk_func_name == "chunk_semantic":
            if "chunking_embeddings" in chunk_params and isinstance(chunk_params["chunking_embeddings"], str):
                model_name = chunk_params["chunking_embeddings"]
                print(f"  - Initializing semantic chunking model: {model_name}")
                chunk_params["chunking_embeddings"] = Vectorizer.from_model_name(model_name)

        all_chunks = []
        chunk_metadata = []

        print("  - Generating chunks for all documents...")
        for doc in tqdm(dataset, desc="Chunking"):
            doc_id = doc["sample_id"]
            full_context = doc["document_text"]
            chunks = chunk_function(full_context, **chunk_params)
            for i, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                chunk_metadata.append({"doc_id": doc_id, "chunk_index": i})

        if not all_chunks:
            print(f"  - No chunks were generated for '{exp_name}'. Skipping.")
            continue

        # --- Create Embeddings and Build Index in Batches to save memory ---
        print(f"  - Creating {len(all_chunks)} embeddings and building FAISS index...")

        # 1. Get dimension from the first chunk's embedding
        first_embedding = vectorizer.embed_documents([all_chunks[0]])
        dimension = np.array(first_embedding).shape[1]

        # 2. Create an IndexIVFFlat index
        nlist = 1024  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        # 3. Train the index on a subset of the data
        print(f"  - Training index on a sample of {min(len(all_chunks), 256 * nlist)} vectors...")
        # FAISS recommends training on 30x to 256x the number of clusters
        num_train_vectors = min(len(all_chunks), 256 * nlist)
        train_indices = np.random.choice(len(all_chunks), num_train_vectors, replace=False)

        # Generate embeddings only for the training set
        train_chunks = [all_chunks[i] for i in train_indices]
        train_embeddings = vectorizer.embed_documents(train_chunks)
        train_embeddings_np = np.array(train_embeddings, dtype='float32')

        index.train(train_embeddings_np)
        print("  - Index training complete.")

        # 4. Add all vectors to the index in batches
        batch_size = 16384

        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing Batches"):
            batch_chunks = all_chunks[i:i + batch_size]
            if not batch_chunks:
                continue

            batch_embeddings = vectorizer.embed_documents(batch_chunks)
            batch_embeddings_np = np.array(batch_embeddings, dtype='float32')
            index.add(batch_embeddings_np)

        print(f"  - FAISS index built with {index.ntotal} vectors.")

        index_path = os.path.join(output_dir, "index.faiss")
        faiss.write_index(index, index_path)
        print(f"  - FAISS index saved to '{index_path}'")

        # --- Save Chunks and Metadata ---
        chunks_path = os.path.join(output_dir, "chunks.json")
        with open(chunks_path, "w") as f:
            json.dump(all_chunks, f)

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(chunk_metadata, f)
        print(f"  - Chunks and metadata saved.")

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
