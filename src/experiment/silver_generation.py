import json
import os
from typing import Any

from langchain_community.llms import Ollama

from src.experiment.retriever import FaissRetriever
from src.experiment.silver_standard import SilverStandardGenerator
from src.experiment.utils import create_index_name
from src.vectorizer.vectorizer import Vectorizer


def generate_silver_for_experiment(
    experiment: dict[str, Any],
    config: dict[str, Any],
    vectorizer: Vectorizer,
    llm: Ollama,
    limit: int,
) -> str:
    """
    Generates a silver standard dataset for a specific experiment.
    Returns the path to the generated file.
    """
    index_name = create_index_name(experiment["name"], config["embedding_model"])
    index_dir = os.path.join("data", "indices", index_name)
    chunks_path = os.path.join(index_dir, "chunks.json")

    output_dir = os.path.join("data", "silver")
    output_file = os.path.join(output_dir, f"{index_name}_silver.jsonl")

    if not os.path.exists(chunks_path):
        print(f"Skipping {index_name}: Chunks file not found at {chunks_path}")
        return ""

    if os.path.exists(output_file):
        print(f"Skipping {index_name}: Silver standard already exists at {output_file}")
        return output_file

    print(f"Generating silver standard for {index_name}...")

    # Initialize retriever with chunks only (no index needed for random sampling)
    retriever = FaissRetriever(vectorizer)
    with open(chunks_path, encoding="utf-8") as f:
        retriever.chunks = json.load(f)

    # Adjust limit if it's -1 (all) or larger than available chunks
    current_limit = limit
    if current_limit == -1 or current_limit > len(retriever.chunks):
        current_limit = len(retriever.chunks)

    hops_count = config.get("hops_count", 2)
    generator = SilverStandardGenerator(retriever, llm)
    dataset = generator.generate_dataset(current_limit, num_hops=hops_count)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} samples to {output_file}")

    return output_file
