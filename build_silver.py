import argparse
import json
import os
from typing import Any

from langchain_community.llms import Ollama
from tqdm import tqdm

from src.experiment.retriever import FaissRetriever
from src.experiment.silver_standard import SilverStandardGenerator
from src.vectorizer.vectorizer import Vectorizer


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


def generate_silver_for_experiment(
    experiment: dict[str, Any],
    config: dict[str, Any],
    vectorizer: Vectorizer,
    llm: Ollama,
    limit: int,
) -> None:
    index_name = create_index_name(experiment["name"], config["embedding_model"])
    index_dir = os.path.join("data", "indices", index_name)
    chunks_path = os.path.join(index_dir, "chunks.json")

    output_dir = os.path.join("data", "silver")
    output_file = os.path.join(output_dir, f"{index_name}_silver.jsonl")

    if not os.path.exists(chunks_path):
        print(f"Skipping {index_name}: Chunks file not found at {chunks_path}")
        return

    if os.path.exists(output_file):
        print(f"Skipping {index_name}: Silver standard already exists at {output_file}")
        return

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


def main(config_path: str) -> None:
    config = load_config(config_path)

    # Default limit or from config
    limit = config.get("silver_limit", 10)
    model_name = config.get("llm_model", "gpt-oss")  # Allow config to specify LLM model

    print(f"Using LLM: {model_name}")
    print(f"Limit per experiment: {limit}")

    # Initialize shared resources
    # Vectorizer is needed for FaissRetriever constructor
    vectorizer = Vectorizer.from_model_name(config["embedding_model"])
    llm = Ollama(model=model_name)

    for experiment in tqdm(config["experiments"], desc="Processing experiments"):
        generate_silver_for_experiment(experiment, config, vectorizer, llm, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build silver standard datasets for experiments.")
    parser.add_argument("config_path", help="Path to the experiment configuration file.")
    args = parser.parse_args()
    main(args.config_path)
