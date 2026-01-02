import json
import os
from typing import Any

from src.experiment.retriever import FaissRetriever
from src.experiment.silver_standard import SilverStandardGenerator
from src.experiment.utils import create_index_name
from src.vectorizer.vectorizer import Vectorizer


def load_documents_from_input_file(input_file: str) -> list[str]:
    """
    Load document texts from the input file (JSONL format).

    Args:
        input_file: Path to the input JSONL file

    Returns:
        List of document texts
    """
    documents = []
    if not os.path.exists(input_file):
        return documents

    try:
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        # Extract text field - adapt based on your data structure
                        if "document_text" in entry:
                            documents.append(entry["document_text"])
                        elif "text" in entry:
                            documents.append(entry["text"])
                        elif "content" in entry:
                            documents.append(entry["content"])
                        elif "document" in entry:
                            documents.append(entry["document"])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Error loading documents from {input_file}: {e}")

    return documents


def generate_silver_for_experiment(
    experiment: dict[str, Any],
    config: dict[str, Any],
    vectorizer: Vectorizer,
    llm_client: Any,
    llm_type: str,
    limit: int,
) -> str:
    """
    Generates a silver standard dataset for a specific experiment.

    Args:
        experiment: Experiment configuration
        config: Model configuration
        vectorizer: Vectorizer instance
        llm_client: LLM client (genai.Client for Gemini or Ollama for local)
        llm_type: "gemini" or "ollama"
        limit: Number of samples to generate

    Returns:
        Path to the generated file
    """
    index_name = create_index_name(experiment["name"], config["embedding_model"])
    output_dir = os.path.join("data", "silver")
    output_file = os.path.join(output_dir, f"{index_name}_silver.jsonl")

    if os.path.exists(output_file):
        print(f"Skipping {index_name}: Silver standard already exists at {output_file}")
        return output_file

    print(f"Generating silver standard for {index_name}...")

    # Load documents from the input file
    input_file = config.get(
        "input_file", "data/preprocessed/preprocessed_2025-11-03_all_categorized.jsonl"
    )
    documents = load_documents_from_input_file(input_file)

    if not documents:
        print(f"Skipping {index_name}: No documents found in {input_file}")
        return ""

    # Initialize retriever with documents (no index needed for random sampling)
    retriever = FaissRetriever(vectorizer)
    retriever.chunks = documents

    # Adjust limit if it's -1 (all) or larger than available documents
    current_limit = limit
    if current_limit == -1 or current_limit > len(retriever.chunks):
        current_limit = len(retriever.chunks)

    hops_count = config.get("hops_count", 3)
    generator = SilverStandardGenerator(retriever, llm_client, llm_type=llm_type)
    dataset = generator.generate_dataset(current_limit, num_hops=hops_count)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} samples to {output_file}")

    return output_file
