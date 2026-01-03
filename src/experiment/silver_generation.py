import datetime
import json
import os
from typing import Any

from src.experiment.silver_standard import SilverStandardGenerator


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
        llm_client: Any,
        limit: int,
) -> str:
    """
    Generates a silver standard dataset for a specific experiment.

    Args:
        llm_client: LLM client (genai.Client for Gemini)
        limit: Number of samples to generate

    Returns:
        Path to the generated file
    """
    input_file = "data/preprocessed/preprocessed_2025-11-03_all_categorized.jsonl"
    documents = load_documents_from_input_file(input_file)

    generator = SilverStandardGenerator(llm_client, documents)
    dataset = generator.generate_dataset(limit, num_hops=2)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = "data/silver"
    output_file = os.path.join(output_dir, f"{timestamp}_silver.jsonl")

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} samples to {output_file}")

    return output_file
