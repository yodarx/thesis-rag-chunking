import datetime
import json
import logging
import os
from typing import Any

from src.experiment.silver_standard import SilverStandardGenerator

# Setup logger
logger = logging.getLogger(__name__)


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
        logger.warning(f"Input file does not exist: {input_file}")
        return documents

    logger.info(f"Loading documents from {input_file}")
    try:
        with open(input_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
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
                    except json.JSONDecodeError as e:
                        logger.debug(f"Skipping line {line_num}: Invalid JSON - {e}")
                        continue
    except Exception as e:
        logger.error(f"Error loading documents from {input_file}: {e}")

    logger.info(f"Successfully loaded {len(documents)} documents from {input_file}")
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
    logger.info(f"Starting silver generation with limit={limit}")
    start_time = datetime.datetime.now()

    input_file = "data/preprocessed/preprocessed_2025-11-03_all_categorized.jsonl"
    documents = load_documents_from_input_file(input_file)

    if not documents:
        logger.warning("No documents loaded. Generation will produce empty dataset.")
    else:
        logger.info(f"Initialized generator with {len(documents)} documents")

    generator = SilverStandardGenerator(llm_client, documents)
    logger.debug(f"Generating dataset with {limit} samples, num_hops=2")
    dataset = generator.generate_dataset(limit, num_hops=2)

    generation_time = datetime.datetime.now() - start_time
    logger.info(f"Generation completed in {generation_time.total_seconds():.2f}s, generated {len(dataset)} samples")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = "data/silver"
    output_file = os.path.join(output_dir, f"{timestamp}_silver.jsonl")

    logger.info(f"Writing results to {output_file}")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(dataset, 1):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Successfully saved {len(dataset)} samples to {output_file}")
    print(f"✓ Saved {len(dataset)} samples to {output_file}")

    return output_file
