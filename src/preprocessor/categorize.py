import argparse
import json
import os
import sys
from typing import Any

# Add the project root to the python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from langchain_community.llms import Ollama

from src.experiment.data_loader import DatasetCategorizer


def load_jsonl(filepath: str) -> list[dict[str, Any]]:
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: list[dict[str, Any]], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(input_file: str, output_file: str, model_name: str, limit: int | None = None):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} entries.")

    if limit:
        print(f"Limiting to first {limit} entries.")
        data = data[:limit]

    print(f"Initializing LLM ({model_name})...")
    llm = Ollama(model=model_name)
    categorizer = DatasetCategorizer(llm=llm)

    print("Categorizing dataset...")
    categorized_data = categorizer.categorize_dataset(data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Saving categorized data to {output_file}...")
    save_jsonl(categorized_data, output_file)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize a preprocessed dataset using an LLM.")
    parser.add_argument("input_file", help="Path to the input JSONL file.")
    parser.add_argument("output_file", help="Path to the output JSONL file.")
    parser.add_argument(
        "--model", default="gpt-oss", help="Ollama model name to use (default: gpt-oss)."
    )
    parser.add_argument("--limit", type=int, help="Limit the number of entries to categorize.")

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.model, args.limit)
