import argparse
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from langchain_community.llms import Ollama

from src.experiment.retriever import FaissRetriever
from src.experiment.silver_standard import SilverStandardGenerator
from src.vectorizer.vectorizer import Vectorizer


def main(index_path: str, chunks_path: str, output_file: str, model_name: str, limit: int):
    if not os.path.exists(chunks_path):
        print(f"Error: Chunks file '{chunks_path}' not found.")
        sys.exit(1)

    # We need a vectorizer to init FaissRetriever.
    # Using a small model as we only need access to the chunks, not necessarily embedding for search here.
    vectorizer = Vectorizer.from_model_name("all-MiniLM-L6-v2")

    retriever = FaissRetriever(vectorizer)

    if os.path.exists(index_path):
        print(f"Loading index from {index_path}...")
        retriever.load_index(index_path, chunks_path)
    else:
        print(
            f"Warning: Index file '{index_path}' not found. Loading only chunks from '{chunks_path}'."
        )
        with open(chunks_path, encoding="utf-8") as f:
            retriever.chunks = json.load(f)

    if limit == -1:
        limit = len(retriever.chunks)
        print(f"Limit set to 'all' ({limit} samples).")

    print(f"Initializing LLM ({model_name})...")
    llm = Ollama(model=model_name)

    generator = SilverStandardGenerator(retriever, llm)

    print(f"Generating {limit} silver standard samples...")
    dataset = generator.generate_dataset(limit)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Saving dataset to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a silver standard dataset using an LLM.")
    parser.add_argument("index_path", help="Path to the FAISS index file.")
    parser.add_argument("chunks_path", help="Path to the chunks JSON file.")
    parser.add_argument("output_file", help="Path to the output JSONL file.")
    parser.add_argument(
        "--model", default="gpt-oss", help="Ollama model name to use (default: gpt-oss)."
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to generate.")

    args = parser.parse_args()
    main(args.index_path, args.chunks_path, args.output_file, args.model, args.limit)
