import argparse
import os
from typing import Any

try:
    from google import genai
except ImportError:
    genai = None

from src.experiment.silver_generation import generate_silver_for_experiment


def initialize_llm_client() -> Any:
    """
    Initialize LLM client.

    Args:
    Returns:
        Client
    """
    if genai is None:
        raise ImportError("google-genai not installed. Install with: pip install google-genai")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )
    return genai.Client(vertexai=True, api_key=api_key)


def main(limit: int = 10) -> None:
    print(f"Limit: {limit}")

    client = initialize_llm_client()

    generate_silver_for_experiment(client, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build silver standard datasets for experiments.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many silver samples to generate per experiment (default: 10)",
    )
    parser.add_argument(
        "--llm",
        default="gemini",
        choices=["gemini", "ollama"],
        help="LLM backend to use (default: gemini)",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama2",
        help="Ollama model name to use (default: llama2). Only used if --llm is 'ollama'",
    )
    args = parser.parse_args()
    main(limit=args.limit)
