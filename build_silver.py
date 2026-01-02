import argparse
import json
import os
from typing import Any

from langchain_community.llms import Ollama

try:
    from google import genai
except ImportError:
    genai = None

from src.experiment.silver_generation import generate_silver_for_experiment
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


def initialize_llm_client(llm_type: str, model_name: str | None = None) -> tuple[Any, str]:
    """
    Initialize LLM client based on type.

    Args:
        llm_type: "gemini" or "ollama"
        model_name: Model name (for Ollama, defaults to "llama2")

    Returns:
        Tuple of (client, llm_type)
    """
    llm_type = llm_type.lower()

    if llm_type == "gemini":
        if genai is None:
            raise ImportError("google-genai not installed. Install with: pip install google-genai")

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        client = genai.Client(vertexai=True, api_key=api_key)
        print("Using LLM: Gemini 2.0 Flash")

    elif llm_type == "ollama":
        model_name = model_name or "llama2"
        client = Ollama(model=model_name)
        print(f"Using LLM: Ollama ({model_name})")

    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}. Use 'gemini' or 'ollama'.")

    return client, llm_type


def main(config_path: str, llm_type: str = "gemini", ollama_model: str | None = None) -> None:
    config = load_config(config_path)

    # Default limit or  from config
    limit = config.get("silver_limit", 10)

    print(f"Limit per experiment: {limit}")

    # Initialize LLM client
    client, llm_type_resolved = initialize_llm_client(llm_type, ollama_model)

    # Initialize shared resources
    # Vectorizer is needed for FaissRetriever constructor
    vectorizer = Vectorizer.from_model_name(config["embedding_model"])

    experiments = config.get("experiments", [])
    for i, experiment in enumerate(experiments, 1):
        print(f"Processing experiment {i}/{len(experiments)}: {experiment.get('name', 'unnamed')}")
        generate_silver_for_experiment(
            experiment, config, vectorizer, client, llm_type_resolved, limit
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build silver standard datasets for experiments.")
    parser.add_argument("config_path", help="Path to the experiment configuration file.")
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
    main(args.config_path, llm_type=args.llm, ollama_model=args.ollama_model)
