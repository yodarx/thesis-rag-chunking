import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from src.chunking.chunk_fixed import chunk_fixed_size
from src.chunking.chunk_recursive import chunk_recursive
from src.chunking.chunk_semantic import chunk_semantic
from src.chunking.chunk_sentence import chunk_by_sentence
from src.experiment.data_loader import load_asqa_dataset
from src.experiment.results import ResultsHandler
from src.experiment.retriever import FaissRetriever
from src.experiment.runner import ExperimentRunner
from src.plotting.plotting import visualize_and_save_results
from src.vectorizer.vectorizer import Vectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def _create_output_directory() -> (str, str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use /workspace for persistent storage on RunPod.io
    base_dir = "/workspace" if os.path.exists("/workspace") else "results"
    output_dir = os.path.join(base_dir, "results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this run will be saved in '{output_dir}'.")
    return output_dir, timestamp


def load_config_and_experiments(json_path):
    with open(json_path) as f:
        config = json.load(f)
    # Map function names to actual functions
    chunk_functions = {
        "chunk_fixed_size": chunk_fixed_size,
        "chunk_by_sentence": chunk_by_sentence,
        "chunk_recursive": chunk_recursive,
        "chunk_semantic": chunk_semantic,
    }
    experiments = []
    for exp in config["experiments"]:
        func = chunk_functions[exp["function"]]
        experiments.append(
            {
                "name": exp["name"],
                "function": func,
                "params": exp["params"],
            }
        )
    return experiments, config


def main(
    input_filepath: str = None,
    limit: int | None = None,
    embedding_model_name: str = None,
    retriever_type: str = None,
    top_k: int = None,
    config_json: str = None,
):
    output_dir, timestamp = _create_output_directory()
    if not config_json:
        print("Error: --config-json is required. Please provide a config file.")
        sys.exit(1)
    # Copy config file to results directory
    config_copy_path = os.path.join(output_dir, "experiment_config.json")
    shutil.copy(config_json, config_copy_path)
    experiments, config = load_config_and_experiments(config_json)
    input_filepath = config["input_file"]
    limit = config.get("limit")
    embedding_model_name = config["embedding_model"]
    retriever_type = config["retriever_type"]
    top_k = config["top_k"]

    dataset = load_asqa_dataset(input_filepath, limit=limit)
    if not dataset:
        print("Dataset could not be loaded. Exiting.")
        return

    print(f"Initializing Retrieval Vectorizer with {embedding_model_name}...")
    vectorizer = Vectorizer.from_model_name(model_name=embedding_model_name)

    if retriever_type == "faiss":
        print(f"Using FAISS Retriever with top_k={top_k}.")
        retriever = FaissRetriever(vectorizer)
    else:
        print(f"Error: Unknown retriever type '{retriever_type}'. Only 'faiss' is supported.")
        sys.exit(1)

    results_handler = ResultsHandler(output_dir, timestamp)

    # ExperimentRunner will load chunking models as needed from experiment params
    runner = ExperimentRunner(
        experiments=experiments,
        dataset=dataset,
        vectorizer=vectorizer,
        retriever=retriever,
        results_handler=results_handler,
        top_k=top_k,
    )

    summary_df = runner.run_all()

    if not summary_df.empty:
        visualize_and_save_results(summary_df, output_dir, timestamp)
    else:
        print("No summary DataFrame to visualize.")


def cli_entry():
    parser = argparse.ArgumentParser(description="Run RAG chunking experiments.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=False,
        help="Path to the preprocessed input JSONL file (e.g., data/processed/preprocessed_....jsonl).",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit the number of dataset entries to process (default: process all).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the Sentence Transformer model to use for embeddings.",
    )
    parser.add_argument(
        "--retriever-type",
        type=str,
        default="faiss",
        choices=["faiss"],
        help="Type of retriever to use (currently only 'faiss').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve (retriever's K).",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        required=False,
        help="Path to the experiment config JSON file (default: configs/base_experiment.json)",
    )
    parser.add_argument(
        "--run-all-configs",
        action="store_true",
        help="Run all experiment configs in the configs/ folder sequentially.",
    )
    args = parser.parse_args()

    if args.run_all_configs:
        # Create ONE shared output directory for all configs
        output_dir, timestamp = _create_output_directory()
        config_dir = os.path.join(os.path.dirname(__file__), "configs")
        config_files = sorted(
            [os.path.join(config_dir, f) for f in os.listdir(config_dir) if f.endswith(".json")]
        )

        print(
            f"Found {len(config_files)} config files: {[os.path.basename(f) for f in config_files]}"
        )

        if not config_files:
            print("No config files found in configs/ directory.")
            sys.exit(1)

        for config_path in config_files:
            print(f"\nRunning experiment for config: {os.path.basename(config_path)}")
            try:
                main_with_shared_output(
                    config_json=config_path, output_dir=output_dir, timestamp=timestamp
                )
                print(f"Experiment completed for {os.path.basename(config_path)}")
            except Exception as e:
                print(
                    f"Experiment failed for {os.path.basename(config_path)}: {e}", file=sys.stderr
                )

        print("All experiments completed successfully!")
        sys.exit(0)
    elif args.config_json:
        main(config_json=args.config_json)
        print("Experiment completed successfully!")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(0)


def main_with_shared_output(config_json: str, output_dir: str, timestamp: str):
    """Run experiment with pre-created output directory"""
    if not config_json:
        print("Error: --config-json is required. Please provide a config file.")
        sys.exit(1)

    # Copy config file to results directory
    config_copy_path = os.path.join(
        output_dir, f"experiment_config_{os.path.basename(config_json)}"
    )
    shutil.copy(config_json, config_copy_path)

    experiments, config = load_config_and_experiments(config_json)
    input_filepath = config["input_file"]
    limit = config.get("limit")
    embedding_model_name = config["embedding_model"]
    retriever_type = config["retriever_type"]
    top_k = config["top_k"]

    dataset = load_asqa_dataset(input_filepath, limit=limit)
    if not dataset:
        print("Dataset could not be loaded. Exiting.")
        return

    print(f"Initializing Retrieval Vectorizer with {embedding_model_name}...")
    vectorizer = Vectorizer.from_model_name(model_name=embedding_model_name)

    if retriever_type == "faiss":
        print(f"Using FAISS Retriever with top_k={top_k}.")
        retriever = FaissRetriever(vectorizer)
    else:
        print(f"Error: Unknown retriever type '{retriever_type}'. Only 'faiss' is supported.")
        sys.exit(1)

    results_handler = ResultsHandler(output_dir, timestamp)

    # ExperimentRunner will load chunking models as needed from experiment params
    runner = ExperimentRunner(
        experiments=experiments,
        dataset=dataset,
        vectorizer=vectorizer,
        retriever=retriever,
        results_handler=results_handler,
        top_k=top_k,
    )

    summary_df = runner.run_all()

    if not summary_df.empty:
        visualize_and_save_results(summary_df, output_dir, timestamp)
    else:
        print("No summary DataFrame to visualize.")


if __name__ == "__main__":
    cli_entry()
