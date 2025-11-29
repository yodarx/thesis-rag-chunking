import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any

from src.experiment.data_loader import load_asqa_dataset
from src.experiment.results import ResultsHandler
from src.experiment.retriever import FaissRetriever
from src.experiment.runner import ExperimentRunner
from src.plotting.plotting import visualize_and_save_results
from src.vectorizer.vectorizer import Vectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def create_output_directory() -> tuple[str, str]:
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir: str = "/workspace" if os.path.exists("/workspace") else "results"
    output_dir: str = os.path.join(base_dir, "results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this run will be saved in '{output_dir}'.")
    return output_dir, timestamp


def load_config(json_path: str) -> dict[str, Any]:
    with open(json_path) as f:
        return json.load(f)


def run_experiments(config_json: str, output_dir: str, timestamp: str) -> None:
    config: dict[str, Any] = load_config(config_json)
    dataset = load_asqa_dataset(config["input_file"], config.get("limit"))
    retriever_type = config.get("retriever_type", "faiss")
    vectorizer: Vectorizer = Vectorizer.from_model_name(config["embedding_model"])
    if retriever_type == "faiss":
        retriever: FaissRetriever = FaissRetriever(vectorizer)
    else:
        print("Error: Unknown retriever type")
        sys.exit(1)
    results_handler: ResultsHandler = ResultsHandler(output_dir, timestamp)
    runner: ExperimentRunner = ExperimentRunner(
        config["experiments"],
        dataset,
        vectorizer,
        retriever,
        results_handler,
        config["top_k"],
        config["embedding_model"],
    )
    summary_df = runner.run_all()
    if not summary_df.empty:
        visualize_and_save_results(summary_df, output_dir, timestamp)
    else:
        print("No summary DataFrame to visualize.")


def main(config_json: str = None) -> None:
    output_dir, timestamp = create_output_directory()
    # Copy config file to results directory
    config_filename = os.path.basename(config_json)
    dest_path = os.path.join(output_dir, f"experiment_config_{config_filename}")
    shutil.copy(config_json, dest_path)
    run_experiments(config_json, output_dir, timestamp)


def cli_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG chunking experiments from pre-built indices."
    )
    parser.add_argument(
        "--config-json",
        type=str,
        required=True,
        help="Path to the experiment config JSON file (e.g., configs/base_experiment.json)",
    )
    args = parser.parse_args()
    main(args.config_json)


if __name__ == "__main__":
    print(f"--- Analysis started at: {datetime.now()} ---")
    cli_entry()
    print(f"--- Analysis finished at: {datetime.now()} ---")
