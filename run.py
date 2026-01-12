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


def create_output_directory(suffix: str = "") -> tuple[str, str]:
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{suffix}" if suffix else timestamp
    output_dir: str = os.path.join("results", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this run will be saved in '{output_dir}'.")
    return output_dir, timestamp


def load_config(json_path: str) -> dict[str, Any]:
    with open(json_path) as f:
        return json.load(f)


def run_experiments(
        config: dict[str, Any],
        output_dir: str,
        timestamp: str,
        difficulty: str | None = None,
) -> None:
    retriever_type = config.get("retriever_type", "faiss")
    vectorizer: Vectorizer = Vectorizer.from_model_name(config["embedding_model"])

    if retriever_type == "faiss":
        retriever: FaissRetriever = FaissRetriever(vectorizer)
    else:
        print("Error: Unknown retriever type")
        sys.exit(1)

    results_handler: ResultsHandler = ResultsHandler(output_dir, timestamp)

    dataset = load_asqa_dataset(config["input_file"], config.get("limit"))
    runner = ExperimentRunner(
        config["experiments"],
        dataset,
        vectorizer,
        retriever,
        results_handler,
        config["top_k"],
        config["embedding_model"],
        difficulty=difficulty,
    )
    summary_df = runner.run_all()
    if not summary_df.empty:
        visualize_and_save_results(summary_df, output_dir, timestamp)
    else:
        print("No summary DataFrame to visualize.")


def main(config_json: str = None, difficulty: str | None = None) -> None:
    config = load_config(config_json)

    suffix = ""

    if difficulty:
        suffix = f"{suffix}_{difficulty}" if suffix else difficulty

    output_dir, timestamp = create_output_directory(suffix)

    config_filename = os.path.basename(config_json)
    dest_path = os.path.join(output_dir, f"experiment_config_{config_filename}")
    shutil.copy(config_json, dest_path)

    run_experiments(config, output_dir, timestamp, difficulty=difficulty)


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
    parser.add_argument(
        "--difficulty",
        type=str,
        help="Filter dataset by difficulty (e.g., 'Hard', 'Medium', 'Easy')",
    )
    args = parser.parse_args()
    main(args.config_json, difficulty=args.difficulty)


if __name__ == "__main__":
    print(f"--- Analysis started at: {datetime.now()} ---")
    cli_entry()
    print(f"--- Analysis finished at: {datetime.now()} ---")
