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


def create_output_directory(prefix: str = "") -> str:
    folder_name = prefix if prefix else "experiment_results"
    output_dir: str = os.path.join("results", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this run will be saved in '{output_dir}'.")
    return output_dir


def load_config(json_path: str) -> dict[str, Any]:
    with open(json_path) as f:
        return json.load(f)


def generate_experiment_prefix(
    config: dict[str, Any], config_path: str, difficulty: str | None = None
) -> str:
    """
    Construct output directory prefix based on experiment parameters.
    Format: $embedding_$experimentName_$usedInputFile_$difficulty
    """
    embedding_model = config.get("embedding_model", "unknown").replace("/", "_")
    experiment_name = (
        os.path.splitext(os.path.basename(config_path))[0]
        if config_path
        else "unknown_experiment"
    )

    input_file = config.get("input_file")
    if input_file and isinstance(input_file, str):
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]
    else:
        input_file_name = "unknown_input"

    parts = [embedding_model, experiment_name, input_file_name]
    if difficulty:
        parts.append(difficulty)

    return "_".join(parts)


def run_experiments(
    config: dict[str, Any],
    output_dir: str,
    timestamp: str,
    difficulty: str | None = None,
) -> int:
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

    return len(runner.dataset)


def main(config_json: str = None, difficulty: str | None = None) -> None:
    start_time = datetime.now()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    config = load_config(config_json)

    prefix = generate_experiment_prefix(config, config_json, difficulty)

    if config.get("input_file") and isinstance(config["input_file"], str):
        input_file = config["input_file"]
        if not os.path.exists(input_file):
            print(f"Warning: Dataset file not found: {input_file}")

    # Check if results directory already exists to implement caching
    potential_output_dir = os.path.join("results", prefix)
    if os.path.exists(potential_output_dir):
        print(f"Skipping experiment: Results directory '{potential_output_dir}' already exists.")
        return

    output_dir = create_output_directory(prefix)

    config_filename = os.path.basename(config_json)
    dest_path = os.path.join(output_dir, f"experiment_config_{config_filename}")
    shutil.copy(config_json, dest_path)

    dataset_size = run_experiments(config, output_dir, timestamp_str, difficulty=difficulty)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    metadata = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "duration_human": str(end_time - start_time),
        "dataset_size": dataset_size,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


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
        help="Filter dataset by difficulty (e.g., 'Hard', 'Moderate', 'Easy')",
    )
    args = parser.parse_args()
    main(args.config_json, difficulty=args.difficulty)


if __name__ == "__main__":
    print(f"--- Analysis started at: {datetime.now()} ---")
    cli_entry()
    print(f"--- Analysis finished at: {datetime.now()} ---")
