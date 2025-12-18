import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Any

from src.experiment.data_loader import load_asqa_dataset
from src.experiment.results import ResultsHandler
from src.experiment.retriever import FaissRetriever
from src.experiment.runner import ExperimentRunner
from src.experiment.utils import create_index_name
from src.plotting.plotting import visualize_and_save_results
from src.vectorizer.vectorizer import Vectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def create_output_directory(suffix: str = "") -> tuple[str, str]:
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{suffix}" if suffix else timestamp
    base_dir: str = "/workspace" if os.path.exists("/workspace") else "results"
    output_dir: str = os.path.join(base_dir, "results", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this run will be saved in '{output_dir}'.")
    return output_dir, timestamp


def load_config(json_path: str) -> dict[str, Any]:
    with open(json_path) as f:
        return json.load(f)


def run_experiments(
    config: dict[str, Any], output_dir: str, timestamp: str, use_silver: bool = False
) -> None:
    retriever_type = config.get("retriever_type", "faiss")
    vectorizer: Vectorizer = Vectorizer.from_model_name(config["embedding_model"])

    if retriever_type == "faiss":
        retriever: FaissRetriever = FaissRetriever(vectorizer)
    else:
        print("Error: Unknown retriever type")
        sys.exit(1)

    results_handler: ResultsHandler = ResultsHandler(output_dir, timestamp)

    if use_silver:
        print("Running in Silver Standard mode (per-experiment datasets)...")

        for experiment in config["experiments"]:
            # Check if a specific silver file is defined for this experiment
            silver_path = experiment.get("input_silver_file")

            if silver_path:
                print(f"Using provided silver dataset for {experiment['name']}: {silver_path}")
                if not os.path.exists(silver_path):
                    print(f"Error: Provided silver file not found: {silver_path}")
                    continue
            else:
                # 1. Check if silver dataset exists (do NOT auto-generate)
                index_name = create_index_name(experiment["name"], config["embedding_model"])
                silver_path = os.path.join("data", "silver", f"{index_name}_silver.jsonl")

                if not os.path.exists(silver_path):
                    logging.warning(
                        f"Silver dataset not found for {experiment['name']} at {silver_path}. Skipping."
                    )
                    silver_path = None

            if not silver_path:
                print(f"Skipping experiment {experiment['name']} due to missing silver dataset.")
                continue

            # 2. Load specific dataset
            dataset = load_asqa_dataset(silver_path, config.get("limit"))

            # 3. Run experiment
            runner = ExperimentRunner(
                [experiment],  # Run only this experiment
                dataset,
                vectorizer,
                retriever,
                results_handler,
                config["top_k"],
                config["embedding_model"],
            )
            runner.run_all()

    else:
        dataset = load_asqa_dataset(config["input_file"], config.get("limit"))
        runner = ExperimentRunner(
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


def main(config_json: str = None, use_silver: bool = False) -> None:
    # Load config first to determine run type
    config = load_config(config_json)
    input_file = config.get("input_file", "")

    suffix = ""
    if use_silver or "silver" in input_file.lower():
        suffix = "silver"
    elif "gold" in input_file.lower():
        suffix = "gold"

    output_dir, timestamp = create_output_directory(suffix)

    # Copy config file to results directory
    config_filename = os.path.basename(config_json)
    dest_path = os.path.join(output_dir, f"experiment_config_{config_filename}")
    shutil.copy(config_json, dest_path)

    # Pass loaded config to avoid reloading
    run_experiments(config, output_dir, timestamp, use_silver=use_silver)


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
        "--silver",
        action="store_true",
        help="Run in Silver Standard mode (auto-generate/use per-experiment silver datasets)",
    )
    args = parser.parse_args()
    main(args.config_json, use_silver=args.silver)


if __name__ == "__main__":
    print(f"--- Analysis started at: {datetime.now()} ---")
    cli_entry()
    print(f"--- Analysis finished at: {datetime.now()} ---")
