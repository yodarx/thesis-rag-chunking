import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from src.experiment.data_loader import load_asqa_dataset
from src.experiment.results import ResultsHandler
from src.experiment.retriever import FaissRetriever
from src.experiment.runner import ExperimentRunner
from src.plotting.plotting import visualize_and_save_results
from src.vectorizer.vectorizer import Vectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def _create_output_directory() -> (str, str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = "/workspace" if os.path.exists("/workspace") else "results"
    output_dir = os.path.join(base_dir, "results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this run will be saved in '{output_dir}'.")
    return output_dir, timestamp


def load_config_and_experiments(json_path):
    with open(json_path) as f:
        config = json.load(f)
    # We only need the experiment configurations, not the functions themselves
    experiments = config["experiments"]
    return experiments, config


def main(config_json: str = None):
    output_dir, timestamp = _create_output_directory()
    main_with_shared_output(config_json, output_dir, timestamp)


def cli_entry():
    parser = argparse.ArgumentParser(description="Run RAG chunking experiments from pre-built indices.")
    parser.add_argument(
        "--config-json",
        type=str,
        required=False,
        help="Path to the experiment config JSON file (e.g., configs/base_experiment.json)",
    )
    parser.add_argument(
        "--run-all-configs",
        action="store_true",
        help="Run all experiment configs in the configs/ folder sequentially.",
    )
    args = parser.parse_args()

    if args.run_all_configs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = "/workspace" if os.path.exists("/workspace") else "results"
        main_output_dir = os.path.join(base_dir, "results", timestamp)
        os.makedirs(main_output_dir, exist_ok=True)

        config_dir = os.path.join(os.path.dirname(__file__), "configs")
        config_files = sorted(
            [os.path.join(config_dir, f) for f in os.listdir(config_dir) if f.endswith(".json")]
        )

        print(f"Found {len(config_files)} config files: {[os.path.basename(f) for f in config_files]}")
        print(f"Main results directory: {main_output_dir}")

        if not config_files:
            print("No config files found in configs/ directory.")
            sys.exit(1)

        for config_path in config_files:
            config_name = os.path.splitext(os.path.basename(config_path))[0]
            experiment_output_dir = os.path.join(main_output_dir, config_name)
            os.makedirs(experiment_output_dir, exist_ok=True)

            print(f"\nRunning experiment for config: {config_name}")
            print(f"Experiment output directory: {experiment_output_dir}")

            try:
                main_with_shared_output(
                    config_json=config_path,
                    output_dir=experiment_output_dir,
                    timestamp=timestamp
                )
                print(f"Experiment completed for {config_name}")
            except Exception as e:
                print(f"Experiment failed for {config_name}: {e}", file=sys.stderr)

        print("\nAll experiments completed successfully!")
        sys.exit(0)
    elif args.config_json:
        main(config_json=args.config_json)
        print("\nExperiment completed successfully!")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(0)


def main_with_shared_output(config_json: str, output_dir: str, timestamp: str):
    """Run experiment with pre-created output directory using pre-built indices."""
    if not config_json:
        print("Error: --config-json is required. Please provide a config file.")
        sys.exit(1)

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

    # This vectorizer is now only for embedding the questions
    print(f"Initializing query vectorizer with {embedding_model_name}...")
    vectorizer = Vectorizer.from_model_name(model_name=embedding_model_name)

    if retriever_type == "faiss":
        print(f"Using FAISS Retriever with top_k={top_k}.")
        retriever = FaissRetriever(vectorizer)
    else:
        print(f"Error: Unknown retriever type '{retriever_type}'. Only 'faiss' is supported.")
        sys.exit(1)

    results_handler = ResultsHandler(output_dir, timestamp)

    runner = ExperimentRunner(
        experiments=experiments,
        dataset=dataset,
        vectorizer=vectorizer,
        retriever=retriever,
        results_handler=results_handler,
        top_k=top_k,
        embedding_model_name=embedding_model_name,
    )

    summary_df = runner.run_all()

    if not summary_df.empty:
        visualize_and_save_results(summary_df, output_dir, timestamp)
    else:
        print("No summary DataFrame to visualize.")


if __name__ == "__main__":
    cli_entry()
