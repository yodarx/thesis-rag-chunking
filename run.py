import argparse
import os
import sys
from datetime import datetime

from src.experiment.data_loader import load_asqa_dataset
from src.experiment.experiment import (
    TOP_K_FOR_METRICS,
    get_experiments,
)
from src.experiment.results import ResultsHandler
from src.experiment.retriever import FaissRetriever
from src.experiment.runner import ExperimentRunner
from src.plotting.plotting import visualize_and_save_results
from src.vectorizer.vectorizer import Vectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def _create_output_directory() -> (str, str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this run will be saved in '{output_dir}'.")
    return output_dir, timestamp


def main(input_filepath: str, limit: int | None):
    output_dir, timestamp = _create_output_directory()
    experiments = get_experiments()

    dataset = load_asqa_dataset(input_filepath, limit=None)
    if not dataset:
        print("Dataset could not be loaded. Exiting.")
        return

    print("Initializing Vectorizer...")
    vectorizer = Vectorizer.from_model_name()
    retriever = FaissRetriever(vectorizer)
    # Cacher-Initialisierung entfernt
    results_handler = ResultsHandler(output_dir, timestamp)

    runner = ExperimentRunner(
        experiments=experiments,
        dataset=dataset,
        vectorizer=vectorizer,
        retriever=retriever,
        results_handler=results_handler,
        top_k=TOP_K_FOR_METRICS,
    )

    summary_df = runner.run_all()

    if not summary_df.empty:
        visualize_and_save_results(summary_df, output_dir, timestamp)
    else:
        print("No summary DataFrame to visualize.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG chunking experiments.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,  # Make input file mandatory
        help="Path to the preprocessed input JSONL file (e.g., data/processed/preprocessed_....jsonl).",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit the number of dataset entries to process (default: process all).",
    )
    args = parser.parse_args()

    main(input_filepath=args.input, limit=args.limit)
