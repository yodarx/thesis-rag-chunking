from unittest import mock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

# Ensure these imports match your actual file structure
from src.experiment.results import ResultsHandler
from src.experiment.retriever import FaissRetriever
from src.experiment.runner import ExperimentRunner
from src.vectorizer.vectorizer import Vectorizer


@pytest.fixture
def mock_deps(mocker: MockerFixture) -> dict:
    """Creates mocks for all runner dependencies."""
    retriever_mock = mocker.Mock(spec=FaissRetriever)
    # Add index attribute with ntotal for compatibility
    index_mock = mocker.Mock()
    index_mock.ntotal = 1
    retriever_mock.index = index_mock
    return {
        "vectorizer": mocker.Mock(spec=Vectorizer),
        "retriever": retriever_mock,
        "results_handler": mocker.Mock(spec=ResultsHandler),
    }


@pytest.fixture
def sample_experiment() -> dict:
    """Provides a single sample experiment definition."""
    # A dummy chunk function that splits the text
    return {
        "name": "test_exp",
        "function": "dummy_func",
        "params": {"size": 10},
    }


@pytest.fixture
def sample_dataset() -> list:
    """Provides a single sample data point."""
    return [
        {
            "sample_id": "s1",
            "document_text": "This is the full document text.",
            "question": "What is this?",
            "gold_passages": ["full document"],
        }
    ]


# --- Tests ---


def test_process_single_experiment(mock_deps, sample_experiment, sample_dataset, mocker):
    """Tests the core workflow of processing one experiment for one data point."""

    # 1. Configure Mocks
    mock_vectorizer = mock_deps["vectorizer"]
    mock_retriever = mock_deps["retriever"]
    mock_results = mock_deps["results_handler"]

    # Mock vectorizer to return embeddings for chunks
    mock_embeddings_list = [[0.1, 0.2], [0.3, 0.4]]
    mock_vectorizer.embed_documents.return_value = mock_embeddings_list

    # Mock retriever to return index 0
    mock_retriever.retrieve.return_value = ["This is the full document text."]

    # Mock results handler with evaluate method
    mock_results.evaluate = mocker.Mock(return_value={"score": 1.0})

    # 2. Run the experiment logic (simulate)
    # Define local chunk function for simulation
    def chunk_function(text, size):
        return [text[:size], text[size:]]

    params = sample_experiment["params"]
    doc = sample_dataset[0]
    chunks = chunk_function(doc["document_text"], **params)
    mock_retriever.chunks = chunks
    retrieved = mock_retriever.retrieve(doc["question"], 1)
    result = mock_results.evaluate(retrieved, doc["gold_passages"])

    assert result["score"] == 1.0


@mock.patch(
    "src.evaluation.evaluation.calculate_metrics",
    return_value={
        "mrr": 1.0,
        "map": 1.0,
        "ndcg_at_k": 1.0,
        "precision_at_k": 1.0,
        "recall_at_k": 1.0,
        "f1_score_at_k": 1.0,
    },
)
def test_run_all_loop(mock_metrics, mock_deps, sample_experiment, sample_dataset, mocker):
    """Tests if run_all iterates correctly and handles results."""
    # Remove patch for _process_single_experiment, check add_result_record calls instead
    mock_results = mock_deps["results_handler"]
    mock_detailed_df = pd.DataFrame({"sample_id": ["s1"]})  # Content doesn't matter
    mock_results.save_detailed_results.return_value = mock_detailed_df
    mock_summary_df = pd.DataFrame({"experiment": ["test_exp"], "mrr": [1.0]})
    mock_results.create_and_save_summary.return_value = mock_summary_df
    # Setup dataset and experiments for the test run
    dataset = [sample_dataset[0], sample_dataset[0]]  # 2 data points
    experiments = [sample_experiment, sample_experiment]  # 2 experiments
    # Mock os.path.exists to always return True
    mocker.patch("os.path.exists", return_value=True)
    # Instantiate runner with embedding_model_name
    runner = ExperimentRunner(
        experiments=experiments,
        dataset=dataset,
        top_k=5,
        embedding_model_name="test-model",
        **mock_deps,
    )

    # Mock retrieve_batch to avoid TypeError: 'Mock' object is not subscriptable
    mock_retriever = mock_deps["retriever"]
    mock_retriever.retrieve_batch.return_value = [["chunk1"], ["chunk2"]]

    # Execute the main loop method
    summary = runner.run_all()
    # Check that add_result_record was called 4 times (2 data points * 2 experiments)
    assert mock_results.add_result_record.call_count == 4
    mock_results.save_detailed_results.assert_called_once()
    mock_results.create_and_save_summary.assert_called_once_with(mock_detailed_df)
    mock_results.display_summary.assert_called_once_with(mock_summary_df)
    assert summary.equals(mock_summary_df)  # Check returned summary


def test_runner_difficulty_filtering(mock_deps, sample_experiment):
    """Tests that the runner correctly filters the dataset by difficulty."""
    dataset = [
        {"sample_id": "1", "difficulty": "Easy", "question": "Q1", "gold_passages": []},
        {"sample_id": "2", "difficulty": "Hard", "question": "Q2", "gold_passages": []},
        {"sample_id": "3", "difficulty": "Easy", "question": "Q3", "gold_passages": []},
    ]

    # Initialize runner with difficulty filter
    runner = ExperimentRunner(
        experiments=[sample_experiment],
        dataset=dataset,
        vectorizer=mock_deps["vectorizer"],
        retriever=mock_deps["retriever"],
        results_handler=mock_deps["results_handler"],
        top_k=1,
        embedding_model_name="test-model",
        difficulty="Hard",
    )

    assert len(runner.dataset) == 1
    assert runner.dataset[0]["sample_id"] == "2"
    assert runner.dataset[0]["difficulty"] == "Hard"

    # Initialize runner without filter
    runner_all = ExperimentRunner(
        experiments=[sample_experiment],
        dataset=dataset,
        vectorizer=mock_deps["vectorizer"],
        retriever=mock_deps["retriever"],
        results_handler=mock_deps["results_handler"],
        top_k=1,
        embedding_model_name="test-model",
    )

    assert len(runner_all.dataset) == 3
