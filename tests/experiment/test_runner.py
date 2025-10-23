import numpy as np
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
    return {
        "vectorizer": mocker.Mock(spec=Vectorizer),
        "retriever": mocker.Mock(spec=FaissRetriever),
        "results_handler": mocker.Mock(spec=ResultsHandler),
    }


@pytest.fixture
def sample_experiment() -> dict:
    """Provides a single sample experiment definition."""
    # A dummy chunk function that splits the text
    return {
        "name": "test_exp",
        "function": lambda text, size: [text[:size], text[size:]],
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
    mock_embeddings_array = np.array(mock_embeddings_list, dtype="float32")
    # Note: embed_documents will be called twice (chunks, question)
    # We only explicitly set the return value needed for the chunk embedding check.
    mock_vectorizer.embed_documents.return_value = mock_embeddings_list

    # Mock retriever to return index 0
    mock_indices = [0]
    mock_retriever.search.return_value = mock_indices

    # 2. Instantiate Runner
    runner = ExperimentRunner(
        experiments=[sample_experiment], dataset=sample_dataset, top_k=5, **mock_deps
    )

    # 3. Call Method Under Test
    data_point = sample_dataset[0]
    runner._process_single_experiment(data_point, sample_experiment)

    # 4. Assertions

    # Was vectorizer called for the chunks?
    expected_chunks = ["This is th", "e full document text."]
    # Use assert_any_call as it's called again for the question inside retriever
    mock_vectorizer.embed_documents.assert_any_call(expected_chunks)

    # Was retriever called correctly?
    mock_retriever.search.assert_called_once()
    assert mock_retriever.search.call_args[0][0] == "What is this?"  # question
    np.testing.assert_array_equal(
        mock_retriever.search.call_args[0][1],
        mock_embeddings_array,  # chunk_embeddings
    )
    assert mock_retriever.search.call_args[0][2] == 5  # top_k

    # Was the result recorded correctly?
    mock_results.add_result_record.assert_called_once()
    call_args, _ = mock_results.add_result_record.call_args  # Unpack args only

    assert call_args[0] == data_point  # data_point
    assert call_args[1] == "test_exp"  # experiment_name
    assert isinstance(call_args[2], float)  # chunking_time
    assert call_args[3] == 2  # num_chunks
    # Gold "full document" is in chunk 1, retriever found chunk 0 -> MRR 0.0
    assert call_args[4]["mrr"] == 0.0  # metrics


def test_run_all_loop(mock_deps, sample_experiment, sample_dataset, mocker):
    """Tests if run_all iterates correctly and handles results."""

    # Mock the internal processing method to isolate the loop logic
    mock_process_exp = mocker.patch(
        "src.experiment.runner.ExperimentRunner._process_single_experiment"
    )

    mock_results = mock_deps["results_handler"]

    mock_detailed_df = pd.DataFrame({"sample_id": ["s1"]})  # Content doesn't matter
    mock_results.save_detailed_results.return_value = mock_detailed_df

    mock_summary_df = pd.DataFrame({"experiment": ["test_exp"], "mrr": [1.0]})
    mock_results.create_and_save_summary.return_value = mock_summary_df

    # Setup dataset and experiments for the test run
    dataset = [sample_dataset[0], sample_dataset[0]]  # 2 data points
    experiments = [sample_experiment, sample_experiment]  # 2 experiments

    # Instantiate runner
    runner = ExperimentRunner(experiments=experiments, dataset=dataset, top_k=5, **mock_deps)

    # Execute the main loop method
    summary = runner.run_all()

    assert mock_process_exp.call_count == 4  # 2 data points * 2 experiments
    mock_results.save_detailed_results.assert_called_once()
    mock_results.create_and_save_summary.assert_called_once_with(mock_detailed_df)
    mock_results.display_summary.assert_called_once_with(mock_summary_df)
    assert summary.equals(mock_summary_df)  # Check returned summary
