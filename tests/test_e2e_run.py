import json
import os
from unittest.mock import patch, MagicMock

import faiss
import numpy as np
import pytest

import run


@pytest.fixture(autouse=True)
def mock_tqdm_everywhere():
    """
    Globally mock tqdm to avoid segmentation faults from monitor threads
    and to keep test output clean. Handles both iterable wrapping and
    context manager usage.
    """
    mock_obj = MagicMock()

    def side_effect(*args, **kwargs):
        if args:
            return args[0]  # Return iterable if passed
        return mock_obj  # Return mock for context manager logic

    mock_obj.side_effect = side_effect
    mock_obj.__enter__.return_value = mock_obj
    mock_obj.__exit__.return_value = None

    with patch("src.experiment.runner.tqdm", new=mock_obj), \
         patch("src.experiment.data_loader.tqdm", new=mock_obj):
        yield mock_obj


@pytest.fixture
def e2e_gold_setup(tmp_path):
    # Setup config
    config = {
        "embedding_model": "dummy-model",
        "top_k": 2,
        "input_file": "placeholder",
        "experiments": [
            {
                "name": "test_exp",
                "function": "chunk_fixed_size",
                "params": {"chunk_size": 100, "chunk_overlap": 0},
            }
        ],
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Setup chunks and index
    index_name = "test_exp_dummy-model"
    index_dir = tmp_path / "data" / "indices" / index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content", "Chunk 4 content"]

    chunks_filename = "test_exp_test_exp_chunk_fixed_size_chunks_SORTED.json"
    chunks_dir = tmp_path / "data" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = chunks_dir / chunks_filename

    with open(chunks_path, "w") as f:
        json.dump(chunks, f)

    # Create dummy FAISS index
    dimension = 10
    index = faiss.IndexFlatL2(dimension)
    vectors = np.random.rand(len(chunks), dimension).astype("float32")
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    return config_path, tmp_path


def test_e2e_gold_pipeline(e2e_gold_setup):
    """
    End-to-End test for Gold Standard execution.
    Runs the experiment using a provided 'gold' dataset.
    """
    config_path, tmp_path = e2e_gold_setup

    # Create a dummy Gold Standard dataset
    gold_data = [
        {
            "sample_id": "gold_1",
            "question": "What is in Chunk 1?",
            "gold_passages": ["Chunk 1 content"],
            "category": "Factoid",
            "difficulty": "Easy",
        },
        {
            "sample_id": "gold_2",
            "question": "What is in Chunk 2?",
            "gold_passages": ["Chunk 2 content"],
            "category": "Factoid",
            "difficulty": "Easy",
        },
    ]
    gold_path = tmp_path / "data" / "gold.jsonl"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gold_path, "w") as f:
        for entry in gold_data:
            f.write(json.dumps(entry) + "\n")

    # Update config to use the gold file
    with open(config_path) as f:
        config = json.load(f)
    config["input_file"] = str(gold_path)
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Mock Vectorizer and TQDM
    with (
        patch("run.Vectorizer") as mock_vectorizer_run,
        patch("run.create_output_directory") as mock_create_out,
    ):

        # Setup Mock Vectorizer
        mock_vectorizer_instance = mock_vectorizer_run.from_model_name.return_value

        def side_effect_embed(docs, *args, **kwargs):
            # Determine if we should return numpy or list
            convert_to_numpy = kwargs.get("convert_to_numpy", False)
            embeddings = np.random.rand(len(docs), 10).astype("float32")
            if convert_to_numpy:
                return embeddings
            return embeddings.tolist()

        mock_vectorizer_instance.embed_documents.side_effect = side_effect_embed

        # Setup output directory mock
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        mock_create_out.return_value = str(results_dir)

        # Run the experiment runner
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            run.main(str(config_path))
            mock_create_out.assert_called_once_with("dummy-model_config_gold")

            # Verify results
            files = list(results_dir.glob("*"))
            assert len(files) > 0, "No results files generated"

            # Check validation of metadata.json
            metadata_file = results_dir / "metadata.json"
            assert metadata_file.exists(), "metadata.json not found"

            with open(metadata_file) as f:
                metadata = json.load(f)
                assert "dataset_size" in metadata, "dataset_size missing from metadata"
                assert metadata["embedding_model"] == "dummy-model", "Incorrect embedding model in metadata"
                assert "experiments" in metadata, "Experiments list missing from metadata"

            # Find detailed results csv (timestamp varies)
            detailed_csv_candidates = list(results_dir.glob("*_detailed_results.csv"))
            assert len(detailed_csv_candidates) == 1, "Detailed results CSV not found or multiple found"
            detailed_csv = detailed_csv_candidates[0]

            # Check content of detailed results
            with open(detailed_csv) as f:
                content = f.read()
                assert "What is in Chunk 1?" in content
                assert "gold_1" in content

        finally:
            os.chdir(cwd)


def test_e2e_run_with_difficulty(e2e_gold_setup):
    """
    End-to-End test for execution with difficulty filter.
    Verifies that the output directory includes the difficulty suffix.
    """
    config_path, tmp_path = e2e_gold_setup

    # Create dataset
    gold_data = [
        {
            "sample_id": "gold_1",
            "question": "Easy Q",
            "gold_passages": ["Chunk 1 content"],
            "category": "Factoid",
            "difficulty": "Easy",
        },
        {
            "sample_id": "gold_2",
            "question": "Hard Q",
            "gold_passages": ["Chunk 2 content"],
            "category": "Factoid",
            "difficulty": "Hard",
        },
    ]
    gold_path = tmp_path / "data" / "gold.jsonl"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gold_path, "w") as f:
        for entry in gold_data:
            f.write(json.dumps(entry) + "\n")

    # Update config
    with open(config_path) as f:
        config = json.load(f)
    config["input_file"] = str(gold_path)
    config["difficulty"] = "Hard"
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Mock Vectorizer and create output
    with (
        patch("run.Vectorizer") as mock_vectorizer_run,
        patch("run.create_output_directory") as mock_create_out,
    ):
        mock_vectorizer_instance = mock_vectorizer_run.from_model_name.return_value
        mock_vectorizer_instance.embed_documents.return_value = np.random.rand(1, 10).tolist()

        results_dir = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)
        # Mock returns (dir, timestamp) -> now just dir
        mock_create_out.return_value = str(results_dir)

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            run.main(str(config_path))


            # Verify prefix logic
            # embedding: dummy-model
            # experiment: config
            # input: gold
            # difficulty: Hard
            mock_create_out.assert_called_once_with("dummy-model_config_gold_Hard")

        finally:
            os.chdir(cwd)


def test_e2e_silver_missing_dataset(e2e_gold_setup, caplog):
    """
    Test that if a dataset file is missing, we handle it gracefully
    by printing an error message and returning.
    """
    config_path, tmp_path = e2e_gold_setup

    silver_path = tmp_path / "data" / "silver" / "test_exp_dummy-model_silver.jsonl"

    # Ensure no silver file exists
    if silver_path.exists():
        os.remove(silver_path)

    with open(config_path) as f:
        config = json.load(f)
    config["input_file"] = str(silver_path)
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Mock dependencies
    with patch("run.Vectorizer"), patch("run.create_output_directory") as mock_create_out:
        results_dir = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)
        mock_create_out.return_value = str(results_dir)

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # Capture both stdout and logs
            import io
            import sys

            captured_output = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured_output

            run.main(str(config_path))

            # Verify prefix logic
            # embedding: dummy-model
            # experiment: config (from config.json)
            # input: test_exp_dummy-model_silver (from test_exp_dummy-model_silver.jsonl)
            mock_create_out.assert_called_once_with("dummy-model_config_test_exp_dummy-model_silver")

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            # Assert error message was printed about missing silver file
            assert "not found" in output.lower()

            # Assert file was NOT created
            assert not silver_path.exists()
        finally:
            os.chdir(cwd)
