import json
import os
from unittest.mock import patch

import faiss
import numpy as np
import pytest

import run


@pytest.fixture
def e2e_gold_setup(tmp_path):
    # Setup config
    config = {
        "embedding_model": "dummy-model",
        "top_k": 2,
        "input_file": "placeholder",  # Will be updated in test
        "silver_file": "data/silver/test_exp_dummy-model_silver.jsonl",
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
    chunks_path = index_dir / "chunks.json"
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

    # Mock Vectorizer
    with (
        patch("run.Vectorizer") as mock_vectorizer_run,
        patch("run.create_output_directory") as mock_create_out,
    ):
        # Setup Mock Vectorizer
        mock_vectorizer_instance = mock_vectorizer_run.from_model_name.return_value

        def side_effect_embed(docs):
            return np.random.rand(len(docs), 10).tolist()

        mock_vectorizer_instance.embed_documents.side_effect = side_effect_embed

        # Setup output directory mock
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        mock_create_out.return_value = (str(results_dir), "test_timestamp")

        # Run the experiment runner
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            run.main(str(config_path))

            # Verify results
            files = list(results_dir.glob("*"))
            assert len(files) > 0, "No results files generated"

            detailed_csv = results_dir / "test_timestamp_detailed_results.csv"
            assert detailed_csv.exists(), "Detailed results CSV not found"

            # Check content of detailed results
            with open(detailed_csv) as f:
                content = f.read()
                assert "What is in Chunk 1?" in content
                assert "gold_1" in content

        finally:
            os.chdir(cwd)


def test_e2e_silver_missing_file_warning(e2e_gold_setup, caplog):
    """
    Test that if silver file is missing, we handle it gracefully
    by printing an error message and returning.
    """
    config_path, tmp_path = e2e_gold_setup

    silver_path = tmp_path / "data" / "silver" / "test_exp_dummy-model_silver.jsonl"

    # Ensure no silver file exists
    if silver_path.exists():
        os.remove(silver_path)

    # Mock dependencies
    with patch("run.Vectorizer"), patch("run.create_output_directory") as mock_create_out:
        results_dir = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)
        mock_create_out.return_value = (str(results_dir), "test_ts")

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # Capture both stdout and logs
            import io
            import sys

            captured_output = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured_output

            # Run main with config path AND use_silver=True
            run.main(str(config_path), use_silver=True)

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            # Assert error message was printed about missing silver file
            assert "Error: Silver file not found" in output or "not found" in output.lower()

            # Assert file was NOT created
            assert not silver_path.exists()
        finally:
            os.chdir(cwd)
