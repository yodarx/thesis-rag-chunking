import json
import os
from unittest.mock import patch

import faiss
import numpy as np
import pytest

import build_silver
import run


@pytest.fixture
def e2e_setup(tmp_path):
    # Setup config
    config = {
        "embedding_model": "dummy-model",
        "silver_limit": 2,
        "llm_model": "dummy-llm",
        "hops_count": 2,
        "top_k": 2,
        "input_file": "placeholder",  # Will be updated in test
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
    # Structure: data/indices/{index_name}/chunks.json
    # index_name = experiment_name + "_" + sanitized_model_name
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
    # Add random vectors for the 4 chunks
    vectors = np.random.rand(len(chunks), dimension).astype("float32")
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    return config_path, tmp_path


def test_e2e_silver_pipeline(e2e_setup):
    """
    End-to-End test for silver standard generation AND execution.
    1. Generates silver standard dataset.
    2. Runs the experiment using the generated dataset.
    """
    config_path, tmp_path = e2e_setup

    # Mock Vectorizer and Ollama
    # We patch Vectorizer in both modules because they have already imported it
    with (
        patch("build_silver.Vectorizer") as mock_vectorizer_build,
        patch("run.Vectorizer") as mock_vectorizer_run,
        patch("build_silver.Ollama") as mock_ollama,
        patch("run.create_output_directory") as mock_create_out,
    ):
        # Setup Mock LLM response
        mock_llm_instance = mock_ollama.return_value
        mock_llm_instance.invoke.return_value = json.dumps(
            {"question": "What is the combined meaning of these chunks?", "answer": "42"}
        )

        # Setup Mock Vectorizer
        # We want both build_silver and run to use a mock that behaves the same way
        mock_vectorizer_instance = mock_vectorizer_build.from_model_name.return_value

        # Return random embeddings of dimension 10 (matching our dummy index)
        # embed_documents returns list[list[float]]
        def side_effect_embed(docs):
            return np.random.rand(len(docs), 10).tolist()

        mock_vectorizer_instance.embed_documents.side_effect = side_effect_embed

        # Ensure run.py gets the same mock instance
        mock_vectorizer_run.from_model_name.return_value = mock_vectorizer_instance

        # Setup output directory mock
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        mock_create_out.return_value = (str(results_dir), "test_timestamp")

        # --- Step 1: Generate Silver Standard ---
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            build_silver.main(str(config_path))

            # Verify output file creation
            silver_path = tmp_path / "data" / "silver" / "test_exp_dummy-model_silver.jsonl"
            assert silver_path.exists(), "Output silver dataset file was not created"

            # --- Step 2: Run Experiment ---
            # Update config to use the generated silver file
            with open(config_path) as f:
                config = json.load(f)
            config["input_file"] = str(silver_path)
            with open(config_path, "w") as f:
                json.dump(config, f)

            # Run the experiment runner
            run.main(str(config_path))

            # Verify results
            # Should have created a summary CSV
            # The ResultsHandler saves to output_dir
            # It creates "test_timestamp_detailed_results.csv" and "test_timestamp_summary.csv" (maybe)
            # Let's check what files are in results_dir
            files = list(results_dir.glob("*"))
            assert len(files) > 0, "No results files generated"

            detailed_csv = results_dir / "test_timestamp_detailed_results.csv"
            assert detailed_csv.exists(), "Detailed results CSV not found"

            # Check content of detailed results
            with open(detailed_csv) as f:
                content = f.read()
                # Should contain our question
                assert "What is the combined meaning of these chunks?" in content

        finally:
            os.chdir(cwd)
