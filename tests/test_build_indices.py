import json
import os
from pathlib import Path
from unittest import mock

import pytest

import build_indices


@pytest.fixture
def sample_config(tmp_path: Path) -> str:
    config = {
        "input_file": str(tmp_path / "sample_data.jsonl"),
        "embedding_model": "test-model",
        "output_dir": str(tmp_path / "indices"),
        "cache_dir": str(tmp_path / "chunks"),  # Setup temp chunks dir
        "experiments": [
            {
                "name": "fixed",
                "function": "chunk_fixed_size",
                "params": {"chunk_size": 10, "chunk_overlap": 0},
            }
        ],
    }
    # Write a minimal sample data file
    sample_data = [{"sample_id": "doc1", "document_text": "This is a test document."}]
    with open(config["input_file"], "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


@mock.patch("os.path.exists")
@mock.patch("src.chunking.chunk_fixed.chunk_fixed_size")
@mock.patch("build_indices.faiss.write_index")
@mock.patch("build_indices.faiss.IndexIVFFlat")
@mock.patch("build_indices.faiss.IndexFlatL2")
@mock.patch("src.vectorizer.vectorizer.Vectorizer.from_model_name")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_success(
    mock_load_data: mock.Mock,
    mock_vectorizer_from_model_name: mock.Mock,
    mock_flatl2: mock.Mock,
    mock_ivfflat: mock.Mock,
    mock_write_index: mock.Mock,
    mock_chunk_fixed_size: mock.Mock,
    mock_path_exists: mock.Mock,
    sample_config: str,
    tmp_path: Path,
) -> None:
    # Force cache miss for chunks.json to ensure we run chunking
    # Update side_effect to match new path structure: .../chunks.json
    mock_path_exists.side_effect = (
        lambda p: False if str(p).endswith("chunks.json") else Path(p).exists()
    )

    # Mock dataset loader
    mock_load_data.return_value = [
        {"sample_id": "doc1", "document_text": "This is a test document."}
    ]
    mock_chunk_fixed_size.return_value = ["This is a test chunk."]
    mock_flatl2.return_value = mock.Mock()
    mock_flatl2.return_value.add = mock.Mock()  # Ensure add method exists
    mock_vectorizer = mock.Mock()
    mock_vectorizer.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_vectorizer_from_model_name.return_value = mock_vectorizer
    build_indices.main(sample_config)
    mock_write_index.assert_called()


@mock.patch("os.path.exists")
@mock.patch("src.chunking.chunk_fixed.chunk_fixed_size")
@mock.patch("build_indices.faiss.write_index")
@mock.patch("build_indices.build_faiss_index")
@mock.patch("src.vectorizer.vectorizer.Vectorizer.from_model_name")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_with_custom_batch_size(
    mock_load_data: mock.Mock,
    mock_vectorizer_from_model_name: mock.Mock,
    mock_build_faiss_index: mock.Mock,
    mock_write_index: mock.Mock,
    mock_chunk_fixed_size: mock.Mock,
    mock_path_exists: mock.Mock,
    sample_config: str,
) -> None:
    # Force cache miss so logic uses our mocks, but allow other checks to pass
    # Use Path(p).exists() to avoid recursion if os.path.exists is mocked
    mock_path_exists.side_effect = (
        lambda p: False if str(p).endswith("chunks.json") else Path(p).exists()
    )

    # Create multiple chunks to test batching
    mock_load_data.return_value = [
        {"sample_id": "doc1", "document_text": "This is a test document."}
    ]
    # Create 100 chunks to test batching
    chunks = [f"Chunk {i}" for i in range(100)]
    mock_chunk_fixed_size.return_value = chunks
    mock_vectorizer = mock.Mock()
    mock_vectorizer_from_model_name.return_value = mock_vectorizer

    # Mock build_faiss_index to return a fake index so saving proceeds
    mock_index = mock.Mock()
    mock_index.ntotal = 100
    mock_build_faiss_index.return_value = mock_index

    # Call with custom batch size of 16
    build_indices.main(sample_config, batch_size=16)

    # Verify the index was built and written
    mock_write_index.assert_called()

    # Verify proper batch size passed to build_faiss_index
    # build_faiss_index(chunks, vectorizer, gpu_batch_size)
    assert mock_build_faiss_index.called
    _args, kwargs = mock_build_faiss_index.call_args
    assert kwargs.get("gpu_batch_size") == 16, (
        f"Expected gpu_batch_size=16, got {kwargs.get('gpu_batch_size')}"
    )


@mock.patch("build_indices.faiss.write_index")
@mock.patch("build_indices.Vectorizer")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_with_custom_output_dir(
    mock_load_data, mock_vectorizer, mock_write_index, tmp_path
):
    custom_output_dir = str(tmp_path / "custom_indices")
    config = {
        "input_file": str(tmp_path / "sample_data.jsonl"),
        "embedding_model": "test-model",
        "output_dir": custom_output_dir,
        "experiments": [
            {
                "name": "fixed",
                "function": "chunk_fixed_size",
                "params": {"chunk_size": 10, "chunk_overlap": 0},
            }
        ],
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    mock_load_data.return_value = []
    build_indices.main(str(config_path))


@mock.patch("build_indices.faiss.write_index")
@mock.patch("build_indices.Vectorizer")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_defaults_output_dir(
    mock_load_data, mock_vectorizer, mock_write_index, tmp_path
):
    config = {
        "input_file": str(tmp_path / "sample_data.jsonl"),
        "embedding_model": "test-model",
        "experiments": [
            {
                "name": "fixed",
                "function": "chunk_fixed_size",
                "params": {"chunk_size": 10, "chunk_overlap": 0},
            }
        ],
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    mock_load_data.return_value = []
    build_indices.main(str(config_path))


@mock.patch("build_indices.Vectorizer")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_missing_config_fields(mock_load_data, mock_vectorizer, tmp_path):
    config = {
        "input_file": str(tmp_path / "sample_data.jsonl"),
        "output_dir": str(tmp_path / "indices"),
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    # Should not raise, but print error and return
    build_indices.main(str(config_path))


@mock.patch("build_indices.faiss.write_index")
@mock.patch("build_indices.Vectorizer")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_empty_dataset(
    mock_load_data, mock_vectorizer, mock_write_index, sample_config
):
    mock_load_data.return_value = []

    original_exists = os.path.exists
    with mock.patch(
        "os.path.exists",
        side_effect=lambda p: False if str(p).endswith("chunks.json") else original_exists(p),
    ):
        build_indices.main(sample_config)

    # write_index might be called even if empty?
    # If chunks is empty, vectorizer returns empty. index added with 0 items.
    # check if called.
    assert not mock_write_index.called


@mock.patch("build_indices.main")
def test_cli_with_batch_size(mock_main):
    """Test that CLI correctly passes batch-size argument"""
    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["build_indices.py", "--config", "test_config.json", "--batch-size", "64"]
        build_indices.cli_entry()
        # Verify main was called with batch_size=64
        mock_main.assert_called_once_with("test_config.json", None, 64)
    finally:
        sys.argv = original_argv


@mock.patch("src.chunking.chunk_fixed.chunk_fixed_size")
@mock.patch("build_indices.faiss.write_index")
@mock.patch("build_indices.faiss.IndexIVFFlat")
@mock.patch("build_indices.faiss.IndexFlatL2")
@mock.patch("src.vectorizer.vectorizer.Vectorizer.from_model_name")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_skips_if_index_exists(
    mock_load_data: mock.Mock,
    mock_vectorizer_from_model_name: mock.Mock,
    mock_flatl2: mock.Mock,
    mock_ivfflat: mock.Mock,
    mock_write_index: mock.Mock,
    mock_chunk_fixed_size: mock.Mock,
    sample_config: str,
    tmp_path: Path,
) -> None:
    # Setup mocks
    mock_load_data.return_value = [
        {"sample_id": "doc1", "document_text": "This is a test document."}
    ]
    mock_chunk_fixed_size.return_value = ["This is a test chunk."]
    mock_flatl2.return_value = mock.Mock()
    mock_flatl2.return_value.add = mock.Mock()
    mock_vectorizer = mock.Mock()
    mock_vectorizer.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_vectorizer_from_model_name.return_value = mock_vectorizer

    # Load config to get output_dir
    with open(sample_config) as f:
        config = json.load(f)
    output_dir = config["output_dir"]

    # Create dummy index files in the expected location
    index_dir = Path(output_dir) / "fixed_test-model"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index.faiss").write_text("dummy")
    (index_dir / "chunks.json").write_text("[]")
    (index_dir / "metadata.json").write_text('{"build_time_seconds": 1.23}')

    build_indices.main(sample_config)
    mock_write_index.assert_not_called()
