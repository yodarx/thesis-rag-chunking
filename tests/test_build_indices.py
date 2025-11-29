import json
import os
from unittest import mock

import pytest

import build_indices


@pytest.fixture
def sample_config(tmp_path):
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
    # Write a minimal sample data file
    sample_data = [{"sample_id": "doc1", "document_text": "This is a test document."}]
    with open(config["input_file"], "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


@mock.patch("build_indices.faiss.write_index")
@mock.patch("build_indices.faiss.IndexIVFFlat")
@mock.patch("build_indices.faiss.IndexFlatL2")
@mock.patch("build_indices.Vectorizer")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_success(
    mock_load_data,
    mock_vectorizer,
    mock_flatl2,
    mock_ivfflat,
    mock_write_index,
    sample_config,
    tmp_path,
):
    os.chdir(tmp_path)
    # Mock dataset loader
    mock_load_data.return_value = [
        {"sample_id": "doc1", "document_text": "This is a test document."}
    ]
    # Mock vectorizer
    mock_vectorizer.from_model_name.return_value = mock_vectorizer
    mock_vectorizer.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    # Mock FAISS index
    mock_index = mock.Mock()
    mock_ivfflat.return_value = mock_index
    mock_flatl2.return_value = mock_index
    mock_index.train.return_value = None
    mock_index.add.return_value = None
    mock_index.ntotal = 1

    # Patch write_index to do nothing
    def dummy_write_index(index, path):
        with open(path, "wb") as f:
            f.write(b"FAKE INDEX")

    mock_write_index.side_effect = dummy_write_index
    # Run main
    build_indices.main(sample_config)
    # Check output files
    index_dir = tmp_path / "indices" / "fixed_test-model"
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "metadata.json").exists()


@mock.patch("build_indices.Vectorizer")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_missing_config_fields(mock_load_data, mock_vectorizer, tmp_path):
    config = {"input_file": str(tmp_path / "sample_data.jsonl")}
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    # Should not raise, but print error and return
    build_indices.main(str(config_path))


@mock.patch("build_indices.Vectorizer")
@mock.patch("build_indices.load_asqa_dataset")
def test_build_indices_empty_dataset(mock_load_data, mock_vectorizer, sample_config):
    mock_load_data.return_value = []
    build_indices.main(sample_config)
