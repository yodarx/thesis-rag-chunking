import json
from unittest.mock import MagicMock, patch

import pytest

from src.chunking.build_chunks import generate_chunks
from src.vectorizer.vectorizer import Vectorizer


@pytest.fixture
def mock_dataset():
    return [{"document_text": "doc1"}, {"document_text": "doc2"}]


@pytest.fixture
def mock_vectorizer():
    return MagicMock(spec=Vectorizer)


def test_generate_chunks_from_cache(tmp_path, mock_dataset, mock_vectorizer):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    exp_config = {
        "name": "test_exp",
        "function": "chunk_fixed_size",
    }

    # Pre-create cache file
    cache_name = "test_exp_test_exp_chunk_fixed_size_chunks.json"
    cache_path = cache_dir / cache_name
    cached_chunks = ["chunk1", "chunk2"]
    with open(cache_path, "w") as f:
        json.dump(cached_chunks, f)

    chunks = generate_chunks(exp_config, mock_dataset, mock_vectorizer, str(cache_dir))

    assert chunks == cached_chunks


@patch("src.chunking.build_chunks.get_chunking_function")
def test_generate_chunks_and_cache(mock_get_func, tmp_path, mock_dataset, mock_vectorizer):
    cache_dir = tmp_path / "cache"

    exp_config = {
        "name": "test_exp",
        "function": "chunk_fixed_size",
        "params": {"size": 100},
    }

    # Mock chunk function behavior
    mock_chunk_func = MagicMock(return_value=["c1", "c2"])
    mock_get_func.return_value = mock_chunk_func

    chunks = generate_chunks(exp_config, mock_dataset, mock_vectorizer, str(cache_dir))

    assert len(chunks) == 4  # 2 docs * 2 chunks each
    assert chunks == ["c1", "c2", "c1", "c2"]

    # Check cache file created
    cache_name = "test_exp_test_exp_chunk_fixed_size_chunks.json"
    cache_path = cache_dir / cache_name
    assert cache_path.exists()

    with open(cache_path) as f:
        data = json.load(f)
        assert data == chunks

    mock_chunk_func.assert_called_with("doc2", size=100)

    # Check metadata file
    metadata_name = "test_exp_test_exp_chunk_fixed_size_metadata.json"
    metadata_path = cache_dir / metadata_name
    assert metadata_path.exists()

    with open(metadata_path) as f:
        metadata = json.load(f)
        assert metadata["experiment_name"] == "test_exp"
        assert metadata["total_chunks"] == 4
        assert metadata["source_documents_count"] == 2
        assert metadata["total_characters"] == 8  # 4 chunks * 2 chars ("c1", "c2")
        assert metadata["avg_chars_per_chunk"] == 2.0
        assert metadata["chunks_per_second"] >= 0
        assert "processing_time_seconds" in metadata
        assert "timestamp" in metadata


@patch("src.chunking.build_chunks.get_chunking_function")
def test_semantic_chunking_vectorizer_injection(mock_get_func, tmp_path, mock_dataset, mock_vectorizer):
    cache_dir = tmp_path / "cache"

    exp_config = {
        "name": "sem_exp",
        "function": "chunk_semantic",
        "params": {"threshold": 0.5},
    }

    mock_chunk_func = MagicMock(return_value=["sc1"])
    mock_get_func.return_value = mock_chunk_func

    generate_chunks(exp_config, mock_dataset, mock_vectorizer, str(cache_dir))

    # Check if vectorizer was injected into kwargs
    call_args = mock_chunk_func.call_args
    assert call_args is not None
    _, kwargs = call_args
    assert kwargs["chunking_embeddings"] == mock_vectorizer
    assert kwargs["threshold"] == 0.5
    assert kwargs["batch_size"] == 1024  # Default injection
