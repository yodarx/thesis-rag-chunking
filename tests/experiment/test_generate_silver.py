import os
from unittest.mock import mock_open, patch

import pytest

from src.experiment.generate_silver import main


@pytest.fixture
def mock_dependencies():
    with (
        patch("src.experiment.generate_silver.os.path.exists") as mock_exists,
        patch("src.experiment.generate_silver.os.makedirs") as mock_makedirs,
        patch("src.experiment.generate_silver.Vectorizer") as mock_vectorizer,
        patch("src.experiment.generate_silver.FaissRetriever") as mock_retriever_cls,
        patch("src.experiment.generate_silver.Ollama") as mock_ollama,
        patch("src.experiment.generate_silver.SilverStandardGenerator") as mock_generator_cls,
        patch("builtins.open", new_callable=mock_open) as mock_file,
    ):
        yield {
            "exists": mock_exists,
            "makedirs": mock_makedirs,
            "vectorizer": mock_vectorizer,
            "retriever_cls": mock_retriever_cls,
            "ollama": mock_ollama,
            "generator_cls": mock_generator_cls,
            "file": mock_file,
        }


def test_main_success(mock_dependencies):
    mocks = mock_dependencies
    mocks["exists"].return_value = True

    # Setup generator mock to return a dummy dataset
    mock_generator_instance = mocks["generator_cls"].return_value
    mock_generator_instance.generate_dataset.return_value = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]

    index_path = "indices/test/index.faiss"
    chunks_path = "indices/test/chunks.json"
    output_file = "data/silver/silver.jsonl"
    model_name = "gpt-oss"
    limit = 2

    main(index_path, chunks_path, output_file, model_name, limit)

    # Verify checks
    mocks["exists"].assert_any_call(index_path)
    mocks["exists"].assert_any_call(chunks_path)

    # Verify initialization
    mocks["vectorizer"].from_model_name.assert_called_once_with("all-MiniLM-L6-v2")
    mocks["retriever_cls"].assert_called_once()
    mocks["retriever_cls"].return_value.load_index.assert_called_once_with(index_path, chunks_path)
    mocks["ollama"].assert_called_once_with(model=model_name)
    mocks["generator_cls"].assert_called_once()

    # Verify generation
    mock_generator_instance.generate_dataset.assert_called_once_with(limit)

    # Verify output
    mocks["makedirs"].assert_called_once_with(os.path.dirname(output_file), exist_ok=True)
    mocks["file"].assert_called_once_with(output_file, "w", encoding="utf-8")

    # Check file writes
    handle = mocks["file"]()
    # We expect 2 writes (one per entry) plus newlines if implemented that way,
    # but the code does `f.write(json.dumps(...) + "\n")`
    assert handle.write.call_count == 2
    # Check content of writes roughly
    args_list = handle.write.call_args_list
    assert "Q1" in args_list[0][0][0]
    assert "Q2" in args_list[1][0][0]


def test_main_index_not_found_but_chunks_exist(mock_dependencies):
    mocks = mock_dependencies

    def side_effect(path):
        if path == "chunks.json":
            return True
        if path == "missing_index.faiss":
            return False
        return False

    mocks["exists"].side_effect = side_effect

    with patch("json.load", return_value=["chunk1", "chunk2"]):
        main("missing_index.faiss", "chunks.json", "out.jsonl", "model", 10)

    # Should not raise SystemExit
    mocks["retriever_cls"].return_value.load_index.assert_not_called()
    # Should have set chunks manually
    assert mocks["retriever_cls"].return_value.chunks == ["chunk1", "chunk2"]


def test_main_limit_all(mock_dependencies):
    mocks = mock_dependencies
    mocks["exists"].return_value = True

    # Mock loaded chunks
    mock_retriever = mocks["retriever_cls"].return_value
    mock_retriever.chunks = ["c1", "c2", "c3"]

    main("index.faiss", "chunks.json", "out.jsonl", "model", -1)

    # Should use len(chunks) as limit
    mocks["generator_cls"].return_value.generate_dataset.assert_called_once_with(3)


def test_main_chunks_not_found(mock_dependencies):
    mocks = mock_dependencies
    # chunks check is first now
    mocks["exists"].return_value = False

    with pytest.raises(SystemExit) as excinfo:
        main("index.faiss", "missing_chunks.json", "out.jsonl", "model", 10)

    assert excinfo.value.code == 1
