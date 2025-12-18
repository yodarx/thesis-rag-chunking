import json
from unittest.mock import MagicMock, patch

import pytest

from src.preprocessor.categorize import load_jsonl, main, save_jsonl


def test_load_jsonl(tmp_path):
    # Create a dummy jsonl file
    file_path = tmp_path / "test.jsonl"
    data = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    loaded_data = load_jsonl(str(file_path))
    assert loaded_data == data


def test_save_jsonl(tmp_path):
    file_path = tmp_path / "output.jsonl"
    data = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]

    save_jsonl(data, str(file_path))

    with open(file_path, encoding="utf-8") as f:
        loaded_data = [json.loads(line) for line in f]

    assert loaded_data == data


@patch("src.preprocessor.categorize.load_jsonl")
@patch("src.preprocessor.categorize.save_jsonl")
@patch("src.preprocessor.categorize.DatasetCategorizer")
@patch("src.preprocessor.categorize.Ollama")
@patch("os.path.exists")
@patch("os.makedirs")
def test_main(mock_makedirs, mock_exists, mock_ollama, mock_categorizer_cls, mock_save, mock_load):
    # Setup mocks
    mock_exists.return_value = True
    mock_load.return_value = [{"question": "q1"}]

    mock_categorizer_instance = MagicMock()
    mock_categorizer_cls.return_value = mock_categorizer_instance
    mock_categorizer_instance.categorize_dataset.return_value = [
        {"question": "q1", "category": "Factoid"}
    ]

    # Run main
    input_file = "input.jsonl"
    output_file = "output_dir/output.jsonl"
    model_name = "gpt-oss"

    main(input_file, output_file, model_name)

    # Assertions
    mock_exists.assert_called_with(input_file)
    mock_load.assert_called_with(input_file)
    mock_ollama.assert_called_with(model=model_name)
    mock_categorizer_cls.assert_called()
    mock_categorizer_instance.categorize_dataset.assert_called_with([{"question": "q1"}])
    mock_makedirs.assert_called_with("output_dir", exist_ok=True)
    mock_save.assert_called_with([{"question": "q1", "category": "Factoid"}], output_file)


@patch("os.path.exists")
def test_main_input_not_found(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(SystemExit):
        main("nonexistent.jsonl", "out.jsonl", "model")


@patch("src.preprocessor.categorize.load_jsonl")
@patch("src.preprocessor.categorize.save_jsonl")
@patch("src.preprocessor.categorize.DatasetCategorizer")
@patch("src.preprocessor.categorize.Ollama")
@patch("os.path.exists")
@patch("os.makedirs")
def test_main_with_limit(
    mock_makedirs, mock_exists, mock_ollama, mock_categorizer_cls, mock_save, mock_load
):
    # Setup mocks
    mock_exists.return_value = True
    # Return 3 items
    mock_load.return_value = [{"q": "1"}, {"q": "2"}, {"q": "3"}]

    mock_categorizer_instance = MagicMock()
    mock_categorizer_cls.return_value = mock_categorizer_instance
    # Should only be called with 2 items if limit is 2
    mock_categorizer_instance.categorize_dataset.return_value = [
        {"q": "1", "cat": "A"},
        {"q": "2", "cat": "B"},
    ]

    # Run main with limit=2
    main("in.jsonl", "out.jsonl", "model", limit=2)

    # Assertions
    # Verify categorize_dataset was called with sliced data
    mock_categorizer_instance.categorize_dataset.assert_called_with([{"q": "1"}, {"q": "2"}])
    mock_save.assert_called_with([{"q": "1", "cat": "A"}, {"q": "2", "cat": "B"}], "out.jsonl")
