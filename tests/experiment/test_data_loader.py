import json

import pytest
from pytest_mock import MockerFixture

from src.experiment.data_loader import load_asqa_dataset


@pytest.fixture
def mock_jsonl_file(mocker: MockerFixture):
    """Erstellt einen Mock für eine JSONL-Datei."""
    sample_data = [
        {"id": 1, "text": "Zeile 1"},
        {"id": 2, "text": "Zeile 2"},
        {"id": 3, "text": "Zeile 3"},
    ]
    # Konvertiere Dicts in JSON-Strings mit Zeilenumbruch
    mock_content = "\n".join([json.dumps(line) for line in sample_data])

    # Mocke 'builtins.open'
    m = mocker.patch("builtins.open", mocker.mock_open(read_data=mock_content))
    return m, sample_data


def test_load_asqa_dataset_full(mock_jsonl_file):
    """Testet das Laden der gesamten Datei."""
    mock_open, expected_data = mock_jsonl_file

    data = load_asqa_dataset("dummy/path.jsonl")

    mock_open.assert_called_once_with("dummy/path.jsonl", encoding="utf-8")
    assert data == expected_data


def test_load_asqa_dataset_with_limit(mock_jsonl_file):
    """Testet, ob der 'limit'-Parameter funktioniert."""
    mock_open, expected_data = mock_jsonl_file

    limit = 2
    data = load_asqa_dataset("dummy/path.jsonl", limit=limit)

    assert len(data) == limit
    assert data == expected_data[:limit]


def test_load_asqa_dataset_file_not_found(mocker: MockerFixture):
    """Testet das Verhalten bei einem FileNotFoundError."""
    mocker.patch("builtins.open", side_effect=FileNotFoundError("Test error"))

    data = load_asqa_dataset("bad/path.jsonl")

    assert data == []
