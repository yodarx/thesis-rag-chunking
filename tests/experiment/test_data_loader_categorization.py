from unittest.mock import Mock

import pytest
from langchain_community.llms import Ollama

from src.experiment.data_loader import DatasetCategorizer


@pytest.fixture
def mock_llm():
    llm = Mock(spec=Ollama)
    return llm


def test_categorize_with_llm(mock_llm):
    mock_llm.invoke.return_value = "Multihop | Hard"
    categorizer = DatasetCategorizer(llm=mock_llm)

    cat, diff = categorizer._categorize_with_llm("Complex question")

    assert cat == "Multihop"
    assert diff == "Hard"
    mock_llm.invoke.assert_called_once()


def test_categorize_dataset(mock_llm):
    mock_llm.invoke.return_value = "Factoid | Easy"
    categorizer = DatasetCategorizer(llm=mock_llm)

    dataset = [{"question": "Simple question", "id": 1}]
    updated_dataset = categorizer.categorize_dataset(dataset)

    assert len(updated_dataset) == 1
    assert updated_dataset[0]["category"] == "Factoid"
    assert updated_dataset[0]["difficulty"] == "Easy"
    assert updated_dataset[0]["id"] == 1
