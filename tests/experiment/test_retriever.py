import numpy as np
import pytest
from pytest_mock import MockerFixture

from src.experiment.retriever import FaissRetriever
from src.vectorizer.vectorizer import Vectorizer


@pytest.fixture
def mock_vectorizer(mocker: MockerFixture) -> Vectorizer:
    mock = mocker.Mock(spec=Vectorizer)
    # Simuliere die Einbettung der FRAGE
    mock.embed_documents.return_value = [[0.9, 0.1, 0.0]]
    return mock


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Erstellt Beispiel-Chunk-Embeddings."""
    # chunk 0: [1.0, 0.0, 0.0] (am nächsten zur Frage)
    # chunk 1: [0.0, 1.0, 0.0] (weit weg)
    # chunk 2: [0.0, 0.0, 1.0] (weit weg)
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32")


@pytest.fixture
def mock_faiss(mocker: MockerFixture):
    """Mockt die FAISS Index-Klasse und ihre Methoden."""
    mock_index_instance = mocker.Mock()
    # Simuliere, dass die Suche [0] (chunk 0) als nächsten Index findet
    mock_index_instance.search.return_value = (
        np.array([[0.1]], dtype="float32"),  # Distanzen
        np.array([[0]], dtype="int64"),  # Indizes
    )

    mock_index_constructor = mocker.patch("faiss.IndexFlatL2", return_value=mock_index_instance)
    return mock_index_constructor, mock_index_instance


def test_retriever_search_success(
    mock_vectorizer: Vectorizer, sample_embeddings: np.ndarray, mock_faiss
):
    mock_index_constructor, mock_index_instance = mock_faiss
    mock_index_instance.ntotal = 3
    retriever = FaissRetriever(mock_vectorizer)
    retriever.index = mock_index_instance
    retriever.chunks = ["chunk0", "chunk1", "chunk2"]
    question = "test question"
    k = 1

    result = retriever.retrieve(question, k)
    assert result == ["chunk0"]


def test_retriever_search_empty_embeddings(mock_vectorizer: Vectorizer):
    """Testet, ob bei leeren Embeddings eine leere Liste zurckgegeben wird."""
    retriever = FaissRetriever(mock_vectorizer)
    retriever.index = None
    with pytest.raises(RuntimeError):
        retriever.retrieve("test", 5)


def test_retriever_search_k_too_large(
    mock_vectorizer: Vectorizer, sample_embeddings: np.ndarray, mock_faiss
):
    """Testet, ob k korrekt auf die Anzahl der Dokumente begrenzt wird."""
    mock_index_constructor, mock_index_instance = mock_faiss
    mock_index_instance.ntotal = 3
    retriever = FaissRetriever(mock_vectorizer)
    retriever.index = mock_index_instance
    retriever.chunks = ["chunk0", "chunk1", "chunk2"]
    k = 10  # k ist grer als die 3 Dokumente im Index
    result = retriever.retrieve("test", k)
    assert result == ["chunk0"]
