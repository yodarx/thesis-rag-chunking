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

    retriever = FaissRetriever(mock_vectorizer)
    question = "test question"
    k = 1

    indices = retriever.search(question, sample_embeddings, k)

    # 1. Wurde der Vektorizer für die Frage aufgerufen?
    mock_vectorizer.embed_documents.assert_called_once_with([question])

    # 2. Wurde der FAISS-Index korrekt erstellt?
    mock_index_constructor.assert_called_once_with(sample_embeddings.shape[1])  # Dimension
    mock_index_instance.add.assert_called_once()

    # 3. Wurde die Suche korrekt aufgerufen?
    mock_index_instance.search.assert_called_once()
    assert mock_index_instance.search.call_args[0][1] == k  # k

    # 4. Ist das Ergebnis korrekt?
    assert indices == [0]


def test_retriever_search_empty_embeddings(mock_vectorizer: Vectorizer):
    """Testet, ob bei leeren Embeddings eine leere Liste zurückgegeben wird."""
    retriever = FaissRetriever(mock_vectorizer)
    empty_embeddings = np.array([], dtype="float32").reshape(0, 10)  # 0 Vektoren

    indices = retriever.search("test", empty_embeddings, 5)

    assert indices == []
    mock_vectorizer.embed_documents.assert_not_called()


def test_retriever_search_k_too_large(
    mock_vectorizer: Vectorizer, sample_embeddings: np.ndarray, mock_faiss
):
    """Testet, ob k korrekt auf die Anzahl der Dokumente begrenzt wird."""
    mock_index_constructor, mock_index_instance = mock_faiss

    retriever = FaissRetriever(mock_vectorizer)
    k = 10  # k ist größer als die 3 Dokumente im Index

    retriever.search("test", sample_embeddings, k)

    # Überprüfe, ob k in index.search() auf 3 (die Anzahl der Chunks) begrenzt wurde
    expected_k_for_search = 3
    assert mock_index_instance.search.call_args[0][1] == expected_k_for_search
