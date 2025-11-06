import numpy as np
import pytest
from pytest_mock import MockerFixture
from sentence_transformers import SentenceTransformer

from vectorizer.vectorizer import Vectorizer


@pytest.fixture
def mock_sentence_model(mocker: MockerFixture) -> SentenceTransformer:
    """
    Erstellt ein Mock-Objekt, das sich wie ein SentenceTransformer-Modell verhält.
    """
    # Erstelle ein Mock-Objekt, das die Spezifikation (spec)
    # der echten Klasse einhält
    mock_model = mocker.Mock(spec=SentenceTransformer)

    # Definiere, was die 'encode'-Methode zurückgeben soll: ein numpy-Array
    mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_model.encode.return_value = mock_embeddings

    return mock_model


def test_vectorizer_init(mock_sentence_model: SentenceTransformer):
    """Testet, ob die Initialisierung das Modell korrekt speichert."""
    vectorizer = Vectorizer(model=mock_sentence_model)
    assert vectorizer.model is mock_sentence_model


def test_embed_documents_calls_model_correctly(mock_sentence_model: SentenceTransformer):
    """
    Testet, ob embed_documents die 'encode'-Methode des Modells
    mit den korrekten Argumenten aufruft.
    """
    vectorizer = Vectorizer(model=mock_sentence_model)
    documents = ["Hallo Welt", "Dies ist ein Test"]

    result = vectorizer.embed_documents(documents)

    # 1. Überprüfen, ob das Modell korrekt aufgerufen wurde
    mock_sentence_model.encode.assert_called_once_with(documents, show_progress_bar=False)

    # 2. Überprüfen, ob das Ergebnis korrekt in eine Liste umgewandelt wurde
    expected_list = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert result == expected_list


def test_embed_documents_with_empty_list(mock_sentence_model: SentenceTransformer):
    """
    Testet, ob eine leere Liste von Dokumenten eine leere Liste zurückgibt,
    ohne das Modell aufzurufen.
    """
    vectorizer = Vectorizer(model=mock_sentence_model)
    result = vectorizer.embed_documents([])

    # 1. Das Ergebnis sollte eine leere Liste sein
    assert result == []

    # 2. Die 'encode'-Methode sollte nie aufgerufen worden sein
    mock_sentence_model.encode.assert_not_called()


def test_from_model_name_factory_method(mocker: MockerFixture):
    """
    Testet die '.from_model_name' Factory-Methode, um sicherzustellen,
    dass sie den SentenceTransformer-Konstruktor aufruft.
    """
    model_name = "test-model-name"

    # Mocke die *gesamte* SentenceTransformer-Klasse
    mock_model_instance = mocker.Mock(spec=SentenceTransformer)
    mock_constructor = mocker.patch(
        "vectorizer.vectorizer.SentenceTransformer",  # Pfad zur Klasse in *deiner* Datei
        return_value=mock_model_instance,
    )

    # Rufe die Factory-Methode auf
    vectorizer = Vectorizer.from_model_name(model_name)

    # 1. Überprüfe, ob der Konstruktor mit dem Namen aufgerufen wurde
    mock_constructor.assert_called_once_with(model_name)

    # 2. Überprüfe, ob die erstellte Instanz im Vectorizer gespeichert wurde
    assert vectorizer.model is mock_model_instance


def test_from_model_name_factory_method_gpu_selection(mocker: MockerFixture):
    """
    Tests the '.from_model_name' factory method to ensure it selects GPU if available,
    otherwise CPU, and passes the correct device to SentenceTransformer.
    """
    model_name = "test-model-name"
    mock_model_instance = mocker.Mock(spec=SentenceTransformer)
    mock_constructor = mocker.patch(
        "vectorizer.vectorizer.SentenceTransformer",
        return_value=mock_model_instance,
    )

    # Case 1: GPU available
    mocker.patch("torch.cuda.is_available", return_value=True)
    vectorizer_gpu = Vectorizer.from_model_name(model_name)
    mock_constructor.assert_called_with(model_name, device="cuda")
    assert vectorizer_gpu.model is mock_model_instance

    mock_constructor.reset_mock()

    # Case 2: GPU not available
    mocker.patch("torch.cuda.is_available", return_value=False)
    vectorizer_cpu = Vectorizer.from_model_name(model_name)
    mock_constructor.assert_called_with(model_name, device=None)
    assert vectorizer_cpu.model is mock_model_instance
