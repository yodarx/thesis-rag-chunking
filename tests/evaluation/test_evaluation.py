import numpy as np
import pytest
from pytest_mock import MockerFixture

from evaluation.evaluation import (
    calculate_f1_score_at_k,
    calculate_map,
    calculate_metrics,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)

# --- Fixtures für Testdaten ---


@pytest.fixture
def perfect_scores() -> list[int]:
    """Relevanz-Scores, bei denen das erste Ergebnis relevant ist."""
    return [1, 1, 1]


@pytest.fixture
def good_scores() -> list[int]:
    """Relevanz-Scores mit Treffern in der Mitte."""
    return [0, 1, 1, 0, 1]


@pytest.fixture
def poor_scores() -> list[int]:
    """Relevanz-Scores mit einem späten Treffer."""
    return [0, 0, 0, 1, 0]


@pytest.fixture
def no_scores() -> list[int]:
    """Keine relevanten Ergebnisse."""
    return [0, 0, 0, 0, 0]


# --- Tests für einzelne Metrik-Helfer ---


def test_calculate_mrr(perfect_scores, good_scores, poor_scores, no_scores):
    assert calculate_mrr(perfect_scores) == pytest.approx(1.0)
    assert calculate_mrr(good_scores) == pytest.approx(0.5)  # 1 / 2. Position
    assert calculate_mrr(poor_scores) == pytest.approx(0.25)  # 1 / 4. Position
    assert calculate_mrr(no_scores) == pytest.approx(0.0)
    assert calculate_mrr([]) == pytest.approx(0.0)


def test_calculate_map(perfect_scores, good_scores, poor_scores, no_scores):
    # P@1=1/1, P@2=2/2, P@3=3/3. mean(1, 1, 1) = 1.0
    assert calculate_map(perfect_scores) == pytest.approx(1.0)

    # P@2=1/2, P@3=2/3, P@5=3/5. mean(0.5, 0.666, 0.6) = 0.588...
    assert calculate_map(good_scores) == pytest.approx(np.mean([0.5, 2 / 3, 0.6]))

    # P@4=1/4. mean(0.25) = 0.25
    assert calculate_map(poor_scores) == pytest.approx(0.25)

    assert calculate_map(no_scores) == pytest.approx(0.0)
    assert calculate_map([]) == pytest.approx(0.0)


def test_calculate_ndcg(perfect_scores, good_scores, poor_scores, no_scores):
    k = 5
    # Perfect DCG == Perfect IDCG
    assert calculate_ndcg(perfect_scores, k) == pytest.approx(1.0)

    # DCG = 1/log2(3) + 1/log2(4) + 1/log2(6) = 0.6309 + 0.5 + 0.3868 = 1.5177
    # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309 + 0.5 = 2.1309
    # NDCG = 1.5177 / 2.1309 = 0.712...
    dcg = 1 / np.log2(2 + 1) + 1 / np.log2(3 + 1) + 1 / np.log2(5 + 1)
    idcg = 1 / np.log2(1 + 1) + 1 / np.log2(2 + 1) + 1 / np.log2(3 + 1)
    assert calculate_ndcg(good_scores, k) == pytest.approx(dcg / idcg)

    # DCG = 1/log2(5) = 0.4306
    # IDCG = 1/log2(2) = 1.0
    # NDCG = 0.4306 / 1.0 = 0.4306...
    assert calculate_ndcg(poor_scores, k) == pytest.approx((1 / np.log2(4 + 1)) / 1.0)

    assert calculate_ndcg(no_scores, k) == pytest.approx(0.0)
    assert calculate_ndcg([], k) == pytest.approx(0.0)


def test_calculate_precision_at_k(good_scores, no_scores):
    # [0, 1, 1, 0, 1] -> 3 relevante
    # P@3: sum([0, 1, 1]) = 2. 2 / 3 = 0.666...
    assert calculate_precision_at_k(good_scores[:3]) == pytest.approx(2 / 3)

    # P@5: sum([0, 1, 1, 0, 1]) = 3. 3 / 5 = 0.6
    assert calculate_precision_at_k(good_scores) == pytest.approx(0.6)

    assert calculate_precision_at_k(no_scores[:3]) == pytest.approx(0.0)
    assert calculate_precision_at_k([]) == pytest.approx(0.0)


def test_calculate_recall_at_k(good_scores, no_scores):
    total_relevant = 3  # (basierend auf good_scores)

    # R@3: sum([0, 1, 1]) = 2. 2 / 3 = 0.666...
    assert calculate_recall_at_k(good_scores[:3], total_relevant) == pytest.approx(2 / 3)

    # R@5: sum([0, 1, 1, 0, 1]) = 3. 3 / 3 = 1.0
    assert calculate_recall_at_k(good_scores, total_relevant) == pytest.approx(1.0)

    # Edge case: keine relevanten Dokumente
    assert calculate_recall_at_k(good_scores, 0) == pytest.approx(0.0)
    assert calculate_recall_at_k(no_scores, total_relevant) == pytest.approx(0.0)


def test_calculate_f1_score_at_k():
    assert calculate_f1_score_at_k(0.5, 0.5) == pytest.approx(0.5)
    assert calculate_f1_score_at_k(0.8, 1.0) == pytest.approx(8 / 9)
    assert calculate_f1_score_at_k(0.0, 0.0) == pytest.approx(0.0)
    assert calculate_f1_score_at_k(0.0, 1.0) == pytest.approx(0.0)


# --- Integrationstests für calculate_metrics ---


@pytest.fixture
def sample_data():
    """Ein Beispieldatensatz für Integrationstests."""
    retrieved_chunks = [
        "Dies ist der erste Chunk. Er enthält GOLD EINS.",  # R=1
        "Dies ist ein irrelevanter Chunk.",  # R=0
        "Der dritte Chunk hat GOLD ZWEI.",  # R=1
        "Noch ein irrelevanter Chunk.",  # R=0
    ]
    gold_passages = ["Gold Eins", "Gold Zwei", "Gold Drei"]
    return retrieved_chunks, gold_passages


def test_calculate_metrics_integration(sample_data):
    retrieved_chunks, gold_passages = sample_data
    k = 3

    # Erwartete Relevanz für k=3: [1, 0, 1]
    # Total Gold = 3

    metrics = calculate_metrics(retrieved_chunks, gold_passages, k)

    # P@3 = 2 / 3
    precision = 2 / 3
    # R@3 = 2 / 3
    recall = 2 / 3

    assert metrics["precision_at_k"] == pytest.approx(precision)
    assert metrics["recall_at_k"] == pytest.approx(recall)
    assert metrics["f1_score_at_k"] == pytest.approx(calculate_f1_score_at_k(precision, recall))

    # MRR: Erster Treffer an Position 1. 1/1 = 1.0
    assert metrics["mrr"] == pytest.approx(1.0)

    # MAP: P@1=1/1, P@3=2/3. mean(1, 0.666) = 0.833...
    assert metrics["map"] == pytest.approx(np.mean([1.0, 2 / 3]))

    # NDCG@3:
    # Scores = [1, 0, 1]
    # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5 = 1.5
    # IDCG (sorted [1, 1, 0]) = 1/log2(2) + 1/log2(3) + 0/log2(4) = 1 + 0.6309 + 0.5 = 2.1309 (Fehler, IDCG sortiert [1,1,0])
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    dcg = 1 / np.log2(1 + 1) + 1 / np.log2(3 + 1)
    idcg = 1 / np.log2(1 + 1) + 1 / np.log2(2 + 1)
    assert metrics["ndcg_at_k"] == pytest.approx(dcg / idcg)


def test_calculate_metrics_empty_inputs():
    """Testet, ob bei leeren Eingaben Nullen zurückgegeben werden."""
    empty_metrics = {
        "mrr": 0.0,
        "map": 0.0,
        "ndcg_at_k": 0.0,
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "f1_score_at_k": 0.0,
    }

    # Leere Chunks
    metrics = calculate_metrics([], ["Gold"], k=3)
    assert metrics == empty_metrics

    # Leere Gold-Passagen
    metrics = calculate_metrics(["Chunk 1"], [], k=3)
    assert metrics == empty_metrics


def test_calculate_metrics_logging(sample_data, mocker: MockerFixture):
    """Testet, ob die Logging-Funktion korrekt aufgerufen wird."""
    mock_print = mocker.patch("builtins.print")
    retrieved_chunks, gold_passages = sample_data
    k = 4
    question = "Test Frage?"

    calculate_metrics(retrieved_chunks, gold_passages, k, question=question)

    # Überprüfen, ob print aufgerufen wurde
    assert mock_print.call_count > 0

    # Überprüfen, ob die Schlüsselausdrücke geloggt wurden
    call_args_list = [call.args[0] for call in mock_print.call_args_list]
    log_output = "\n".join(call_args_list)

    assert "Test Frage?" in log_output
    assert "MATCH FOUND!" in log_output
    assert "Rank 1" in log_output
    assert "Gold Eins" in log_output
    assert "Rank 3" in log_output
    assert "Gold Zwei" in log_output


def test_calculate_metrics_logging_no_match(mocker: MockerFixture):
    """Testet das Logging, wenn kein Treffer gefunden wurde."""
    mock_print = mocker.patch("builtins.print")
    retrieved_chunks = ["Falsch 1", "Falsch 2"]
    gold_passages = ["Gold"]
    k = 2
    question = "Test?"

    calculate_metrics(retrieved_chunks, gold_passages, k, question=question)

    call_args_list = [call.args[0] for call in mock_print.call_args_list]
    log_output = "\n".join(call_args_list)

    assert "Test?" in log_output
    assert "NO MATCHES FOUND" in log_output
