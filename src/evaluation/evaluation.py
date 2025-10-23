from typing import NamedTuple

import numpy as np

# --- Datenstrukturen ---


class MatchResult(NamedTuple):
    """Speichert das Ergebnis eines Relevanz-Checks."""

    is_relevant: bool
    matching_gold: str | None


# --- Haupt-Metrik-Funktion ---


def calculate_metrics(
    retrieved_chunks: list[str], gold_passages: list[str], k: int, question: str | None = None
) -> dict[str, float]:
    """
    Berechnet Retrieval-Metriken (MRR, MAP, NDCG@k, P@k, R@k, F1@k)
    und loggt optional Treffer.
    """
    empty_metrics = {
        "mrr": 0.0,
        "map": 0.0,
        "ndcg_at_k": 0.0,
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "f1_score_at_k": 0.0,
    }
    if not retrieved_chunks or not gold_passages:
        return empty_metrics

    chunks_at_k = retrieved_chunks[:k]
    match_results = _get_match_results(chunks_at_k, gold_passages)

    if question is not None:
        _log_matches(question, chunks_at_k, match_results)

    relevance_scores = [1 if res.is_relevant else 0 for res in match_results]
    total_relevant_gold = len(gold_passages)

    precision = calculate_precision_at_k(relevance_scores)
    recall = calculate_recall_at_k(relevance_scores, total_relevant_gold)
    f1 = calculate_f1_score_at_k(precision, recall)

    return {
        "mrr": calculate_mrr(relevance_scores),
        "map": calculate_map(relevance_scores),
        "ndcg_at_k": calculate_ndcg(relevance_scores, k),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_score_at_k": f1,
    }


# --- Metrik-Berechnungs-Helfer ---


def calculate_mrr(relevance_scores: list[int]) -> float:
    """Berechnet den Mean Reciprocal Rank für eine einzelne Abfrage."""
    for i, score in enumerate(relevance_scores):
        if score > 0:
            return 1.0 / (i + 1)
    return 0.0


def calculate_map(relevance_scores: list[int]) -> float:
    """Berechnet die Mean Average Precision für eine einzelne Abfrage."""
    if not any(relevance_scores):
        return 0.0

    precisions: list[float] = []
    relevant_count = 0
    for i, score in enumerate(relevance_scores):
        if score > 0:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            precisions.append(precision)

    return np.mean(precisions) if precisions else 0.0


def calculate_ndcg(relevance_scores: list[int], k: int) -> float:
    """Berechnet den Normalized Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, score in enumerate(relevance_scores[:k]):
        dcg += score / np.log2(i + 2)

    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, score in enumerate(ideal_scores[:k]):
        idcg += score / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_at_k(relevance_scores_at_k: list[int]) -> float:
    """Berechnet die Precision für die abgerufenen k Ergebnisse."""
    num_retrieved = len(relevance_scores_at_k)
    if num_retrieved == 0:
        return 0.0

    relevant_at_k = sum(relevance_scores_at_k)
    return relevant_at_k / num_retrieved


def calculate_recall_at_k(relevance_scores_at_k: list[int], total_relevant_gold: int) -> float:
    """Berechnet den Recall für die abgerufenen k Ergebnisse."""
    if total_relevant_gold == 0:
        return 0.0

    relevant_retrieved = sum(relevance_scores_at_k)
    return relevant_retrieved / total_relevant_gold


def calculate_f1_score_at_k(precision: float, recall: float) -> float:
    """Berechnet den F1-Score aus Precision und Recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# --- Logging- und Matching-Helfer ---


def _find_first_matching_gold(chunk: str, gold_passages: list[str]) -> str | None:
    """Findet die erste Gold-Passage, die im Chunk enthalten ist."""
    chunk_lower = chunk.lower()
    for gold in gold_passages:
        if gold.lower() in chunk_lower:
            return gold
    return None


def _get_match_results(chunks_at_k: list[str], gold_passages: list[str]) -> list[MatchResult]:
    """Erstellt eine Liste von Match-Ergebnissen für die Relevanzprüfung."""
    results: list[MatchResult] = []
    for chunk in chunks_at_k:
        match = _find_first_matching_gold(chunk, gold_passages)
        if match:
            results.append(MatchResult(is_relevant=True, matching_gold=match))
        else:
            results.append(MatchResult(is_relevant=False, matching_gold=None))
    return results


def _log_matches(question: str, chunks_at_k: list[str], match_results: list[MatchResult]) -> None:
    """Gibt gefundene Treffer in der Konsole aus."""
    print("=" * 60)
    print(f"Question: {question}")

    found_match = False
    for i, (chunk, result) in enumerate(zip(chunks_at_k, match_results, strict=False)):
        if result.is_relevant:
            found_match = True
            print("-" * 20)
            print(f"MATCH FOUND! (Rank {i + 1})")
            print(f"  Matched Chunk: {chunk}")
            print(f"  Gold Standard: {result.matching_gold}")
            print("-" * 20)

    if not found_match:
        print("  NO MATCHES FOUND in Top-K.")

    print("=" * 60)
