from typing import NamedTuple

import numpy as np

# --- Datenstrukturen ---


class MatchResult(NamedTuple):
    """Speichert das Ergebnis eines Relevanz-Checks."""

    is_relevant: bool
    matching_gold: str | None


# --- Haupt-Metrik-Funktion ---


def calculate_metrics(
    retrieved_chunks: list[str],
    gold_passages: list[str],
    k: int,
    question: str | None = None,
    log_matches: bool = False,
) -> dict[str, float]:
    """
    Berechnet Retrieval-Metriken (MRR, MAP, NDCG@k, P@k, R@k, F1@k)
    und loggt optional Treffer.
    """
    empty_metrics: dict[str, float] = {
        "mrr": 0.0,
        "map": 0.0,
        "ndcg_at_k": 0.0,
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "f1_score_at_k": 0.0,
    }
    if not retrieved_chunks or not gold_passages:
        return empty_metrics

    chunks_at_k: list[str] = retrieved_chunks[:k]
    match_results: list[MatchResult] = _get_match_results(chunks_at_k, gold_passages)

    relevance_scores: list[int] = [1 if res.is_relevant else 0 for res in match_results]
    total_relevant_gold: int = len(gold_passages)

    precision: float = calculate_precision_at_k(relevance_scores)
    recall: float = calculate_recall_at_k(relevance_scores, total_relevant_gold)

    # --- Logging logic ---
    if log_matches and question is not None:
        print(f"Question: {question}")
        print(f"Retrieved Chunks: {chunks_at_k}")
        print(f"Gold Passages: {gold_passages}")
        found_match = False
        for idx, res in enumerate(match_results):
            if res.is_relevant:
                print(f"Match at position {idx + 1}: {res.matching_gold}")
                found_match = True
        if not found_match:
            print("NO MATCHES FOUND")

    return {
        "mrr": calculate_mrr(relevance_scores),
        "map": calculate_map(relevance_scores),
        "ndcg_at_k": calculate_ndcg_at_k(relevance_scores),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_score_at_k": calculate_f1_score_at_k(precision, recall),
    }


# --- Metrik-Berechnungs-Helfer ---
def calculate_mrr(relevance_scores: list[int]) -> float:
    """Berechnet den Mean Reciprocal Rank für eine einzelne Abfrage."""
    for idx, rel in enumerate(relevance_scores, 1):
        if rel:
            return 1.0 / idx
    return 0.0


def calculate_map(relevance_scores: list[int]) -> float:
    """Berechnet die Mean Average Precision für eine einzelne Abfrage."""
    relevant: int = 0
    score: float = 0.0
    for idx, rel in enumerate(relevance_scores, 1):
        if rel:
            relevant += 1
            score += relevant / idx
    return score / relevant if relevant else 0.0


def calculate_ndcg_at_k(relevance_scores: list[int]) -> float:
    """Berechnet den Normalized Discounted Cumulative Gain at k."""
    dcg: float = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    ideal: float = sum(1.0 / np.log2(idx + 2) for idx in range(sum(relevance_scores)))
    return dcg / ideal if ideal > 0 else 0.0


def calculate_precision_at_k(relevance_scores: list[int]) -> float:
    """Berechnet die Precision für die abgerufenen k Ergebnisse."""
    return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0


def calculate_recall_at_k(relevance_scores: list[int], total_relevant_gold: int) -> float:
    """Berechnet den Recall für die abgerufenen k Ergebnisse."""
    return sum(relevance_scores) / total_relevant_gold if total_relevant_gold else 0.0


def calculate_f1_score_at_k(precision: float, recall: float) -> float:
    """Berechnet den F1-Score aus Precision und Recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _get_match_results(chunks_at_k: list[str], gold_passages: list[str]) -> list[MatchResult]:
    """Erstellt eine Liste von Match-Ergebnissen für die Relevanzprüfung (case-insensitive substring match)."""
    results = []
    for chunk in chunks_at_k:
        match = None
        for gold in gold_passages:
            if gold.lower() in chunk.lower():
                match = gold
                break
        results.append(MatchResult(is_relevant=match is not None, matching_gold=match))
    return results
