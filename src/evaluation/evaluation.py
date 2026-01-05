import re
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
        # print(f"Question: {question}")
        # print(f"Retrieved Chunks: {chunks_at_k}")
        # print(f"Gold Passages: {gold_passages}")
        found_match = False
        for idx, res in enumerate(match_results):
            if res.is_relevant:
                # print(f"Match at position {idx + 1}: {res.matching_gold}")
                found_match = True
        if not found_match:
            print("NO MATCHES FOUND")

    return {
        "mrr": calculate_mrr(relevance_scores),
        "map": calculate_map(relevance_scores),
        "ndcg_at_k": calculate_ndcg_at_k(relevance_scores, total_relevant_gold),
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


def calculate_ndcg_at_k(relevance_scores: list[int], total_gold_items: int) -> float:
    """
    Berechnet den Normalized Discounted Cumulative Gain at k.
    Korrektur: IDCG basiert auf dem Minimum aus k und verfügbaren Gold-Items.
    """
    if not relevance_scores:
        return 0.0

    k = len(relevance_scores)

    # 1. Berechne DCG (Was wir tatsächlich erreicht haben)
    dcg: float = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    # 2. Berechne IDCG (Was maximal möglich gewesen wäre)
    # Das Ideal ist: Wir hätten 'k' Treffer, ODER alle 'total_gold_items' (falls weniger als k vorhanden sind)
    max_possible_matches = min(total_gold_items, k)

    if max_possible_matches == 0:
        return 0.0

    # Das ideale Ranking sind lauter 1en an der Spitze: [1, 1, 1...]
    ideal_dcg: float = sum(1.0 / np.log2(idx + 2) for idx in range(max_possible_matches))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

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
    r"""
    Erstellt eine Liste von Match-Ergebnissen für die Relevanzprüfung mit Wortgrenzen.

    Nutzt Regex mit Wortgrenzen (\b) um zu verhindern, dass "win" in "winter" matched.
    Erlaubt flexible Whitespace-Matches (\s+) zwischen Wörtern, inkl. Newlines, Tabs und mehrfache Leerzeichen.
    """
    results = []
    for chunk in chunks_at_k:
        match = None
        for gold in gold_passages:
            # Escape special regex characters
            gold_lower = gold.lower()
            safe_gold = re.escape(gold_lower)
            # Replace escaped spaces with flexible whitespace pattern (\s+ matches any whitespace including newlines, tabs)
            flexible_gold = safe_gold.replace(r"\ ", r"\s+")
            # Build pattern with word boundaries
            pattern = r"(?<!\w)" + flexible_gold + r"(?!\w)"

            if re.search(pattern, chunk.lower(), re.DOTALL):
                match = gold
                break
        results.append(MatchResult(is_relevant=match is not None, matching_gold=match))
    return results
