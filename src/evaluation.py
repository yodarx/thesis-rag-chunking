import re


def _normalize_text(text: str) -> str:
    """Bereinigt Text für den Vergleich (Kleinbuchstaben, keine Satzzeichen)."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def calculate_metrics(retrieved_chunks: list[str], gold_passages: list[str], k: int, question: str = ""):
    """
    Berechnet die Metriken Precision@k, Recall und MRR.
    """
    if not gold_passages:
        return {"precision_at_k": 0, "recall": 0, "mrr": 0}

    norm_retrieved_chunks = [_normalize_text(chunk) for chunk in retrieved_chunks]
    norm_gold_passages = [_normalize_text(passage) for passage in gold_passages]

    found_passages = set()
    first_hit_rank = 0

    for rank, chunk in enumerate(norm_retrieved_chunks, 1):
        for i, passage in enumerate(norm_gold_passages):
            if passage in chunk:
                found_passages.add(passage)
                if first_hit_rank == 0:
                    first_hit_rank = rank

                # Print match information
                print(f"\n{'='*60}")
                print(f"MATCH FOUND!")
                print(f"Question: {question}")
                print(f"Matched Chunk: {retrieved_chunks[rank-1]}")
                print(f"Gold Standard: {gold_passages[i]}")
                print(f"Rank: {rank}")
                print(f"{'='*60}")

    mrr = 1 / first_hit_rank if first_hit_rank > 0 else 0.0

    hits_at_k = 0
    for chunk in norm_retrieved_chunks[:k]:
        if any(passage in chunk for passage in norm_gold_passages):
            hits_at_k += 1
    precision_at_k = hits_at_k / k if k > 0 else 0.0

    recall = len(found_passages) / len(norm_gold_passages)

    return {
        "precision_at_k": precision_at_k,
        "recall": recall,
        "mrr": mrr
    }