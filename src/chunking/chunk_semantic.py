from typing import Protocol

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    raise ImportError(
        "NLTK not found. Please run 'pip install nltk' and download 'punkt' model."
    ) from None


class EmbeddingVectorizer(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def _calculate_adjacent_similarities(
    embeddings: np.ndarray,
) -> list[float]:
    similarities: list[float] = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i + 1].reshape(1, -1))[0][
            0
        ]
        similarities.append(sim)
    return similarities


def chunk_semantic(
    text: str,
    *,
    vectorizer: EmbeddingVectorizer,
    similarity_threshold: float = 0.8,
) -> list[str]:
    """
    Splits text based on the semantic similarity of adjacent sentences.
    """
    if not text:
        return []

    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return sentences

    sentence_embeddings = np.array(vectorizer.embed_documents(sentences))
    if sentence_embeddings.shape[0] != len(sentences):
        raise ValueError("Vectorizer returned a different number of embeddings than sentences.")

    similarities = _calculate_adjacent_similarities(sentence_embeddings)

    chunks: list[str] = []
    current_chunk_sentences: list[str] = [sentences[0]]

    for i, similarity in enumerate(similarities):
        if similarity < similarity_threshold:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []

        current_chunk_sentences.append(sentences[i + 1])

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks
