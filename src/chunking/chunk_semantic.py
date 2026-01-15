from typing import Protocol

from langchain_experimental.text_splitter import SemanticChunker


class EmbeddingVectorizer(Protocol):
    def embed_documents(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LangChainEmbeddingWrapper:
    """Wrapper to make our vectorizer compatible with LangChain's SemanticChunker."""

    def __init__(self, vectorizer: EmbeddingVectorizer) -> None:
        self.vectorizer = vectorizer

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.embed_documents(texts, batch_size=2048)

    def embed_query(self, text: str) -> list[float]:
        return self.vectorizer.embed_query(text)


def chunk_semantic(
        text: str | list[str],
        *,
        chunking_embeddings: EmbeddingVectorizer | str,
        similarity_threshold: float = 0.8,
) -> list[str]:
    """
    Splits text based on the semantic similarity of adjacent sentences using LangChain's SemanticChunker.

    Supports batch processing by accepting a list of strings.

    Args:
        text: The text to chunk (str) or a list of texts to chunk (list[str]).
        chunking_embeddings: Embedding model used for determining semantic chunk boundaries.
                            Can be either:
                            - An EmbeddingVectorizer instance (for testing)
                            - A string model name (loaded by ExperimentRunner)
        similarity_threshold: Threshold for semantic similarity (0-1). Higher values create fewer, larger chunks.
    """
    if not text:
        return []

    # 1. Normalize input to a list of strings
    texts = [text] if isinstance(text, str) else text

    # Filter out empty strings to prevent errors in embeddings
    texts = [t for t in texts if t and t.strip()]

    if not texts:
        return []

    if isinstance(chunking_embeddings, str):
        raise ValueError(
            f"chunking_embeddings must be an EmbeddingVectorizer instance, not a string. "
            f"Got: {chunking_embeddings}. This should be resolved by ExperimentRunner."
        )

    wrapped_embeddings = LangChainEmbeddingWrapper(chunking_embeddings)

    breakpoint_percentile = int((1 - similarity_threshold) * 100)

    text_splitter = SemanticChunker(
        embeddings=wrapped_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_percentile,
    )

    chunks = text_splitter.create_documents(texts)

    return [doc.page_content for doc in chunks]
