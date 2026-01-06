from typing import Protocol

from langchain_experimental.text_splitter import SemanticChunker


class EmbeddingVectorizer(Protocol):
    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LangChainEmbeddingWrapper:
    """Wrapper to make our vectorizer compatible with LangChain's SemanticChunker."""

    def __init__(self, vectorizer: EmbeddingVectorizer) -> None:
        self.vectorizer = vectorizer

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.embed_documents(texts, batch_size=32)

    def embed_query(self, text: str) -> list[float]:
        return self.vectorizer.embed_query(text)


def chunk_semantic(
        text: str,
        *,
        chunking_embeddings: EmbeddingVectorizer | str,
        similarity_threshold: float = 0.8,
) -> list[str]:
    """
    Splits text based on the semantic similarity of adjacent sentences using LangChain's SemanticChunker.

    Args:
        text: The text to chunk
        chunking_embeddings: Embedding model used for determining semantic chunk boundaries.
                            Can be either:
                            - An EmbeddingVectorizer instance (for testing)
                            - A string model name (loaded by ExperimentRunner)
                            This is separate from retrieval embeddings to maintain scientific validity.
        similarity_threshold: Threshold for semantic similarity (0-1). Higher values create fewer, larger chunks.

    Note:
        For scientific experiments, the chunking embeddings should be specified separately from
        retrieval embeddings to avoid confounding variables. When chunking_embeddings is a string,
        it will be resolved to a Vectorizer instance by the ExperimentRunner.
    """
    if not text:
        return []

    # If chunking_embeddings is a string, it should have been converted to a Vectorizer by now
    if isinstance(chunking_embeddings, str):
        raise ValueError(
            f"chunking_embeddings must be an EmbeddingVectorizer instance, not a string. "
            f"Got: {chunking_embeddings}. This should be resolved by ExperimentRunner."
        )

    # Wrap the vectorizer to make it compatible with LangChain
    wrapped_embeddings = LangChainEmbeddingWrapper(chunking_embeddings)

    # Create the semantic chunker with percentile-based breakpoint detection
    # Convert similarity_threshold to percentile (higher threshold = lower percentile)
    # similarity_threshold of 0.8 means we want to split when similarity < 0.8
    # This roughly corresponds to a percentile threshold
    breakpoint_percentile = int((1 - similarity_threshold) * 100)

    text_splitter = SemanticChunker(
        embeddings=wrapped_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_percentile,
    )

    chunks = text_splitter.create_documents([text])
    return [doc.page_content for doc in chunks]
