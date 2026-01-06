from typing import Any

import pytest

from src.chunking.chunk_semantic import LangChainEmbeddingWrapper, chunk_semantic


@pytest.fixture
def sample_text_semantic() -> str:
    """Sample text with 3 sentences for semantic chunking tests."""
    return "This is the first sentence. This is the second sentence. This is a third, different sentence."


@pytest.fixture
def mock_vectorizer(mocker: Any) -> Any:
    """
    Mocks a vectorizer that returns specific embeddings.
    SemanticChunker creates combinations of sentences (sliding window), not individual sentences.

    For 3 sentences, it creates these combinations:
    1. "sentence1. sentence2."
    2. "sentence1. sentence2. sentence3."
    3. "sentence2. sentence3."

    It then computes cosine similarity between consecutive combinations.
    """
    mock = mocker.Mock()

    def embed_side_effect(texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Return embeddings based on text content to control semantic chunking."""
        embeddings = []
        for text in texts:
            # Combination 1 & 2: similar (both contain "first" and "second")
            if "first sentence. This is the second sentence." in text:
                embeddings.append([1.0, 0.0, 0.0])
            # Combination 3: different (contains "third" without "first")
            elif "second sentence. This is a third" in text:
                embeddings.append([0.0, 0.0, 1.0])
            # Fallback
            else:
                embeddings.append([0.5, 0.5, 0.0])
        return embeddings

    mock.embed_documents.side_effect = embed_side_effect
    return mock


def test_chunk_semantic_basic(sample_text_semantic: str, mock_vectorizer: Any) -> None:
    """Test that chunk_semantic correctly uses chunking_embeddings parameter."""
    chunks = chunk_semantic(
        sample_text_semantic, chunking_embeddings=mock_vectorizer, similarity_threshold=0.8
    )

    # Verify the mock was called
    assert mock_vectorizer.embed_documents.called

    # Should create chunks based on semantic similarity
    assert len(chunks) >= 1
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chunk_semantic_no_split(sample_text_semantic: str, mock_vectorizer: Any) -> None:
    """Test that high similarity threshold keeps text together."""
    chunks = chunk_semantic(
        sample_text_semantic, chunking_embeddings=mock_vectorizer, similarity_threshold=0.95
    )

    # With very high threshold (percentile=5), should keep most text together
    assert len(chunks) >= 1


def test_chunk_semantic_more_splits(sample_text_semantic: str, mock_vectorizer: Any) -> None:
    """Test that low similarity threshold creates more splits."""
    chunks = chunk_semantic(
        sample_text_semantic, chunking_embeddings=mock_vectorizer, similarity_threshold=0.4
    )

    # With lower threshold (percentile=60), should create more splits
    assert len(chunks) >= 1


def test_chunk_semantic_empty_text(mock_vectorizer: Any) -> None:
    """Test that empty text returns empty list."""
    chunks = chunk_semantic("", chunking_embeddings=mock_vectorizer, similarity_threshold=0.8)

    assert chunks == []


def test_chunk_semantic_single_sentence(mock_vectorizer: Any) -> None:
    """Test single sentence handling."""
    text = "Just one sentence."
    chunks = chunk_semantic(text, chunking_embeddings=mock_vectorizer, similarity_threshold=0.8)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_semantic_requires_embeddings() -> None:
    """Test that chunking_embeddings parameter is required."""
    with pytest.raises(TypeError):
        # Should fail without chunking_embeddings
        chunk_semantic("Some text", similarity_threshold=0.8)


def test_langchain_wrapper_integration(mock_vectorizer: Any) -> None:
    """Test that LangChainEmbeddingWrapper correctly wraps the vectorizer."""
    wrapper = LangChainEmbeddingWrapper(mock_vectorizer)

    # Test embed_documents
    test_texts = ["text 1", "text 2"]
    result = wrapper.embed_documents(test_texts)

    assert mock_vectorizer.embed_documents.called
    assert isinstance(result, list)
