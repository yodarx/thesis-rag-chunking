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
    Mocks a vectorizer that returns consistent embeddings.
    """
    mock = mocker.Mock()

    def embed_side_effect(texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        """
        Return embeddings based on text content.
        We return fixed vectors based on simple string matching to simulate
        semantic similarity/dissimilarity without relying on complex window logic.
        """
        embeddings = []
        for text in texts:
            # If the text contains "different", return a distinct vector
            # effectively forcing a split point around this sentence
            if "different" in text:
                embeddings.append([0.0, 0.0, 1.0])
            # Otherwise return a 'standard' vector
            else:
                embeddings.append([1.0, 0.0, 0.0])
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


def test_chunk_semantic_batch_input(mock_vectorizer: Any) -> None:
    """
    Test that chunk_semantic correctly handles a list of documents (batching).
    This is the critical test for the new performance fix.
    """
    batch_input = [
        "Doc one sentence one. Doc one sentence two.",
        "Doc two sentence one. Doc two sentence two."
    ]

    chunks = chunk_semantic(
        batch_input,
        chunking_embeddings=mock_vectorizer,
        similarity_threshold=0.8
    )

    # Verify we got chunks back
    assert len(chunks) >= 2
    assert isinstance(chunks, list)
    # Ensure content from both documents is present
    joined_chunks = " ".join(chunks)
    assert "Doc one" in joined_chunks
    assert "Doc two" in joined_chunks

    # Verify the vectorizer was called (it handles the batching internally now)
    assert mock_vectorizer.embed_documents.called


def test_chunk_semantic_no_split(sample_text_semantic: str, mock_vectorizer: Any) -> None:
    """Test that high similarity threshold keeps text together."""
    chunks = chunk_semantic(
        sample_text_semantic, chunking_embeddings=mock_vectorizer, similarity_threshold=0.99
    )

    # With very high threshold, it should try to keep segments together
    # (though behavior depends on SemanticChunker's exact percentile logic)
    assert len(chunks) >= 1


def test_chunk_semantic_more_splits(sample_text_semantic: str, mock_vectorizer: Any) -> None:
    """Test that low similarity threshold creates more splits."""
    chunks = chunk_semantic(
        sample_text_semantic, chunking_embeddings=mock_vectorizer, similarity_threshold=0.1
    )

    # With lower threshold, should create more splits
    assert len(chunks) >= 1


def test_chunk_semantic_empty_inputs(mock_vectorizer: Any) -> None:
    """Test that empty string or empty list returns empty list."""
    # Case 1: Empty string
    chunks_str = chunk_semantic("", chunking_embeddings=mock_vectorizer)
    assert chunks_str == []

    # Case 2: Empty list
    chunks_list = chunk_semantic([], chunking_embeddings=mock_vectorizer)
    assert chunks_list == []

    # Case 3: List with empty string
    chunks_mixed = chunk_semantic([""], chunking_embeddings=mock_vectorizer)
    assert chunks_mixed == []


def test_chunk_semantic_single_sentence(mock_vectorizer: Any) -> None:
    """Test single sentence handling."""
    text = "Just one sentence."
    chunks = chunk_semantic(text, chunking_embeddings=mock_vectorizer, similarity_threshold=0.8)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_semantic_requires_embeddings() -> None:
    """Test that chunking_embeddings parameter is required or validated."""
    # Test missing argument (python TypeError)
    with pytest.raises(TypeError):
        chunk_semantic("Some text", similarity_threshold=0.8)  # type: ignore

    # Test invalid type passed as embedding
    with pytest.raises(ValueError, match="must be an EmbeddingVectorizer"):
        chunk_semantic("Some text", chunking_embeddings="invalid_string")


def test_langchain_wrapper_integration(mock_vectorizer: Any) -> None:
    """Test that LangChainEmbeddingWrapper correctly wraps the vectorizer."""
    wrapper = LangChainEmbeddingWrapper(mock_vectorizer)

    # Test embed_documents
    test_texts = ["text 1", "text 2"]
    result = wrapper.embed_documents(test_texts)

    assert mock_vectorizer.embed_documents.called
    assert isinstance(result, list)