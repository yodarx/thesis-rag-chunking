import pytest
from pytest_mock import MockerFixture

from chunking.chunk_semantic import EmbeddingVectorizer, chunk_semantic


@pytest.fixture
def sample_text_semantic() -> str:
    return "This is the first sentence. This is the second sentence. This is a third, different sentence."


@pytest.fixture
def mock_vectorizer(mocker: MockerFixture) -> EmbeddingVectorizer:
    """
    Mocks a vectorizer that returns specific embeddings.
    - Sentences 1 and 2 will be VERY similar (> 0.8)
    - Sentences 2 and 3 will be VERY different (< 0.8)
    """
    mock = mocker.Mock(spec=EmbeddingVectorizer)

    # Mock embeddings
    embedding_1 = [1.0, 0.0, 0.0]  # First sentence
    embedding_2 = [0.9, 0.1, 0.0]  # Second sentence (very similar to 1)
    embedding_3 = [0.0, 0.0, 1.0]  # Third sentence (very different)

    mock.embed_documents.return_value = [embedding_1, embedding_2, embedding_3]
    return mock


def test_chunk_semantic_basic(sample_text_semantic: str, mock_vectorizer: EmbeddingVectorizer):
    # similarity(1,2) will be high (~0.99).
    # similarity(2,3) will be low (0.0).
    # Threshold is 0.8. A split should occur between sentence 2 and 3.

    chunks = chunk_semantic(
        sample_text_semantic,
        vectorizer=mock_vectorizer,
        similarity_threshold=0.8,
    )

    mock_vectorizer.embed_documents.assert_called_once_with(
        [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is a third, different sentence.",
        ]
    )

    assert len(chunks) == 2
    assert chunks[0] == "This is the first sentence. This is the second sentence."
    assert chunks[1] == "This is a third, different sentence."


def test_chunk_semantic_no_split(sample_text_semantic: str, mock_vectorizer: EmbeddingVectorizer):
    # Set threshold very low, so no split should occur
    chunks = chunk_semantic(
        sample_text_semantic,
        vectorizer=mock_vectorizer,
        similarity_threshold=0.0,  # similarity(2,3) is 0.0, so this threshold is met
    )
    assert len(chunks) == 1
    assert chunks[0] == sample_text_semantic


def test_chunk_semantic_all_splits(sample_text_semantic: str, mock_vectorizer: EmbeddingVectorizer):
    # Set threshold very high, so all sentences should be split
    chunks = chunk_semantic(
        sample_text_semantic,
        vectorizer=mock_vectorizer,
        similarity_threshold=0.999,  # sim(1,2) is ~0.994, sim(2,3) is 0.0
    )

    # Both similarities are below the threshold, so 2 splits occur, resulting in 3 chunks.
    assert len(chunks) == 3
    assert chunks[0] == "This is the first sentence."
    assert chunks[1] == "This is the second sentence."
    assert chunks[2] == "This is a third, different sentence."


def test_chunk_semantic_empty_text(mock_vectorizer: EmbeddingVectorizer):
    chunks = chunk_semantic("", vectorizer=mock_vectorizer)
    assert chunks == []


def test_chunk_semantic_single_sentence(mock_vectorizer: EmbeddingVectorizer):
    text = "Just one sentence."
    chunks = chunk_semantic(text, vectorizer=mock_vectorizer)
    assert chunks == ["Just one sentence."]
