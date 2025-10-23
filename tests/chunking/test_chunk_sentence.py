import pytest

from chunking.chunk_sentence import chunk_by_sentence


@pytest.fixture
def sample_text_sentences() -> str:
    return "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."


def test_chunk_by_sentence_basic(sample_text_sentences: str):
    chunks = chunk_by_sentence(sample_text_sentences, sentences_per_chunk=2)
    assert len(chunks) == 3
    assert chunks[0] == "Sentence one. Sentence two."
    assert chunks[1] == "Sentence three. Sentence four."
    assert chunks[2] == "Sentence five."


def test_chunk_by_sentence_single_chunk(sample_text_sentences: str):
    chunks = chunk_by_sentence(sample_text_sentences, sentences_per_chunk=10)
    assert len(chunks) == 1
    assert chunks[0] == sample_text_sentences


def test_chunk_by_sentence_empty_text():
    chunks = chunk_by_sentence("", sentences_per_chunk=2)
    assert chunks == []


def test_chunk_by_sentence_invalid_chunk_size():
    with pytest.raises(ValueError, match="must be greater than 0"):
        chunk_by_sentence("Test text.", sentences_per_chunk=0)
