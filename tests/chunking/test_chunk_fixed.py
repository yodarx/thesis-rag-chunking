import pytest

from chunking.chunk_fixed import chunk_fixed_size


@pytest.fixture
def sample_text() -> str:
    return "This is a test text. It has several sentences. We are testing chunking."


def test_chunk_fixed_size_basic(sample_text: str):
    chunks = chunk_fixed_size(sample_text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    assert chunks[0] == "This is a test text."
    assert chunks[1] == "text. It has several"


def test_chunk_fixed_size_no_overlap(sample_text: str):
    chunks = chunk_fixed_size(sample_text, chunk_size=20, chunk_overlap=0)
    assert chunks[0] == "This is a test text."
    assert chunks[1] == "It has several"  # No overlap


def test_chunk_fixed_size_empty_text():
    chunks = chunk_fixed_size("", chunk_size=20, chunk_overlap=5)
    assert chunks == []


def test_chunk_fixed_size_smaller_than_chunk(sample_text: str):
    chunks = chunk_fixed_size("Small text.", chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == "Small text."
