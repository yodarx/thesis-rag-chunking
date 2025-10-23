import pytest

from chunking.chunk_recursive import chunk_recursive


@pytest.fixture
def sample_text_recursive() -> str:
    return "This is a test text.\n\nIt has a paragraph.\n\nAnd another one."


def test_chunk_recursive_basic(sample_text_recursive: str):
    chunks = chunk_recursive(sample_text_recursive, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    assert chunks[0] == "This is a test text."
    assert chunks[1] == "It has a paragraph."


def test_chunk_recursive_empty_text():
    chunks = chunk_recursive("", chunk_size=20, chunk_overlap=5)
    assert chunks == []


def test_chunk_recursive_no_separators():
    text = "A single long sentence without any special separators."
    chunks = chunk_recursive(text, chunk_size=10, chunk_overlap=0)
    assert chunks[0] == "A single"
    assert chunks[1] == "long"
