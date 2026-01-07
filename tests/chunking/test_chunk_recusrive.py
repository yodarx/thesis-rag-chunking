import pytest

from chunking.chunk_recursive import chunk_recursive


@pytest.fixture
def sample_text_recursive() -> str:
    return "This is a test text.\n\nIt has a paragraph.\n\nAnd another one."


def test_chunk_recursive_basic(sample_text_recursive: str) -> None:
    chunks = chunk_recursive(sample_text_recursive, chunk_size=20, chunk_overlap=5)

    # With token-based splitting, short texts may remain a single chunk depending
    # on tokenizer behavior. We primarily assert basic invariants.
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)

    # If a split happens, we expect multiple distinct chunks.
    if len(chunks) > 1:
        assert len(set(chunks)) == len(chunks)


def test_chunk_recursive_empty_text() -> None:
    chunks = chunk_recursive("", chunk_size=20, chunk_overlap=5)
    assert chunks == []


def test_chunk_recursive_no_separators() -> None:
    text = "A single long sentence without any special separators."
    chunks = chunk_recursive(text, chunk_size=10, chunk_overlap=0)

    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)

    # No overlap: if it splits, chunks should be different.
    if len(chunks) > 1:
        assert chunks[0] != chunks[1]
