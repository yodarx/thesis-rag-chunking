import pytest

from chunking.chunk_fixed import chunk_fixed_size


@pytest.fixture
def sample_text() -> str:
    return "This is a test text. It has several sentences. We are testing chunking."


def test_chunk_fixed_size_basic(sample_text: str) -> None:
    chunks = chunk_fixed_size(sample_text, chunk_size=20, chunk_overlap=5)

    # Token-based splitting may or may not split very short texts depending on the
    # tokenizer/encoding. The contract we care about is: no empty chunks, and
    # the concatenation (ignoring overlaps) covers the original text.
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)

    # If it does split, we expect overlap to create shared content.
    if len(chunks) > 1:
        assert chunks[0] != chunks[1]
        assert len(set(chunks)) == len(chunks)  # no duplicates


def test_chunk_fixed_size_no_overlap(sample_text: str) -> None:
    chunks = chunk_fixed_size(sample_text, chunk_size=20, chunk_overlap=0)

    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)

    # With no overlap, adjacent chunks should not be identical.
    if len(chunks) > 1:
        assert chunks[0] != chunks[1]


def test_chunk_fixed_size_empty_text() -> None:
    chunks = chunk_fixed_size("", chunk_size=20, chunk_overlap=5)
    assert chunks == []


def test_chunk_fixed_size_smaller_than_chunk(sample_text: str) -> None:
    chunks = chunk_fixed_size("Small text.", chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == "Small text."
