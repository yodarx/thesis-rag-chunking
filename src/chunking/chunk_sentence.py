try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    raise ImportError(
        "NLTK not found. Please run 'pip install nltk' and download 'punkt' model."
    ) from None


def chunk_by_sentence(text: str, sentences_per_chunk: int) -> list[str]:
    if not text:
        return []

    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be greater than 0")

    sentences: list[str] = sent_tokenize(text)
    chunks: list[str] = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk: str = " ".join(sentences[i : i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks
