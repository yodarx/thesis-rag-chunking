from langchain_text_splitters import TokenTextSplitter


def chunk_fixed_size(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name="gpt-3.5-turbo",
        strip_whitespace=True
    )

    docs = text_splitter.create_documents([text])
    return [doc.page_content for doc in docs]