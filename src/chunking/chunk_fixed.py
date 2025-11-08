from langchain_text_splitters import CharacterTextSplitter


def chunk_fixed_size(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([text])
    return [doc.page_content for doc in docs]
