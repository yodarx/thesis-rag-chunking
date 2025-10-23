from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([text])
    return [doc.page_content for doc in docs]
