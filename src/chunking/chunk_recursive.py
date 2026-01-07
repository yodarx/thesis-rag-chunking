from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name="gpt-3.5-turbo",
        separators=["\n\n", "\n", " ", ""]
    )

    docs = text_splitter.create_documents([text])
    return [doc.page_content for doc in docs]