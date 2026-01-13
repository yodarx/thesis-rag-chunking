import json

import faiss
import numpy as np

from src.vectorizer.vectorizer import Vectorizer


class FaissRetriever:
    def __init__(self, vectorizer: Vectorizer) -> None:
        self.vectorizer: Vectorizer = vectorizer
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[str] = []

    def build_index(self, chunks: list[str]) -> None:
        embeddings: np.ndarray = np.array(self.vectorizer.embed_documents(chunks)).astype("float32")
        dimension: int = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.chunks = chunks

    def load_index(self, index_path: str, chunks_path: str) -> None:
        self.index = faiss.read_index(index_path)
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, top_k: int) -> list[str]:
        if self.index is None:
            raise RuntimeError(
                "Index is not built or loaded. Call build_index() or load_index() first."
            )
        query_embedding: np.ndarray = np.array(self.vectorizer.embed_documents([query])).astype(
            "float16"
        )
        k_for_search: int = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k_for_search)
        return [self.chunks[i] for i in indices[0]]
