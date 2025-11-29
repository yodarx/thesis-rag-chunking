import json

import faiss
import numpy as np

from src.vectorizer.vectorizer import Vectorizer


class FaissRetriever:
    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer
        self.index = None
        self.chunks = []

    def build_index(self, chunks: list[str]):
        """Builds the FAISS index from a list of text chunks."""
        print(f"Creating {len(chunks)} embeddings for the index...")
        embeddings = self.vectorizer.embed_documents(chunks)
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.chunks = chunks
        print("FAISS index built successfully.")

    def load_index(self, index_path: str, chunks_path: str):
        """Loads a pre-built FAISS index and corresponding chunks."""
        print(f"Loading FAISS index from '{index_path}'...")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"Index and {len(self.chunks)} chunks loaded successfully.")

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Retrieves the top_k most relevant chunks for a given query."""
        if self.index is None:
            raise RuntimeError(
                "Index is not built or loaded. Call build_index() or load_index() first."
            )

        query_embedding = self.vectorizer.embed_documents([query])
        query_embedding = np.array(query_embedding).astype("float32")

        k_for_search = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, k_for_search)

        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        return retrieved_chunks
