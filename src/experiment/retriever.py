import json

import faiss
import numpy as np

from src.vectorizer.vectorizer import Vectorizer


class FaissRetriever:
    def __init__(self, vectorizer: Vectorizer) -> None:
        self.vectorizer: Vectorizer = vectorizer
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []

    def build_index(self, chunks: list[str]) -> None:
        # Hier sicherstellen, dass wir float32 für den Index-Bau nutzen
        embeddings: np.ndarray = self.vectorizer.embed_documents(chunks, batch_size=32).astype("float32")
        dimension: int = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.chunks = chunks

    def load_index(self, index_path: str, chunks_path: str) -> None:
        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)

        # Performance-Tipp: Bei 40M Chunks ist json.load langsam.
        # Aber da du 128GB RAM hast, ist es okay.
        print(f"Loading chunks from {chunks_path}...")
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        print("Index and chunks loaded.")

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Wrapper für Einzelanfragen (nutzt intern batching für Konsistenz)"""
        return self.retrieve_batch([query], top_k)[0]

    def retrieve_batch(self, queries: list[str], top_k: int) -> list[list[str]]:
        if self.index is None:
            raise RuntimeError("Index is not built or loaded.")

        # 1. Vektorisieren
        query_embeddings = self.vectorizer.embed_documents(
            queries,
            batch_size=len(queries),
            convert_to_numpy=True
        )

        # --- FIX: Sicherheits-Konvertierung falls Liste kommt ---
        if isinstance(query_embeddings, list):
            query_embeddings = np.array(query_embeddings)
        # -------------------------------------------------------

        # 2. Float32 Check
        if query_embeddings.dtype != "float32":
            query_embeddings = query_embeddings.astype("float32")

        # 3. FAISS Suche
        k_for_search: int = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embeddings, k_for_search)

        results: list[list[str]] = []
        for row_indices in indices:
            row_chunks = [self.chunks[i] for i in row_indices if i != -1]
            results.append(row_chunks)

        return results
