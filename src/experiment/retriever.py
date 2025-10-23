import faiss
import numpy as np

from src.vectorizer.vectorizer import Vectorizer


class FaissRetriever:
    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer

    def search(self, question: str, chunk_embeddings: np.ndarray, top_k: int) -> list[int]:
        """Führt eine FAISS-Suche für eine Frage durch."""
        if chunk_embeddings.shape[0] == 0:
            return []

        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings)

        question_embedding = self.vectorizer.embed_documents([question])
        question_embedding_np = np.array(question_embedding, dtype="float32")

        k_for_search = min(top_k, chunk_embeddings.shape[0])

        _, indices = index.search(question_embedding_np, k_for_search)
        return indices[0].tolist()
