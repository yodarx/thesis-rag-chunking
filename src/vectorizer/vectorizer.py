import numpy as np
from sentence_transformers import SentenceTransformer


class Vectorizer:
    def __init__(self, model: SentenceTransformer):
        """
        Initialisiert den Vectorizer mit einem bereits geladenen
        SentenceTransformer-Modell.
        """
        self.model = model

    @classmethod
    def from_model_name(cls, model_name: str = "all-MiniLM-L6-v2") -> "Vectorizer":
        """
        Erstellt eine neue Vectorizer-Instanz durch Laden eines Modells
        anhand seines Namens.
        """
        loaded_model = SentenceTransformer(model_name)
        return cls(loaded_model)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Wandelt eine Liste von Textdokumenten in Vektor-Embeddings um.
        """
        if not documents:
            return []

        embeddings_array: np.ndarray = self.model.encode(documents, show_progress_bar=False)

        return embeddings_array.tolist()
