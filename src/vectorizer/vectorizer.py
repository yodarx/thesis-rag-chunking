import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Vectorizer:
    def __init__(self, model: SentenceTransformer) -> None:
        """
        Initialisiert den Vectorizer mit einem bereits geladenen
        SentenceTransformer-Modell.
        """
        self.model: SentenceTransformer = model

    @classmethod
    def from_model_name(cls, model_name: str = "all-MiniLM-L6-v2") -> "Vectorizer":
        """
        Erstellt eine neue Vectorizer-Instanz durch Laden eines Modells
        anhand seines Namens. Wählt automatisch GPU, falls verfügbar, sonst CPU.
        """
        device: str | None = cls._get_device()
        print(f"Selected device for SentenceTransformer: {device if device else 'cpu'}")
        loaded_model: SentenceTransformer = SentenceTransformer(model_name, device=device)
        return cls(loaded_model)

    @staticmethod
    def _get_device() -> str | None:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return None

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Wandelt eine Liste von Textdokumenten in Vektor-Embeddings um.
        """
        if not documents:
            return []

        embeddings_array: np.ndarray = self.model.encode(documents, show_progress_bar=False)

        return embeddings_array.tolist()
