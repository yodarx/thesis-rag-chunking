import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Vectorizer:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model: SentenceTransformer = model

    @classmethod
    def from_model_name(cls, model_name: str = "all-MiniLM-L6-v2") -> "Vectorizer":
        device: str | None = cls._get_device()
        print(f"Selected device for SentenceTransformer: {device if device else 'cpu'}")

        loaded_model: SentenceTransformer = SentenceTransformer(model_name, device=device)
        loaded_model.half()

        if hasattr(torch, "compile"):
            try:
                print("Compiling transformer backbone with torch.compile...")
                transformer_module = loaded_model._first_module().auto_model

                loaded_model._first_module().auto_model = torch.compile(transformer_module)
            except Exception as e:
                print(f"Compilation failed (running eager mode): {e}")

        return cls(loaded_model)

    @staticmethod
    def _get_device() -> str | None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return None

    def embed_documents(self, documents: list[str], batch_size: int) -> np.ndarray:
        """
        Wandelt eine Liste von Textdokumenten in Vektor-Embeddings um.
        Gibt direkt ein Numpy-Array zurück (Float32).
        """
        if not documents:
            return np.empty((0, 0), dtype=np.float32)

        embeddings_array = self.model.encode(
            documents,
            show_progress_bar=False,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings_array
