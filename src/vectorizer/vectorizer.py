import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Vectorizer:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    @classmethod
    def from_model_name(cls, model_name: str = "all-MiniLM-L6-v2") -> "Vectorizer":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing {model_name} on {device.upper()}")

        model = SentenceTransformer(model_name, device=device)

        if device == "cuda":
            model.half()

        return cls(model)

    def embed_documents(
            self,
            texts: list[str],
            batch_size: int = 2048,
            convert_to_numpy: bool = True
    ) -> np.ndarray | list[list[float]]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32) if convert_to_numpy else []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if convert_to_numpy:
            return embeddings.astype("float32")

        return embeddings.tolist()