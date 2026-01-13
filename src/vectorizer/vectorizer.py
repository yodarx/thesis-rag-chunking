import contextlib

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Vectorizer:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model: SentenceTransformer = model

    @classmethod
    def from_model_name(cls, model_name: str = "all-MiniLM-L6-v2") -> "Vectorizer":
        device = cls._get_device()
        print(f"Selected device for SentenceTransformer: {device if device else 'cpu'}")

        loaded_model = SentenceTransformer(model_name, device=device)
        # half() is only valid on some backends; keep it best-effort to avoid test/machine issues
        with contextlib.suppress(Exception):
            loaded_model.half()

        return cls(loaded_model)

    @staticmethod
    def _get_device() -> str | None:
        if torch.cuda.is_available():
            return "cuda"
        # Prefer MPS on macOS when available
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        # Tests expect CPU to be represented as "None" for SentenceTransformer(device=...)
        return None

    def embed_documents(self, documents: list[str], batch_size: int = 2048) -> list[list[float]]:
        """Convert a list of documents to embeddings.

        Tests in this repo expect:
        - empty input => [] and encode() not called
        - encode(documents, show_progress_bar=True, batch_size=...)
        - return value as a Python list of lists
        """
        if not documents:
            return []

        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            batch_size=batch_size,
        )

        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()

        return list(embeddings)
