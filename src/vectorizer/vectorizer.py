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

        # 1. FP16 (Halbe Präzision) für weniger VRAM und mehr Speed
        # half() is only valid on some backends; keep it best-effort
        with contextlib.suppress(Exception):
            loaded_model.half()

        # 2. TORCH COMPILE (Turbo für Nvidia L4 / A100)
        # Das beschleunigt die Inference massiv, funktioniert aber nur mit neueren PyTorch Versionen
        if hasattr(torch, "compile") and device == "cuda":
            try:
                print("Compiling model with torch.compile for max speed...")
                loaded_model = torch.compile(loaded_model)
            except Exception as e:
                print(f"Could not compile model (continuing normally): {e}")

        return cls(loaded_model)

    @staticmethod
    def _get_device() -> str | None:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return None

    def embed_documents(
            self,
            documents: list[str],
            batch_size: int = 2048,
            convert_to_numpy: bool = False  # <--- HIER IST DER TRICK
    ) -> list[list[float]] | np.ndarray:
        """
        Convert a list of documents to embeddings.

        Args:
            documents: List of texts.
            batch_size: GPU batch size.
            convert_to_numpy: If True, returns fast np.ndarray (for Indexing).
                              If False (default), returns list (for Tests).
        """
        if not documents:
            # Return empty structure matching the requested type
            return np.empty((0, 0), dtype=np.float32) if convert_to_numpy else []

        # WICHTIG: Wir lassen SentenceTransformer IMMER Numpy erstellen (intern effizienter)
        # und wandeln es nur am Ende um, wenn nötig.
        embeddings = self.model.encode(
            documents,
            show_progress_bar=False,  # Progress Bar machen wir lieber extern (tqdm)
            batch_size=batch_size,
            convert_to_numpy=True,  # Immer True für interne Performance
            normalize_embeddings=True  # Optional: Oft besser für RAG
        )

        # Wenn der Aufrufer (Index Builder / Runner) Numpy will -> direkt zurückgeben
        if convert_to_numpy:
            return embeddings.astype("float32")  # Sicherstellen für FAISS

        # Wenn deine Tests Listen wollen (Standardverhalten) -> umwandeln
        return embeddings.tolist()