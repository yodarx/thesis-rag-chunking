import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Vectorizer:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    @classmethod
    def from_model_name(cls, model_name: str = "all-MiniLM-L6-v2") -> "Vectorizer":
        # 1. Force Device Check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{cls.__name__}] Initializing SentenceTransformer on: {device.upper()}")

        loaded_model = SentenceTransformer(model_name, device=device)

        # 2. FP16 Optimization (Safe)
        if device == "cuda":
            print(f"[{cls.__name__}] Enabling FP16 (Half Precision)...")
            loaded_model.half()

        # 3. COMPILATION IS REMOVED COMPLETELY
        # No 'torch.compile' calls here at all.
        # This guarantees standard eager execution (Instant start, consistent speed).

        return cls(loaded_model)

    @staticmethod
    def get_device() -> str | None:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return None

    def embed_documents(
            self,
            documents: list[str],
            batch_size: int = 2048,
            convert_to_numpy: bool = False
    ) -> list[list[float]] | np.ndarray:

        # --- DEBUG LOGGING (Remove later) ---
        # This proves if the GPU is actually receiving work
        # print(f"  -> Vectorizer: Encoding {len(documents)} sentences (Batch size: {batch_size})...")

        if not documents:
            return np.empty((0, 0), dtype=np.float32) if convert_to_numpy else []

        # 4. Critical: Ensure we don't accidentally re-compile or use CPU
        embeddings = self.model.encode(
            documents,
            show_progress_bar=False,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 5. Fast Return
        if convert_to_numpy:
            return embeddings.astype("float32")

        return embeddings.tolist()
