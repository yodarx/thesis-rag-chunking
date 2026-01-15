import re
import time
import numpy as np
from typing import List, Protocol, Literal
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document


# --- Interfaces ---
class EmbeddingVectorizer(Protocol):
    def embed_documents(self, texts: list[str], batch_size: int = 2048) -> np.ndarray: ...

    def embed_query(self, text: str) -> List[float]: ...


class LangChainAdapter:
    """Adapts our clean Vectorizer to LangChain's expected interface with Smart Batching."""

    def __init__(self, vectorizer: EmbeddingVectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.array([])

        # --- SMART BATCHING OPTIMIZATION ---
        # 1. Sort texts by length.
        #    This groups short items with short items, drastically reducing padding overhead.
        indices = np.argsort([len(t) for t in texts])
        sorted_texts = [texts[i] for i in indices]

        # 2. Embed the sorted list.
        #    We safely use batch_size=512. Since data is sorted, memory usage is efficient.
        embeddings = self.vectorizer.embed_documents(
            sorted_texts,
            batch_size=1024,
            convert_to_numpy=True
        )

        # 3. Un-sort (Restore original order).
        #    We need to put embeddings back in the order the chunker expects.
        inverse_indices = np.argsort(indices)
        return embeddings[inverse_indices]

    def embed_query(self, text: str) -> List[float]:
        return self.vectorizer.embed_query(text)


# --- Core Logic ---
class BatchSemanticChunker(SemanticChunker):
    def __init__(
            self,
            embeddings,
            threshold_type: Literal["percentile", "fixed"] = "fixed",
            threshold_amount: float = 0.5
    ):
        super().__init__(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # Parent requires valid enum
            breakpoint_threshold_amount=threshold_amount
        )
        self.threshold_type = threshold_type
        self.threshold_val = threshold_amount

    def create_documents(self, texts: List[str], metadatas: List[dict] = None) -> List[Document]:
        """
        Main entry point. Flattens docs -> embeds batch -> reconstructs docs.
        """
        if not texts: return []

        # [DEBUG] Start Timer
        t0 = time.time()

        sentences, doc_lengths = self._prepare_sentences(texts)
        if not sentences: return []

        # [DEBUG] Data Health Check
        # If 'Max Len' is huge (>5000), your regex is failing and GPU will choke.
        t1 = time.time()
        avg_len = sum(len(s) for s in sentences) / len(sentences)
        max_len = max(len(s) for s in sentences)
        print(
            f"[DEBUG] Batch: {len(texts)} Docs -> {len(sentences)} Sents | Avg Char: {avg_len:.0f} | Max Char: {max_len}")

        # GPU Operation
        embeddings = self.embeddings.embed_documents(sentences)

        # [DEBUG] GPU Timer
        t2 = time.time()

        # Clustering Logic
        docs = self._reconstruct_documents(sentences, embeddings, doc_lengths, metadatas)

        # [DEBUG] Phase Timing
        t3 = time.time()
        print(f"[PROFILE] Prep (CPU): {t1 - t0:.3f}s | Embed (GPU): {t2 - t1:.3f}s | Merge (CPU): {t3 - t2:.3f}s")

        return docs

    def _prepare_sentences(self, texts: List[str]) -> tuple[List[str], List[int]]:
        """Splits texts into sentences, safeguarding against massive blobs."""
        all_sentences = []
        doc_lengths = []

        for text in texts:
            sents = self._split_safe(text)
            all_sentences.extend(sents)
            doc_lengths.append(len(sents))

        return all_sentences, doc_lengths

    def _split_safe(self, text: str, max_chars: int = 1000) -> List[str]:
        """Splits on punctuation, but forcibly chops huge segments to protect GPU."""
        # Split on .?! followed by whitespace
        splits = re.split(r'(?<=[.?!])\s+', text)
        clean_splits = []

        for s in splits:
            s = s.strip()
            if not s: continue

            if len(s) > max_chars:
                # Hard chop O(N^2) killers
                clean_splits.extend([s[i:i + max_chars] for i in range(0, len(s), max_chars)])
            else:
                clean_splits.append(s)

        return clean_splits

    def _reconstruct_documents(
            self,
            sentences: List[str],
            embeddings: np.ndarray,
            counts: List[int],
            metadatas: List[dict]
    ) -> List[Document]:
        documents = []
        cursor = 0

        fixed_threshold = 1.0 - self.threshold_val

        for i, count in enumerate(counts):
            if count == 0: continue

            # Slice batch data for this document
            doc_sents = sentences[cursor: cursor + count]
            doc_embs = embeddings[cursor: cursor + count]
            cursor += count

            # Math
            threshold = self._calculate_threshold(doc_embs, fixed_threshold)
            distances = self._calculate_distances(doc_embs)

            # Build
            documents.extend(self._cluster_sentences(doc_sents, distances, threshold, metadatas, i))

        return documents

    def _calculate_distances(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 2: return np.array([])
        # Cosine Dist = 1 - Dot(NormA, NormB)
        sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        return 1.0 - sims

    def _calculate_threshold(self, embeddings: np.ndarray, fixed_val: float) -> float:
        if self.threshold_type == "fixed":
            return fixed_val

        distances = self._calculate_distances(embeddings)
        if len(distances) == 0: return 0.0

        return np.percentile(distances, self.threshold_val)

    def _cluster_sentences(
            self,
            sentences: List[str],
            distances: np.ndarray,
            threshold: float,
            metadatas: List[dict],
            meta_idx: int
    ) -> List[Document]:
        docs = []
        current_chunk = [sentences[0]]
        metadata = metadatas[meta_idx] if metadatas else {}

        for j, dist in enumerate(distances):
            if dist > threshold:
                # Split
                docs.append(Document(page_content=" ".join(current_chunk), metadata=metadata))
                current_chunk = [sentences[j + 1]]
            else:
                # Merge
                current_chunk.append(sentences[j + 1])

        if current_chunk:
            docs.append(Document(page_content=" ".join(current_chunk), metadata=metadata))

        return docs


# --- Main Entry Point ---
def chunk_semantic(
        text: str | list[str],
        *,
        chunking_embeddings: EmbeddingVectorizer,
        similarity_threshold: float = 0.8,
) -> list[str]:
    if not text: return []
    texts = [text] if isinstance(text, str) else text
    texts = [t for t in texts if t and t.strip()]

    adapter = LangChainAdapter(chunking_embeddings)
    splitter = BatchSemanticChunker(adapter, threshold_type="fixed", threshold_amount=similarity_threshold)

    docs = splitter.create_documents(texts)
    return [d.page_content for d in docs]