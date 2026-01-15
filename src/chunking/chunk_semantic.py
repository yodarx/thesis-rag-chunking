import re
from typing import Literal, Protocol

import numpy as np
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker


class EmbeddingVectorizer(Protocol):
    def embed_documents(self, texts: list[str], batch_size: int = 2048) -> np.ndarray: ...

    def embed_query(self, text: str) -> list[float]: ...


class LangChainAdapter:
    def __init__(self, vectorizer: EmbeddingVectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.embed_documents(texts, batch_size=2048, convert_to_numpy=True)

    def embed_query(self, text: str) -> list[float]:
        return self.vectorizer.embed_query(text)


class BatchSemanticChunker(SemanticChunker):
    def __init__(
            self,
            embeddings,
            threshold_type: Literal["percentile", "fixed"] = "fixed",
            threshold_amount: float = 0.5
    ):
        super().__init__(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # Parent expects valid enum
            breakpoint_threshold_amount=threshold_amount
        )
        self.threshold_type = threshold_type
        self.threshold_val = threshold_amount

    def create_documents(self, texts: list[str], metadatas: list[dict] = None) -> list[Document]:
        if not texts: return []

        sentences, doc_lengths = self._prepare_sentences(texts)
        if not sentences: return []

        embeddings = self.embeddings.embed_documents(sentences)
        return self._reconstruct_documents(sentences, embeddings, doc_lengths, metadatas)

    def _prepare_sentences(self, texts: list[str]) -> tuple[list[str], list[int]]:
        all_sentences = []
        doc_lengths = []

        for text in texts:
            # Split by punctuation but enforce max length to prevent GPU stalls
            sents = self._split_safe(text)
            all_sentences.extend(sents)
            doc_lengths.append(len(sents))

        return all_sentences, doc_lengths

    def _split_safe(self, text: str, max_chars: int = 1000) -> list[str]:
        # Split on punctuation (.?!) followed by whitespace
        splits = re.split(r'(?<=[.?!])\s+', text)
        clean_splits = []

        for s in splits:
            s = s.strip()
            if not s: continue

            if len(s) > max_chars:
                # Hard chop massive strings to protect O(N^2) attention layers
                clean_splits.extend([s[i:i + max_chars] for i in range(0, len(s), max_chars)])
            else:
                clean_splits.append(s)

        return clean_splits

    def _reconstruct_documents(
            self,
            sentences: list[str],
            embeddings: np.ndarray,
            counts: list[int],
            metadatas: list[dict]
    ) -> list[Document]:
        documents = []
        cursor = 0

        fixed_threshold = 1.0 - self.threshold_val

        for i, count in enumerate(counts):
            if count == 0: continue

            doc_sents = sentences[cursor: cursor + count]
            doc_embs = embeddings[cursor: cursor + count]
            cursor += count

            threshold = self._calculate_threshold(doc_embs, fixed_threshold)
            distances = self._calculate_distances(doc_embs)

            documents.extend(self._cluster_sentences(doc_sents, distances, threshold, metadatas, i))

        return documents

    def _calculate_distances(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 2: return np.array([])
        # Cosine Distance = 1 - Dot Product (of normalized vectors)
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
            sentences: list[str],
            distances: np.ndarray,
            threshold: float,
            metadatas: list[dict],
            meta_idx: int
    ) -> list[Document]:
        docs = []
        current_chunk = [sentences[0]]
        metadata = metadatas[meta_idx] if metadatas else {}

        for j, dist in enumerate(distances):
            if dist > threshold:
                docs.append(Document(page_content=" ".join(current_chunk), metadata=metadata))
                current_chunk = [sentences[j + 1]]
            else:
                current_chunk.append(sentences[j + 1])

        if current_chunk:
            docs.append(Document(page_content=" ".join(current_chunk), metadata=metadata))

        return docs


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
