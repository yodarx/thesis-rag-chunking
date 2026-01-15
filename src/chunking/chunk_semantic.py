import re
import time
import numpy as np
from typing import List, Protocol, Literal, Union
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document


# --- Protocol Update: Allow returning np.ndarray directly ---
class EmbeddingVectorizer(Protocol):
    def embed_documents(
            self, texts: list[str], batch_size: int = 512, convert_to_numpy: bool = True
    ) -> Union[List[List[float]], np.ndarray]: ...

    def embed_query(self, text: str) -> List[float]: ...


class LangChainEmbeddingWrapper:
    def __init__(self, vectorizer: EmbeddingVectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.embed_documents(
            texts,
            batch_size=1024,  # Safe batch size for L4
            convert_to_numpy=True
        )

    def embed_query(self, text: str) -> List[float]:
        return self.vectorizer.embed_query(text)


class BatchSemanticChunker(SemanticChunker):
    def __init__(
            self,
            embeddings,
            breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile", "fixed"] = "fixed",
            breakpoint_threshold_amount: float = 0.5
    ):
        super().__init__(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type if breakpoint_threshold_type != "fixed" else "percentile",
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
        self.custom_threshold_type = breakpoint_threshold_type
        self.custom_threshold_amount = breakpoint_threshold_amount

    def create_documents(self, texts: List[str], metadatas: List[dict] = None) -> List[Document]:
        if not texts: return []

        # Performance Timer
        t0 = time.time()

        # 1. ROBUST SENTENCE SPLITTING (CPU)
        # Splits on punctuation followed by a Capital letter.
        sentence_splitter = re.compile(r'(?<=[.?!])\s+(?=[A-Z])')

        all_sentences = []
        doc_sentence_counts = []

        for text in texts:
            sents = [s for s in sentence_splitter.split(text) if s.strip()]
            if not sents:
                doc_sentence_counts.append(0)
                continue
            all_sentences.extend(sents)
            doc_sentence_counts.append(len(sents))

        if not all_sentences: return []

        t1 = time.time()

        # 2. BATCH EMBEDDING (GPU)
        # Direct to NumPy, no list overhead.
        embeddings = self.embeddings.embed_documents(all_sentences)

        # Ensure it is an array (zero-copy if already array)
        np_embeddings = np.asarray(embeddings)

        t2 = time.time()

        # 3. RECONSTRUCTION (CPU)
        documents = []
        cursor = 0

        # Pre-calculate common threshold if fixed
        fixed_threshold = 0.0
        if self.custom_threshold_type == "fixed":
            fixed_threshold = 1.0 - self.custom_threshold_amount

        for i, count in enumerate(doc_sentence_counts):
            if count == 0: continue

            doc_sents = all_sentences[cursor: cursor + count]
            doc_embs = np_embeddings[cursor: cursor + count]
            cursor += count

            # --- Logic ---
            dists = []
            if len(doc_embs) > 1:
                # Fast Vectorized Cosine Distance
                sims = np.sum(doc_embs[:-1] * doc_embs[1:], axis=1)
                dists = 1.0 - sims

            threshold = fixed_threshold
            if self.custom_threshold_type != "fixed" and len(dists) > 0:
                if self.breakpoint_threshold_type == "percentile":
                    threshold = np.percentile(dists, self.breakpoint_threshold_amount)
                elif self.breakpoint_threshold_type == "standard_deviation":
                    threshold = np.mean(dists) + self.breakpoint_threshold_amount * np.std(dists)
                elif self.breakpoint_threshold_type == "interquartile":
                    q1, q3 = np.percentile(dists, [25, 75])
                    threshold = q3 + 1.5 * (q3 - q1)

            current_chunk = [doc_sents[0]]
            for j, distance in enumerate(dists):
                if distance > threshold:
                    content = " ".join(current_chunk)
                    documents.append(Document(page_content=content, metadata=metadatas[i] if metadatas else {}))
                    current_chunk = [doc_sents[j + 1]]
                else:
                    current_chunk.append(doc_sents[j + 1])

            if current_chunk:
                content = " ".join(current_chunk)
                documents.append(Document(page_content=content, metadata=metadatas[i] if metadatas else {}))

        t3 = time.time()

        # Debug Print to isolate bottleneck (Uncomment if still slow)
        print(f"Split: {t1-t0:.3f}s | Embed: {t2-t1:.3f}s | Merge: {t3-t2:.3f}s | Docs: {len(texts)}")

        return documents


def chunk_semantic(
        text: str | list[str],
        *,
        chunking_embeddings: EmbeddingVectorizer | str,
        similarity_threshold: float = 0.8,
) -> list[str]:
    if not text: return []
    texts = [text] if isinstance(text, str) else text
    texts = [t for t in texts if t and t.strip()]
    if not texts: return []

    if isinstance(chunking_embeddings, str):
        raise ValueError("chunking_embeddings must be a Vectorizer instance.")

    wrapped_embeddings = LangChainEmbeddingWrapper(chunking_embeddings)

    text_splitter = BatchSemanticChunker(
        embeddings=wrapped_embeddings,
        breakpoint_threshold_type="fixed",
        breakpoint_threshold_amount=similarity_threshold,
    )

    docs = text_splitter.create_documents(texts)
    return [doc.page_content for doc in docs]