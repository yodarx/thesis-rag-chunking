import re
import numpy as np
from typing import List, Protocol, Literal
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document


# --- Protocols ---
class EmbeddingVectorizer(Protocol):
    def embed_documents(self, texts: list[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class LangChainEmbeddingWrapper:
    def __init__(self, vectorizer: EmbeddingVectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.vectorizer.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.vectorizer.embed_query(text)


# --- Optimized Chunker ---
class BatchSemanticChunker(SemanticChunker):
    def __init__(
            self,
            embeddings,
            breakpoint_threshold_type: Literal[
                "percentile", "standard_deviation", "interquartile", "fixed"] = "percentile",
            breakpoint_threshold_amount: float = 0.5
    ):
        # We initialize with standard defaults, but we will override the logic
        super().__init__(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type if breakpoint_threshold_type != "fixed" else "percentile",
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
        self.custom_threshold_type = breakpoint_threshold_type
        self.custom_threshold_amount = breakpoint_threshold_amount

    def create_documents(self, texts: List[str], metadatas: List[dict] = None) -> List[Document]:
        if not texts: return []

        # 1. ROBUST SENTENCE SPLITTING
        # Look for [.?!], followed by whitespace, followed by a CAPITAL letter.
        # Prevents splitting on "approx. 10" or "Mr. Smith".
        sentence_splitter = re.compile(r'(?<=[.?!])\s+(?=[A-Z])')

        all_sentences = []
        doc_sentence_counts = []

        for text in texts:
            # Clean and split
            sents = [s for s in sentence_splitter.split(text) if s.strip()]
            if not sents:
                doc_sentence_counts.append(0)
                continue
            all_sentences.extend(sents)
            doc_sentence_counts.append(len(sents))

        if not all_sentences: return []

        # 2. BATCH EMBEDDING (GPU Saturation)
        embeddings = self.embeddings.embed_documents(all_sentences)
        np_embeddings = np.array(embeddings)

        # 3. RECONSTRUCTION
        documents = []
        cursor = 0

        for i, count in enumerate(doc_sentence_counts):
            if count == 0: continue

            doc_sents = all_sentences[cursor: cursor + count]
            doc_embs = np_embeddings[cursor: cursor + count]
            cursor += count

            # Distance Calculation (Cosine Distance = 1 - Cosine Similarity)
            dists = []
            if len(doc_embs) > 1:
                # Normalized Dot Product
                sims = np.sum(doc_embs[:-1] * doc_embs[1:], axis=1)
                dists = 1.0 - sims

                # 4. THRESHOLD STRATEGY
            threshold = 0.0
            if len(dists) > 0:
                if self.custom_threshold_type == "fixed":
                    # Fixed Similarity Mode (Robust for RAG)
                    # If similarity_threshold is 0.8, we split when similarity < 0.8
                    # Which means Distance > (1 - 0.8) = 0.2
                    threshold = 1.0 - self.custom_threshold_amount

                elif self.breakpoint_threshold_type == "percentile":
                    threshold = np.percentile(dists, self.breakpoint_threshold_amount)
                elif self.breakpoint_threshold_type == "standard_deviation":
                    threshold = np.mean(dists) + self.breakpoint_threshold_amount * np.std(dists)
                elif self.breakpoint_threshold_type == "interquartile":
                    q1, q3 = np.percentile(dists, [25, 75])
                    threshold = q3 + 1.5 * (q3 - q1)

            # 5. CHUNKING LOOP
            current_chunk = [doc_sents[0]]
            for j, distance in enumerate(dists):
                if distance > threshold:
                    # SPLIT
                    content = " ".join(current_chunk)
                    documents.append(Document(page_content=content, metadata=metadatas[i] if metadatas else {}))
                    current_chunk = [doc_sents[j + 1]]
                else:
                    # MERGE
                    current_chunk.append(doc_sents[j + 1])

            if current_chunk:
                content = " ".join(current_chunk)
                documents.append(Document(page_content=content, metadata=metadatas[i] if metadatas else {}))

        return documents


# --- Main Entry Point ---
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