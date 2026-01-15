import re
from typing import Protocol

import numpy as np
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker


# 1. Define Protocols for Type Hinting
class EmbeddingVectorizer(Protocol):
    def embed_documents(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LangChainEmbeddingWrapper:
    """Wrapper to make your Vectorizer compatible with LangChain."""

    def __init__(self, vectorizer: EmbeddingVectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # We assume the internal vectorizer handles the batching logic for the GPU
        return self.vectorizer.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.vectorizer.embed_query(text)


# 2. The Optimized Chunker Class
class BatchSemanticChunker(SemanticChunker):
    """
    A drop-in replacement for SemanticChunker that processes a list of documents
    in a single massive GPU batch instead of sequentially.
    """

    def __init__(
            self,
            embeddings,
            buffer_size: int = 1,
            breakpoint_threshold_type: str = "percentile",
            breakpoint_threshold_amount: float = 95.0,  # Default high for similarity
            number_of_chunks: int = None
    ):
        super().__init__(
            embeddings=embeddings,
            buffer_size=buffer_size,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks
        )

    def create_documents(self, texts: list[str], metadatas: list[dict] = None) -> list[Document]:
        """
        Overridden to flatten all sentences from all texts into one GPU call.
        """
        if not texts:
            return []

        # A. PRE-SPLIT ALL TEXTS (CPU)
        # We use a standard regex for sentence splitting (similar to LangChain default)
        # Using a simple regex is much faster than NLTK/Spacy for massive datasets.
        sentence_splitter = re.compile(r'(?<=[.?!])\s+')

        all_sentences = []
        doc_sentence_counts = []

        for text in texts:
            # Split and remove empty strings
            sents = [s for s in sentence_splitter.split(text) if s.strip()]
            if not sents:
                doc_sentence_counts.append(0)
                continue

            all_sentences.extend(sents)
            doc_sentence_counts.append(len(sents))

        if not all_sentences:
            return []

        # B. BATCH EMBEDDING (GPU)
        # This is the magic. 512 docs -> ~5000 sentences -> 1 GPU Call.
        # We cast to NumPy immediately for fast math.
        embeddings = self.embeddings.embed_documents(all_sentences)
        np_embeddings = np.array(embeddings)

        # C. RECONSTRUCT AND SPLIT (Logic)
        documents = []
        cursor = 0

        # We iterate through the original documents, slicing the big embedding matrix
        for i, count in enumerate(doc_sentence_counts):
            if count == 0:
                continue

            # Slice the pre-computed data for this specific document
            doc_sents = all_sentences[cursor: cursor + count]
            doc_embs = np_embeddings[cursor: cursor + count]
            cursor += count

            # --- LangChain Logic Re-implementation using NumPy ---
            # We calculate cosine distances between adjacent sentences

            # 1. Combine buffer (if buffer_size > 1, we avg sliding windows)
            # For simplicity and speed, we assume buffer_size=1 (standard)
            # If you need buffer_size > 1, we can add window averaging here.

            # 2. Calculate Distances (Cosine Similarity)
            # dist = 1 - dot_product(norm_vec_a, norm_vec_b)
            # But LangChain usually looks at "distances".
            # Dot product of normalized vectors is fast.
            dists = []
            if len(doc_embs) > 1:
                # Vectorized dot product of adjacent rows
                # a[0]•a[1], a[1]•a[2], ...
                sims = np.sum(doc_embs[:-1] * doc_embs[1:], axis=1)
                # Convert similarity (1.0 is identical) to distance (0.0 is identical)
                # Depending on what breakpoint_threshold_type expects. 
                # "percentile" usually works on *distances* (high distance = split).
                dists = 1.0 - sims

                # 3. Determine Threshold
            threshold = 0.0
            if len(dists) > 0:
                if self.breakpoint_threshold_type == "percentile":
                    threshold = np.percentile(dists, self.breakpoint_threshold_amount)
                elif self.breakpoint_threshold_type == "standard_deviation":
                    threshold = np.mean(dists) + self.breakpoint_threshold_amount * np.std(dists)
                elif self.breakpoint_threshold_type == "interquartile":
                    q1, q3 = np.percentile(dists, [25, 75])
                    iqr = q3 - q1
                    threshold = q3 + 1.5 * iqr

            # 4. Build Chunks
            current_chunk = [doc_sents[0]]
            for j, distance in enumerate(dists):
                # If distance is high (similarity low), we split
                if distance > threshold:
                    content = " ".join(current_chunk)
                    documents.append(Document(page_content=content, metadata=metadatas[i] if metadatas else {}))
                    current_chunk = [doc_sents[j + 1]]
                else:
                    current_chunk.append(doc_sents[j + 1])

            # Append final chunk
            if current_chunk:
                content = " ".join(current_chunk)
                documents.append(Document(page_content=content, metadata=metadatas[i] if metadatas else {}))

        return documents


# 3. The Function to call
def chunk_semantic(
        text: str | list[str],
        *,
        chunking_embeddings: EmbeddingVectorizer | str,
        similarity_threshold: float = 0.8,  # This comes from your config
) -> list[str]:
    if not text:
        return []

    texts = [text] if isinstance(text, str) else text
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []

    if isinstance(chunking_embeddings, str):
        raise ValueError("chunking_embeddings must be a Vectorizer instance.")

    wrapped_embeddings = LangChainEmbeddingWrapper(chunking_embeddings)

    # CORRECTION: Restore the logic to map similarity to percentile
    # If similarity_threshold is 0.8 (high similarity), we want to be strict.
    # In 'percentile' mode, a high threshold amount (e.g. 95) means "only split on the top 5% most different sentences".
    # Logic:
    #   threshold 0.9 (Very strict, many splits) -> Percentile 10 ??
    #   Actually, standard logic:
    #   Lower percentile = More splits (lower bar for difference)
    #   Higher percentile = Fewer splits (only split on massive differences)

    # Let's stick to the logic from your original file to maintain behavior:
    # breakpoint_percentile = int((1 - similarity_threshold) * 100)
    breakpoint_percentile = int((1 - similarity_threshold) * 100)

    # Ensure valid bounds for percentile (must be between 0 and 100)
    breakpoint_percentile = max(1, min(99, breakpoint_percentile))

    text_splitter = BatchSemanticChunker(
        embeddings=wrapped_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_percentile,
    )

    docs = text_splitter.create_documents(texts)

    return [doc.page_content for doc in docs]