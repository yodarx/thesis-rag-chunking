import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# NLTK-Modelle müssen einmalig heruntergeladen werden:
# In einer Python-Konsole: import nltk; nltk.download('punkt')
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    print("NLTK nicht gefunden oder 'punkt' nicht heruntergeladen.")
    print("Bitte führe 'pip install nltk' aus und dann in Python: nltk.download('punkt')")

# --- STRATEGIE 1: FIXED-SIZE CHUNKING ---
def chunk_fixed_size(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Zerlegt einen Text in Chunks fester Grösse mit Überlappung."""
    if not text: return []
    chunks = []
    start_index = 0
    step_size = chunk_size - chunk_overlap
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += step_size
    return chunks


# --- STRATEGIE 2: SENTENCE CHUNKING ---
def chunk_by_sentence(text: str, sentences_per_chunk: int) -> list[str]:
    """Zerlegt einen Text in Chunks, die eine bestimmte Anzahl Sätze enthalten."""
    if not text: return []
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks


# --- STRATEGIE 3: RECURSIVE CHARACTER SPLITTING ---
def chunk_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Zerlegt Text rekursiv anhand einer Hierarchie von Trennzeichen."""
    if not text: return []

    separators = ["\n\n", "\n", ". ", " ", ""]

    # Beginne mit dem ganzen Text als erstem zu verarbeitenden "Stück"
    final_chunks = []
    splits = [text]

    for sep in separators:
        new_splits = []
        for split in splits:
            if len(split) <= chunk_size:
                new_splits.append(split)
                continue

            # Teile den Text am aktuellen Trennzeichen
            sub_splits = split.split(sep)

            # Füge die Teile zusammen, bis sie die chunk_size erreichen (mit Überlappung)
            current_chunk = ""
            for sub_split in sub_splits:
                # Wenn der aktuelle Chunk plus der nächste Teil zu gross wäre
                if len(current_chunk) + len(sub_split) + len(sep) > chunk_size and current_chunk:
                    new_splits.append(current_chunk)
                    # Beginne den nächsten Chunk mit einer Überlappung
                    current_chunk = current_chunk[-chunk_overlap:] + sep + sub_split
                else:
                    current_chunk += sep + sub_split
            if current_chunk:
                new_splits.append(current_chunk)

        splits = new_splits

    # Filtere leere Chunks und returniere das Ergebnis
    return [s.strip() for s in splits if s.strip()]


# --- STRATEGIE 4: SEMANTIC CHUNKING ---
def chunk_semantic(text: str, vectorizer, similarity_threshold: float = 0.8) -> list[str]:
    """Zerlegt Text basierend auf der semantischen Ähnlichkeit benachbarter Sätze."""
    if not text: return []

    sentences = sent_tokenize(text)
    if len(sentences) < 2: return sentences

    sentence_embeddings = np.array(vectorizer.embed_documents(sentences))

    similarities = [
        cosine_similarity(sentence_embeddings[i].reshape(1, -1), sentence_embeddings[i + 1].reshape(1, -1))[0][0]
        for i in range(len(sentences) - 1)]

    chunks = []
    current_chunk_sentences = [sentences[0]]

    for i, similarity in enumerate(similarities):
        if similarity < similarity_threshold:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []

        current_chunk_sentences.append(sentences[i + 1])

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks
