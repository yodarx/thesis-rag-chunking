import numpy as np
import faiss
from src.data_loader import get_asqa_example
from src.chunking_strategies import chunk_fixed_size
from src.vectorizer import Vectorizer
from src.evaluation import calculate_metrics


def main():
    """
    Führt die gesamte RAG-Pipeline für ein einzelnes Beispiel aus:
    1. Offline-Prozess: Index aufbauen
    2. Online-Prozess: Suchen und Abrufen
    3. Evaluation: Ergebnisse bewerten
    """

    # --- 1. OFFLINE-PROZESS: INDEX AUFBAUEN ---
    print("--- OFFLINE-PROZESS START ---")
    print("Lade Beispieldaten...")
    example_data = get_asqa_example(0)

    if not example_data or not example_data["document_text"]:
        print("Fehler: Konnte keine Beispieldaten oder Dokumententext laden. Breche ab.")
        return

    document = example_data["document_text"]

    # Daten laden und chunken
    chunks = chunk_fixed_size(text=document, chunk_size=100, chunk_overlap=5)

    # Chunks vektorisieren
    vectorizer = Vectorizer()
    chunk_embeddings = vectorizer.embed_documents(chunks)
    chunk_embeddings_np = np.array(chunk_embeddings).astype('float32')

    # Index erstellen und füllen
    print("\nErstelle FAISS-Index...")
    embedding_dim = chunk_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(chunk_embeddings_np)
    print(f"Index mit {index.ntotal} Vektoren erstellt.")
    print("--- OFFLINE-PROZESS ENDE ---\n")

    # --- 2. ONLINE-PROZESS: SUCHE DURCHFÜHREN ---
    print("--- ONLINE-PROZESS START ---")
    question = example_data["question"]
    gold_passages = example_data["gold_passages"]
    top_k = 5  # Gemäss deiner Disposition

    print(f"Frage: '{question}'")

    # Frage vektorisieren
    question_embedding = vectorizer.embed_documents([question])
    question_embedding_np = np.array(question_embedding).astype('float32')

    # Suche im Index
    print(f"\nSuche die Top {top_k} relevantesten Chunks...")
    distances, indices = index.search(question_embedding_np, top_k)

    # Ergebnisse ausgeben
    print("\n--- SUCHERGEBNISSE (RETRIEVAL) ---")
    retrieved_chunks_text = []
    for i in range(top_k):
        chunk_index = indices[0][i]
        retrieved_chunk = chunks[chunk_index]
        retrieved_chunks_text.append(retrieved_chunk)
        print(f"--- Rang {i + 1} (Distanz: {distances[0][i]:.4f}) ---")
        print(retrieved_chunk)
        print("-" * 50)

    print("\n--- ERWARTETE PASSAGEN (GOLDSTANDARD) ---")
    for passage in gold_passages:
        print(f"- {passage}")

    # --- 3. EVALUATION ---
    print("\n--- EVALUATION DER ERGEBNISSE ---")
    metrics = calculate_metrics(
        retrieved_chunks=retrieved_chunks_text,
        gold_passages=gold_passages,
        k=top_k
    )
    print(f"Precision@{top_k}: {metrics['precision_at_k']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"MRR: {metrics['mrr']:.2f}")


if __name__ == "__main__":
    main()