from sentence_transformers import SentenceTransformer

class Vectorizer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialisiert den Vectorizer durch Laden des Embedding-Modells.
        """
        print(f"Lade Embedding-Modell: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Modell geladen.")

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Wandelt eine Liste von Textdokumenten in Vektor-Embeddings um.
        """
        # Die Vektorisierung ist für die Konsole oft sehr "gesprächig".
        # show_progress_bar=False reduziert die Ausgabe während der grossen Läufe.
        embeddings = self.model.encode(documents, show_progress_bar=False)
        return embeddings.tolist()