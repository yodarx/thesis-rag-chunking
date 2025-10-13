import requests
from bs4 import BeautifulSoup
from datasets import load_dataset

# Lade den Datensatz einmalig.
try:
    asqa_dataset = load_dataset("din0s/asqa", split='train')
except Exception as e:
    print(f"Fehler beim Laden des Datensatzes: {e}")
    asqa_dataset = []


def _fetch_wikipedia_text(url: str) -> str:
    """
    Holt den Textinhalt von einer Wikipedia-URL.
    """
    try:
        # --- NEUER TEIL ---
        # Füge einen User-Agent-Header hinzu, um wie ein Browser auszusehen
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        # --- ENDE NEUER TEIL ---

        # Füge den 'headers'-Parameter zum Request hinzu
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        content_div = soup.find(id='mw-content-text')
        if not content_div:
            return ""

        paragraphs = content_div.find_all('p')
        return "\n".join([p.get_text() for p in paragraphs])

    except requests.RequestException as e:
        print(f"Fehler beim Abrufen der URL {url}: {e}")
        return ""

def get_asqa_example(index: int):
    """
    Lädt ein einzelnes, aufbereitetes Beispiel aus dem ASQA-Datensatz
    und holt den zugehörigen Wikipedia-Text.
    """
    if not 0 <= index < len(asqa_dataset):
        return None

    example = asqa_dataset[index]
    question = example.get('ambiguous_question', '')

    # --- NEUER TEIL ---
    # Rufe den Text für jede verlinkte Wikipedia-Seite ab
    document_texts = []
    wikipages = example.get('wikipages', [])
    for page in wikipages:
        url = page.get('url')
        if url:
            print(f"Rufe Text von URL ab: {url}")
            text = _fetch_wikipedia_text(url)
            if text:
                document_texts.append(text)

    document_text = "\n\n--- NEUER ARTIKEL ---\n\n".join(document_texts)

    # Extrahiere die Goldstandard-Passagen
    gold_passages = []
    qa_pairs = example.get('qa_pairs', [])
    for pair in qa_pairs:
        answers = pair.get('short_answers', [])
        gold_passages.extend(answers)

    unique_gold_passages = sorted(list(set(gold_passages)))

    return {
        "question": question,
        "document_text": document_text,
        "gold_passages": unique_gold_passages
    }