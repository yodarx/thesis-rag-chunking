import json
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm
import os

# Setze hier das Limit. Für den gesamten Datensatz auf None setzen.
PREPROCESS_LIMIT = None

def _fetch_wikipedia_text(url: str) -> str:
    """Holt den Textinhalt von einer Wikipedia-URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content_div = soup.find(id='mw-content-text')
        if not content_div: return ""
        paragraphs = content_div.find_all('p')
        return "\n".join([p.get_text() for p in paragraphs])
    except requests.RequestException:
        return ""


def main():
    """
    Lädt den ASQA-Datensatz, ruft für jeden Eintrag den Wikipedia-Text ab
    und speichert das Ergebnis in einer JSON Lines Datei.
    """
    print("Lade originalen ASQA-Datensatz von Hugging Face...")
    asqa_dataset = load_dataset("din0s/asqa", split='train')

    # Stelle sicher, dass der 'processed' Ordner existiert
    os.makedirs('data/processed', exist_ok=True)
    output_file_path = 'data/processed/asqa_preprocessed.jsonl'
    print(f"Starte Vorverarbeitung. Ergebnisse werden in '{output_file_path}' gespeichert.")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(tqdm(asqa_dataset, desc="Verarbeite ASQA Beispiele")):
            question = example.get('ambiguous_question', '')
            sample_id = example.get('sample_id', '')

            gold_passages = []
            for pair in example.get('qa_pairs', []):
                gold_passages.extend(pair.get('short_answers', []))

            document_texts = []
            for page in example.get('wikipages', []):
                url = page.get('url')
                if url:
                    text = _fetch_wikipedia_text(url)
                    if text:
                        document_texts.append(text)

            document_text = "\n\n--- NEUER ARTIKEL ---\n\n".join(document_texts)

            if document_text:
                processed_entry = {
                    "sample_id": sample_id,
                    "question": question,
                    "gold_passages": sorted(list(set(gold_passages))),
                    "document_text": document_text
                }
                f.write(json.dumps(processed_entry, ensure_ascii=False) + '\n')

            if PREPROCESS_LIMIT and i + 1 >= PREPROCESS_LIMIT:
                print(f"\nLimit von {PREPROCESS_LIMIT} Einträgen erreicht. Breche Vorverarbeitung ab.")
                break

    print(f"Vorverarbeitung abgeschlossen. Verarbeitete Daten unter '{output_file_path}' gespeichert.")


if __name__ == "__main__":
    main()