import json
import requests
import argparse
import re
from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm
import os
from typing import Dict, Any, List, Optional

# Setze hier das Limit. Für den gesamten Datensatz auf None setzen.
PREPROCESS_LIMIT: Optional[int] = 20

_WHITESPACE_COLLAPSE_RE = re.compile(r"[ \t\u00A0]+")
_CONTROL_CHARS_RE = re.compile(r"[\u200B\u200E\u200F]+")


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00A0", " ")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _WHITESPACE_COLLAPSE_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join([ln for ln in lines if ln])
    return cleaned.strip()


def _serialize_table(tbl) -> List[List[str]]:
    rows: List[List[str]] = []
    for tr in tbl.find_all('tr'):
        cells = [_clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(['th', 'td'])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    return rows


def _fetch_wikipedia_page(url: str, as_markdown: bool = False) -> Dict[str, Any]:
    """Holt umfassenden Text + Metadaten und serialisiert alles in text (Autor ignoriert).
    Wenn as_markdown True ist, werden Tabellen als Markdown ausgegeben."""
    page: Dict[str, Any] = {
        "url": url,
        "title": "",
        "last_modified": "",
        "text": "",
        "tables": [],
        "infobox": [],
        "categories": [],
        "references": [],
        "images": []
    }
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        page["last_modified"] = response.headers.get("Last-Modified", "")
        soup = BeautifulSoup(response.content, 'html.parser')

        title_tag = soup.find(id='firstHeading') or soup.find('title')
        if title_tag:
            page["title"] = _clean_text(title_tag.get_text(strip=True))

        content_div = soup.find(id='mw-content-text')
        if not content_div:
            return page

        text_parts: List[str] = []

        if page["title"]:
            text_parts.append(f"TITLE: {page['title']}")
        if page["last_modified"]:
            text_parts.append(f"LAST_MODIFIED: {page['last_modified']}")

        # Headings
        for h in content_div.find_all(['h2', 'h3', 'h4', 'h5', 'h6']):
            htxt = _clean_text(h.get_text(" ", strip=True))
            if htxt:
                text_parts.append(f"# {htxt}")

        # Paragraphs
        for p in content_div.find_all('p'):
            ptxt = _clean_text(p.get_text(" ", strip=True))
            if ptxt:
                text_parts.append(ptxt)

        # Lists
        for lst in content_div.find_all(['ul', 'ol']):
            items = []
            for li in lst.find_all('li', recursive=False):
                litxt = _clean_text(li.get_text(" ", strip=True))
                if litxt:
                    items.append(f"- {litxt}")
            if items:
                text_parts.append("\n".join(items))

        # Blockquotes
        for bq in content_div.find_all('blockquote'):
            bqtxt = _clean_text(bq.get_text(" ", strip=True))
            if bqtxt:
                text_parts.append(f"> {bqtxt}")

        # Image captions and alt texts
        image_lines: List[str] = []
        for cap in content_div.find_all('div', class_='thumbcaption'):
            ctxt = _clean_text(cap.get_text(" ", strip=True))
            if ctxt:
                image_lines.append(f"IMAGE_CAPTION: {ctxt}")
        for figcap in content_div.find_all('figcaption'):
            ftxt = _clean_text(figcap.get_text(" ", strip=True))
            if ftxt:
                image_lines.append(f"FIG_CAPTION: {ftxt}")
        for img in content_div.find_all('img'):
            alt = _clean_text(img.get('alt') or "")
            if alt:
                image_lines.append(f"IMAGE_ALT: {alt}")
        if image_lines:
            page["images"] = image_lines
            text_parts.extend(image_lines)

        # Tables
        tables_serialized: List[Dict[str, Any]] = []
        for idx, tbl in enumerate(content_div.find_all('table')):
            rows = _serialize_table(tbl)
            if rows:
                tables_serialized.append({"index": idx, "rows": rows})
                if as_markdown:
                    header = rows[0]
                    md_lines = [
                        "| " + " | ".join(header) + " |",
                        "| " + " | ".join(["---"] * len(header)) + " |"
                    ]
                    for r in rows[1:]:
                        md_lines.append("| " + " | ".join(r) + " |")
                    text_parts.append("\n".join(md_lines))
                else:
                    flat = ["\t".join(r) for r in rows]
                    text_parts.append("TABLE:")
                    text_parts.append("\n".join(flat))
        page["tables"] = tables_serialized

        # Infobox
        infobox_tbl = soup.find('table', class_='infobox')
        if infobox_tbl:
            ib_rows = _serialize_table(infobox_tbl)
            if ib_rows:
                page["infobox"] = ib_rows
                text_parts.append("INFOBOX:")
                text_parts.extend(["\t".join(r) for r in ib_rows])

        # References
        ref_texts = []
        for ref in soup.select("ol.references li .reference-text"):
            rtxt = _clean_text(ref.get_text(" ", strip=True))
            if rtxt:
                ref_texts.append(rtxt)
        if ref_texts:
            page["references"] = ref_texts
            text_parts.append("REFERENCES:")
            text_parts.extend(ref_texts)

        # Categories
        cat_div = soup.find(id='mw-normal-catlinks')
        if cat_div:
            cats = [_clean_text(a.get_text(strip=True)) for a in cat_div.find_all('a') if a.get_text(strip=True)]
            cats_clean = [c for c in cats if c.lower() not in ('categories', 'kategorien')]
            page["categories"] = cats_clean
            if cats_clean:
                text_parts.append("CATEGORIES: " + ", ".join(cats_clean))

        joined = "\n".join(text_parts).strip()
        page["text"] = _clean_text(joined)
        return page
    except requests.RequestException:
        return page


def _format_document_text(pages: List[Dict[str, Any]], as_markdown: bool) -> str:
    if not pages:
        return ""
    if not as_markdown:
        raw = "\n\n--- ARTICLE SPLIT ---\n\n".join([p["text"] for p in pages])
        return _clean_text(raw)
    segments: List[str] = []
    for p in pages:
        title = p.get("title") or "Untitled Page"
        segments.append(f"# {title}\n\n{p['text']}")
    return _clean_text("\n\n---\n\n".join(segments))


def main():
    """Lädt ASQA, holt Wikipedia Daten, schreibt alles in document_text mit wählbarem Format."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=["plain", "markdown"],
        default=os.getenv("ASQA_OUTPUT_FORMAT", "plain"),
        help="Format für document_text Ausgabe."
    )
    args = parser.parse_args()
    use_markdown = args.format == "markdown"

    print("Lade originalen ASQA-Datensatz von Hugging Face...")
    asqa_dataset = load_dataset("din0s/asqa", split='train')

    os.makedirs('data/processed', exist_ok=True)
    output_file_path = f"data/processed/asqa_preprocessed_{args.format}.jsonl"
    print(f"Starte Vorverarbeitung. Ergebnisse werden in `{output_file_path}` gespeichert.")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(tqdm(asqa_dataset, desc="Verarbeite ASQA Beispiele")):
            question = _clean_text(example.get('ambiguous_question', ''))
            sample_id = example.get('sample_id', '')

            gold_passages: List[str] = []
            for pair in example.get('qa_pairs', []):
                for ans in pair.get('short_answers', []):
                    cleaned = _clean_text(ans)
                    if cleaned:
                        gold_passages.append(cleaned)

            pages: List[Dict[str, Any]] = []
            for page_entry in example.get('wikipages', []):
                url = page_entry.get('url')
                if url:
                    pdata = _fetch_wikipedia_page(url, as_markdown=use_markdown)
                    if pdata.get("text"):
                        pages.append(pdata)

            document_text = _format_document_text(pages, use_markdown)
            last_url = pages[-1]["url"] if pages else ""

            if document_text:
                processed_entry = {
                    "sample_id": sample_id,
                    "question": question,
                    "gold_passages": sorted(list(set(gold_passages))),
                    "document_text": document_text,
                    "pages": pages,
                    "url": last_url,
                    "output_format": args.format
                }
                f.write(json.dumps(processed_entry, ensure_ascii=False) + '\n')

            if PREPROCESS_LIMIT and i + 1 >= PREPROCESS_LIMIT:
                print(f"\nLimit von {PREPROCESS_LIMIT} Einträgen erreicht. Breche Vorverarbeitung ab.")
                break

    print(f"Vorverarbeitung abgeschlossen. Verarbeitete Daten unter `{output_file_path}` gespeichert.")


if __name__ == "__main__":
    main()