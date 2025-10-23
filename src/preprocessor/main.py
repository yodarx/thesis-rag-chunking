import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from datasets import load_dataset
from requests.exceptions import RequestException
from tqdm import tqdm

# --- Globale Konfiguration ---
WHITESPACE_RE = re.compile(r"[ \t\u00A0]+")
CONTROL_CHARS_RE = re.compile(r"[\u200B\u200E\u200F]+")
HTTP_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
}
BASE_PATH = "./data/preprocessed"


# --- Datenstrukturen ---


@dataclass
class SerializedTable:
    index: int
    rows: list[list[str]]


@dataclass
class PageData:
    url: str
    title: str = ""
    last_modified: str = ""
    text: str = ""
    tables: list[SerializedTable] = field(default_factory=list)
    infobox: list[list[str]] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


# --- Text-Bereinigung und Serialisierung ---


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = CONTROL_CHARS_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join([ln for ln in lines if ln])
    return cleaned.strip()


def serialize_table(table_tag: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = [clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    return rows


# --- Wikipedia-Parsing-Helfer ---


def _extract_headings(content_div: Tag) -> list[str]:
    parts: list[str] = []
    for h in content_div.find_all(["h2", "h3", "h4", "h5", "h6"]):
        htxt = clean_text(h.get_text(" ", strip=True))
        if htxt:
            parts.append(f"# {htxt}")
    return parts


def _extract_paragraphs(content_div: Tag) -> list[str]:
    parts: list[str] = []
    for p in content_div.find_all("p"):
        ptxt = clean_text(p.get_text(" ", strip=True))
        if ptxt:
            parts.append(ptxt)
    return parts


def _extract_lists(content_div: Tag) -> list[str]:
    parts: list[str] = []
    for lst in content_div.find_all(["ul", "ol"]):
        items = []
        for li in lst.find_all("li", recursive=False):
            litxt = clean_text(li.get_text(" ", strip=True))
            if litxt:
                items.append(f"- {litxt}")
        if items:
            parts.append("\n".join(items))
    return parts


def _extract_blockquotes(content_div: Tag) -> list[str]:
    parts: list[str] = []
    for bq in content_div.find_all("blockquote"):
        bqtxt = clean_text(bq.get_text(" ", strip=True))
        if bqtxt:
            parts.append(f"> {bqtxt}")
    return parts


def _extract_images(content_div: Tag) -> list[str]:
    image_lines: list[str] = []
    for cap in content_div.find_all("div", class_="thumbcaption"):
        ctxt = clean_text(cap.get_text(" ", strip=True))
        if ctxt:
            image_lines.append(f"IMAGE_CAPTION: {ctxt}")
    for figcap in content_div.find_all("figcaption"):
        ftxt = clean_text(figcap.get_text(" ", strip=True))
        if ftxt:
            image_lines.append(f"FIG_CAPTION: {ftxt}")
    for img in content_div.find_all("img"):
        alt = clean_text(img.get("alt") or "")
        if alt:
            image_lines.append(f"IMAGE_ALT: {alt}")
    return image_lines


def _extract_tables(content_div: Tag) -> tuple[list[SerializedTable], list[str]]:
    tables_data: list[SerializedTable] = []
    text_parts: list[str] = []
    for idx, tbl in enumerate(content_div.find_all("table")):
        rows = serialize_table(tbl)
        if rows:
            tables_data.append(SerializedTable(index=idx, rows=rows))
            flat = ["\t".join(r) for r in rows]
            text_parts.append("TABLE:")
            text_parts.append("\n".join(flat))
    return tables_data, text_parts


def _extract_infobox(soup: BeautifulSoup) -> tuple[list[list[str]], list[str]]:
    infobox_tbl = soup.find("table", class_="infobox")
    if infobox_tbl:
        ib_rows = serialize_table(infobox_tbl)
        if ib_rows:
            text_parts = ["INFOBOX:"] + ["\t".join(r) for r in ib_rows]
            return ib_rows, text_parts
    return [], []


def _extract_references(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    ref_texts: list[str] = []
    for ref in soup.select("ol.references li .reference-text"):
        rtxt = clean_text(ref.get_text(" ", strip=True))
        if rtxt:
            ref_texts.append(rtxt)

    text_parts: list[str] = []
    if ref_texts:
        text_parts.append("REFERENCES:")
        text_parts.extend(ref_texts)
    return ref_texts, text_parts


def _extract_categories(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    cat_div = soup.find(id="mw-normal-catlinks")
    if cat_div:
        cats = [
            clean_text(a.get_text(strip=True))
            for a in cat_div.find_all("a")
            if a.get_text(strip=True)
        ]
        cats_clean = [c for c in cats if c.lower() not in ("categories", "kategorien")]
        if cats_clean:
            text_part = "CATEGORIES: " + ", ".join(cats_clean)
            return cats_clean, [text_part]
    return [], []


# --- Haupt-Parsing- und Formatierungslogik ---


def fetch_and_parse_page(url: str) -> PageData | None:
    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=12)
        response.raise_for_status()
    except RequestException:
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    content_div = soup.find(id="mw-content-text")
    if not content_div:
        return None

    page = PageData(url=url)
    page.last_modified = response.headers.get("Last-Modified", "")

    title_tag = soup.find(id="firstHeading") or soup.find("title")
    if title_tag:
        page.title = clean_text(title_tag.get_text(strip=True))

    text_parts: list[str] = []
    if page.title:
        text_parts.append(f"TITLE: {page.title}")
    if page.last_modified:
        text_parts.append(f"LAST_MODIFIED: {page.last_modified}")

    text_parts.extend(_extract_headings(content_div))
    text_parts.extend(_extract_paragraphs(content_div))
    text_parts.extend(_extract_lists(content_div))
    text_parts.extend(_extract_blockquotes(content_div))

    page.images = _extract_images(content_div)
    text_parts.extend(page.images)

    page.tables, table_texts = _extract_tables(content_div)
    text_parts.extend(table_texts)

    page.infobox, infobox_texts = _extract_infobox(soup)
    text_parts.extend(infobox_texts)

    page.references, ref_texts = _extract_references(soup)
    text_parts.extend(ref_texts)

    page.categories, cat_texts = _extract_categories(soup)
    text_parts.extend(cat_texts)

    joined = "\n".join(text_parts).strip()
    page.text = clean_text(joined)

    return page


def format_document_text(pages: list[PageData]) -> str:
    if not pages:
        return ""
    raw = "\n\n--- ARTICLE SPLIT ---\n\n".join([p.text for p in pages])
    return raw


# --- ASQA-Datensatzverarbeitung ---


def extract_gold_passages(example: dict[str, Any]) -> list[str]:
    gold_passages: list[str] = []
    for pair in example.get("qa_pairs", []):
        for ans in pair.get("short_answers", []):
            cleaned = clean_text(ans)
            if cleaned:
                gold_passages.append(cleaned)
    return sorted(list(set(gold_passages)))


def fetch_pages_for_example(example: dict[str, Any]) -> list[PageData]:
    pages: list[PageData] = []
    for page_entry in example.get("wikipages", []):
        url = page_entry.get("url")
        if url:
            pdata = fetch_and_parse_page(url)
            if pdata and pdata.text:
                pages.append(pdata)
    return pages


def create_output_entry(
    example: dict[str, Any], pages: list[PageData], document_text: str
) -> dict[str, Any]:
    last_url = pages[-1].url if pages else ""
    pages_as_dicts = [asdict(p) for p in pages]

    return {
        "sample_id": example.get("sample_id", ""),
        "question": clean_text(example.get("ambiguous_question", "")),
        "gold_passages": extract_gold_passages(example),
        "document_text": document_text,
        "pages": pages_as_dicts,
        "url": last_url,
    }


def get_output_file_path(limit: str | None) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    count_str = f"limit_{limit}" if limit is not None else "all"

    output_file_path = os.path.join(BASE_PATH, f"preprocessed_{date_str}_{count_str}.jsonl")
    return output_file_path


# --- Hauptskript ---


def main(limit: int | None):
    print("Lade originalen ASQA-Datensatz von Hugging Face...")
    asqa_dataset = load_dataset("din0s/asqa", split="train")

    output_file_path = get_output_file_path(limit)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    print(f"Starte Vorverarbeitung. Ergebnisse werden in `{output_file_path}` gespeichert.")

    with open(output_file_path, "w", encoding="utf-8") as f:
        # Ersetze ASQA_DATASET durch die lokale Variable asqa_dataset
        for i, example in enumerate(tqdm(asqa_dataset, desc="Verarbeite ASQA Beispiele")):
            pages = fetch_pages_for_example(example)
            document_text = format_document_text(pages)

            if document_text:
                processed_entry = create_output_entry(example, pages, document_text)
                f.write(json.dumps(processed_entry, ensure_ascii=False) + "\n")

            if limit and i + 1 >= limit:
                print(f"\nLimit von {limit} Einträgen erreicht. Breche Vorverarbeitung ab.")
                break

    print(
        f"Vorverarbeitung abgeschlossen. Verarbeitete Daten unter `{output_file_path}` gespeichert."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preprocess the ASQA dataset from Wikipedia."
    )

    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit the number of dataset entries to process (default: process all).",
    )

    args = parser.parse_args()

    main(limit=args.limit)
