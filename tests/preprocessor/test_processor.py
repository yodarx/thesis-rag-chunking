import json
from dataclasses import asdict

import pytest
from bs4 import BeautifulSoup
from pytest_mock import MockerFixture
from requests.exceptions import RequestException

from preprocessor.main import (
    HTTP_HEADERS,
    PageData,
    clean_text,
    create_output_entry,
    extract_gold_passages,
    fetch_and_parse_page,
    format_document_text,
    main,
    serialize_table,
)


@pytest.fixture
def sample_html_content() -> str:
    """Eine umfassende HTML-Beispielseite für Tests."""
    return """
    <html>
    <head><title>Test Titel</title></head>
    <body>
        <h1 id="firstHeading">Test Titel</h1>
        <div id="mw-content-text">
            <h2>Überschrift 2</h2>
            <p>Dies ist ein Absatz mit \u00a0 geschütztem Leerzeichen.</p>
            <h3>Überschrift 3</h3>
            <p>Ein weiterer Absatz.</p>
            <ul>
                <li>Listenpunkt 1</li>
                <li>Listenpunkt 2</li>
            </ul>
            <blockquote>Ein Zitat</blockquote>

            <div class="thumbcaption">Eine Bildunterschrift</div>
            <figcaption>Eine Figcaption</figcaption>
            <img alt="Ein Alternativtext" />

            <table class="wikitable">
                <tr><th>Header 1</th><th>Header 2</th></tr>
                <tr><td>Daten 1</td><td>Daten 2</td></tr>
            </table>
        </div>

        <table class="infobox">
            <tr><th>Info</th><td>Wert</td></tr>
        </table>

        <ol class="references">
            <li><span class="reference-text">Referenz 1 Text</span></li>
            <li><span class="reference-text">Referenz 2 Text</span></li>
        </ol>

        <div id="mw-normal-catlinks">
            <ul>
                <li><a href="#">Kategorien</a></li>
                <li><a href="#">Cat 1</a></li>
                <li><a href="#">Cat 2</a></li>
            </ul>
        </div>
    </body>
    </html>
    """


# Die 'sample_soup' und 'sample_content_div' fixtures werden
# von den entfernten Tests verwendet, aber nicht mehr von den verbleibenden.
# Wir können sie entfernen, um die Datei aufzuräumen.

# --- Tests für Hilfsfunktionen ---


def test_clean_text():
    assert clean_text("") == ""
    assert clean_text("  Hallo \t Welt\u00a0!  ") == "Hallo Welt !"
    assert (
        clean_text("Text\u200bmit\u200eunsichtbaren\u200fZeichen") == "TextmitunsichtbarenZeichen"
    )
    assert clean_text("\nZeile 1\n\n   \nZeile 2\n") == "Zeile 1\nZeile 2"
    assert clean_text("Nur Text") == "Nur Text"


def test_serialize_table():
    html = """
    <table>
        <tr><th>Name</th><th>Alter</th></tr>
        <tr><td>Alice</td><td>30</td></tr>
        <tr><td>Bob</td><td></td></tr>
        <tr><td>Charlie</td><td>40</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table_tag = soup.find("table")
    expected = [
        ["Name", "Alter"],
        ["Alice", "30"],
        ["Bob"],  # Leere Zellen werden herausgefiltert
        ["Charlie", "40"],
    ]
    assert serialize_table(table_tag) == expected


# --- Tests für Extraktionsfunktionen (ENTFERNT) ---
# Die Tests für _extract_headings, _extract_paragraphs, _extract_lists,
# _extract_blockquotes, _extract_images, _extract_tables, _extract_infobox,
# _extract_references, und _extract_categories wurden entfernt.
# Ihre Funktionalität wird implizit durch test_fetch_and_parse_page_success abgedeckt.


# --- Tests für Haupt-Parsing- und Formatierungslogik ---


def test_fetch_and_parse_page_success(mocker: MockerFixture, sample_html_content: str):
    """
    Dieser Test ist jetzt der Haupt-Integrationstest für das Parsen.
    Er prüft, ob alle Elemente (Überschriften, Absätze, Tabellen etc.)
    korrekt im finalen Text-Output landen.
    """
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Last-Modified": "Test-Date"}
    mock_response.content = sample_html_content.encode("utf-8")
    mock_response.raise_for_status = mocker.Mock()

    mock_get = mocker.patch("preprocessor.main.requests.get", return_value=mock_response)

    url = "http://fake-wiki.com/Test_Titel"
    page = fetch_and_parse_page(url)

    mock_get.assert_called_once_with(url, headers=HTTP_HEADERS, timeout=12)
    assert page is not None
    assert page.url == url
    assert page.title == "Test Titel"
    assert page.last_modified == "Test-Date"

    # --- Überprüfung der geparsten Elemente ---

    # Testet implizit _extract_headings
    assert "# Überschrift 2" in page.text
    assert "# Überschrift 3" in page.text

    # Testet implizit _extract_paragraphs
    assert "Dies ist ein Absatz mit geschütztem Leerzeichen." in page.text

    # Testet implizit _extract_lists
    assert "- Listenpunkt 1\n- Listenpunkt 2" in page.text

    # Testet implizit _extract_blockquotes
    assert "> Ein Zitat" in page.text

    # Testet implizit _extract_images
    assert "IMAGE_CAPTION: Eine Bildunterschrift" in page.text
    assert "FIG_CAPTION: Eine Figcaption" in page.text
    assert "IMAGE_ALT: Ein Alternativtext" in page.text

    # Testet implizit _extract_tables und clean_text (Tabs -> Spaces)
    assert "TABLE:\nHeader 1 Header 2\nDaten 1 Daten 2" in page.text

    # Testet implizit _extract_infobox
    assert "INFOBOX:\nInfo Wert" in page.text

    # Testet implizit _extract_references
    assert "REFERENCES:\nReferenz 1 Text\nReferenz 2 Text" in page.text

    # Testet implizit _extract_categories
    assert "CATEGORIES: Cat 1, Cat 2" in page.text

    # --- Überprüfung der strukturierten Daten ---
    assert page.categories == ["Cat 1", "Cat 2"]
    assert page.references == ["Referenz 1 Text", "Referenz 2 Text"]
    assert page.infobox == [["Info", "Wert"]]


def test_fetch_and_parse_page_request_error(mocker: MockerFixture):
    mocker.patch("preprocessor.main.requests.get", side_effect=RequestException("Netzwerkfehler"))

    page = fetch_and_parse_page("http://bad-url.com")
    assert page is None


def test_fetch_and_parse_page_no_content(mocker: MockerFixture):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.content = b"<html><body>Kein Inhalt</body></html>"
    mock_response.raise_for_status = mocker.Mock()

    mocker.patch("preprocessor.main.requests.get", return_value=mock_response)

    page = fetch_and_parse_page("http://no-content-div.com")
    assert page is None  # Gibt None zurück, da #mw-content-text fehlt


def test_format_document_text():
    assert format_document_text([]) == ""

    p1 = PageData(url="url1", text="Text von Seite 1")
    assert format_document_text([p1]) == "Text von Seite 1"

    p2 = PageData(url="url2", text="Text von Seite 2")
    expected = "Text von Seite 1\n\n--- ARTICLE SPLIT ---\n\nText von Seite 2"
    assert format_document_text([p1, p2]) == expected


# --- Tests für ASQA-Datensatzverarbeitung ---


def test_extract_gold_passages():
    example = {
        "qa_pairs": [
            {"short_answers": [" Antwort 1 ", "Antwort 2\n", " Antwort 1 "]},
            {"short_answers": ["Antwort 3", "Antwort 2"]},
        ]
    }
    expected = ["Antwort 1", "Antwort 2", "Antwort 3"]
    assert extract_gold_passages(example) == expected


def test_create_output_entry():
    example = {
        "sample_id": "s1",
        "ambiguous_question": " Warum? \n",
        "qa_pairs": [{"short_answers": ["Weil"]}],
    }
    pages = [
        PageData(url="url1", title="Titel1", text="Text1"),
        PageData(url="url2", title="Titel2", text="Text2"),
    ]
    doc_text = "Text1\n\n--- ARTICLE SPLIT ---\n\nText2"

    entry = create_output_entry(example, pages, doc_text)

    expected = {
        "sample_id": "s1",
        "question": "Warum?",
        "gold_passages": ["Weil"],
        "document_text": doc_text,
        "pages": [asdict(p) for p in pages],
        "url": "url2",
    }
    assert entry == expected


# --- Integrationstest für main() ---


def test_main_smoke_test(mocker: MockerFixture):
    # --- 1. Mocks für die Pfad-Erstellung ---
    mock_now_obj = mocker.Mock()
    mock_now_obj.strftime.return_value = "2025-10-23"
    # Stelle sicher, dass der Pfad zu datetime korrekt ist
    mock_datetime_class = mocker.patch("preprocessor.main.datetime")
    mock_datetime_class.now.return_value = mock_now_obj

    # --- 2. Erwarteten Pfad berechnen ---
    # Der Pfad wird mit 'limit=1' erwartet, da wir main(limit=1)
    expected_path = "./data/preprocessed/preprocessed_2025-10-23_limit_1.jsonl"

    # --- 3. Mocks für Datenverarbeitung ---
    mock_example = {
        "sample_id": "s1_main",
        "ambiguous_question": " Test-Frage ",
        "qa_pairs": [{"short_answers": ["Test-Antwort"]}],
        "wikipages": [{"url": "http://mock-url.com"}],
    }

    # Mockt load_dataset, das jetzt in main() aufgerufen wird
    mocker.patch("preprocessor.main.load_dataset", return_value=[mock_example])

    # Mock Wikipedia-Download
    mock_page_data = PageData(
        url="http://mock-url.com",
        title="Mock Titel",
        text="Mock Textinhalt",
        categories=["Mock Cat"],
    )
    mocker.patch("preprocessor.main.fetch_and_parse_page", return_value=mock_page_data)

    # Mock das Öffnen und Schreiben von Dateien
    mock_file = mocker.mock_open()
    mocker.patch("builtins.open", mock_file)

    # Mock os.makedirs
    mocker.patch("os.makedirs")

    # --- 4. Ausführung ---
    # Rufe main() mit dem simulierten 'limit'-Argument auf
    main(limit=1)

    # --- 5. Assertions (Überprüfungen) ---
    # Diese Assertion prüft den korrekt generierten Dateinamen
    mock_file.assert_called_once_with(expected_path, "w", encoding="utf-8")

    # (Restliche Assertions bleiben gleich)
    handle = mock_file()
    handle.write.assert_called_once()
    written_json_str = handle.write.call_args[0][0].strip()
    written_data = json.loads(written_json_str)

    assert written_data["sample_id"] == "s1_main"
    assert written_data["question"] == "Test-Frage"
