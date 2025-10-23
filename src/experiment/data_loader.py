import json
from typing import Any


def load_asqa_dataset(filepath: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Lädt das vorverarbeitete ASQA-Dataset aus einer JSONL-Datei."""
    try:
        with open(filepath, encoding="utf-8") as f:
            all_data = [json.loads(line) for line in f]

        if limit:
            return all_data[:limit]
        return all_data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        return []
