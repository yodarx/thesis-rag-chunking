import json
from typing import Any


def load_asqa_dataset(filepath: str | None, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        with open(filepath, encoding="utf-8") as f:
            all_data: list[dict[str, Any]] = [json.loads(line) for line in f]
        return all_data[:limit] if limit else all_data
    except FileNotFoundError:
        return []
