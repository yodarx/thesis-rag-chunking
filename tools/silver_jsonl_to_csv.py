"""Convert thesis-rag-chunking silver JSONL files to CSV.

This script is intentionally conservative about what it reads from `metadata`.
In particular, it *ignores* `metadata.source_chunks` because it may contain raw
Wikipedia strings or otherwise problematic content that you don't want in the
CSV output.

Usage:
  python tools/silver_jsonl_to_csv.py --input data/silver/file.jsonl --output out.csv

The input is expected to be JSON Lines (one JSON object per line). The attached
example in this repo may be pretty-printed over multiple lines; this script
supports both JSONL (preferred) and multi-line JSON objects (fallback).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Iterator
from typing import Any


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _as_list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_coerce_str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _iter_json_objects(path: str, encoding: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (line_number, obj) from a JSONL-like file.

    Supports:
      - Proper JSONL: one JSON object per line.
      - Fallback: pretty-printed JSON objects spanning multiple lines.

    The yielded `line_number` is the starting 1-based line number of the object.
    """

    decoder = json.JSONDecoder()

    with open(path, encoding=encoding) as f:
        # Fast path: try per-line JSON (true JSONL)
        start_line = 1
        buf: list[str] = []
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # If we have no buffer, try to parse this line as a full JSON object.
            if not buf:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield i, obj
                        continue
                    # if it's not a dict, fall through to buffered parse
                except json.JSONDecodeError:
                    pass
                start_line = i

            buf.append(raw_line)

            joined = "".join(buf)
            try:
                # Attempt to decode a single JSON object from the buffer.
                obj, end_idx = decoder.raw_decode(joined.lstrip())
                if not isinstance(obj, dict):
                    raise json.JSONDecodeError("Top-level JSON value is not an object", joined, 0)

                # Ensure remaining content is only whitespace (avoid concatenated objects).
                remainder = joined[end_idx:].strip()
                if remainder:
                    # If multiple objects got concatenated without newlines, keep it strict.
                    raise json.JSONDecodeError("Extra data after JSON object", joined, end_idx)

                yield start_line, obj
                buf = []
            except json.JSONDecodeError:
                # Keep buffering.
                continue

        # Trailing buffer that never parsed
        if buf:
            raise ValueError(
                f"Input ended with an incomplete JSON object starting at line {start_line}. "
                "If this file is supposed to be JSONL, ensure each line is valid JSON."
            )


def _flatten_record(
    obj: dict[str, Any],
    *,
    max_gold_cols: int,
    input_file: str | None,
    line_number: int | None,
) -> dict[str, Any]:
    gold_passages = _as_list_of_strings(obj.get("gold_passages"))

    metadata = obj.get("metadata")
    bridge_entity = ""
    if isinstance(metadata, dict):
        bridge_entity = _coerce_str(metadata.get("bridge_entity"))
        # Explicitly ignore metadata.source_chunks and any other metadata keys.

    row: dict[str, Any] = {
        "sample_id": _coerce_str(obj.get("sample_id")),
        "question": _coerce_str(obj.get("question")),
        "answer": _coerce_str(obj.get("answer")),
        "category": _coerce_str(obj.get("category")),
        "difficulty": _coerce_str(obj.get("difficulty")),
        "gold_passages_count": len(gold_passages),
        "gold_passages_json": json.dumps(gold_passages, ensure_ascii=False),
        "metadata_bridge_entity": bridge_entity,
    }

    for idx in range(max_gold_cols):
        row[f"gold_passage_{idx + 1}"] = gold_passages[idx] if idx < len(gold_passages) else ""

    if input_file is not None:
        row["input_file"] = input_file
    if line_number is not None:
        row["line_number"] = line_number

    return row


def convert_jsonl_to_csv(
    *,
    input_path: str,
    output_path: str,
    encoding: str = "utf-8",
    delimiter: str = ",",
    max_gold_cols: int = 3,
    include_input_file: bool = True,
    include_line_number: bool = True,
) -> int:
    basename = os.path.basename(input_path) if include_input_file else None

    rows: list[dict[str, Any]] = []
    for ln, obj in _iter_json_objects(input_path, encoding=encoding):
        row = _flatten_record(
            obj,
            max_gold_cols=max_gold_cols,
            input_file=basename if include_input_file else None,
            line_number=ln if include_line_number else None,
        )
        rows.append(row)

    # Stable column order
    fieldnames: list[str] = [
        "sample_id",
        "question",
        "answer",
        "category",
        "difficulty",
        "gold_passages_count",
        "gold_passages_json",
        "metadata_bridge_entity",
    ]
    fieldnames.extend([f"gold_passage_{i + 1}" for i in range(max_gold_cols)])
    if include_input_file:
        fieldnames.append("input_file")
    if include_line_number:
        fieldnames.append("line_number")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return len(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert silver JSONL to CSV (excluding metadata.source_chunks)."
    )
    parser.add_argument("--input", required=True, help="Path to input .jsonl")
    parser.add_argument("--output", required=True, help="Path to output .csv")
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    parser.add_argument(
        "--delimiter", default=",", help="CSV delimiter (default: ,). Use \\t for TSV"
    )
    parser.add_argument(
        "--max-gold-cols", type=int, default=3, help="How many gold_passage_* columns to emit"
    )
    parser.add_argument(
        "--no-input-file",
        action="store_true",
        help="Don't include the input_file column",
    )
    parser.add_argument(
        "--no-line-number",
        action="store_true",
        help="Don't include the line_number column",
    )

    args = parser.parse_args(argv)

    delimiter = args.delimiter
    if delimiter == "\\t":
        delimiter = "\t"

    n = convert_jsonl_to_csv(
        input_path=args.input,
        output_path=args.output,
        encoding=args.encoding,
        delimiter=delimiter,
        max_gold_cols=args.max_gold_cols,
        include_input_file=not args.no_input_file,
        include_line_number=not args.no_line_number,
    )

    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
