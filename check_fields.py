#!/usr/bin/env python3
"""Minimal test to verify the document_text field is correctly identified"""

import json

# Test by reading just the first line directly
input_file = "/Users/jeremy/IdeaProjects/thesis-rag-chunking/data/preprocessed/preprocessed_2025-11-03_all_categorized.jsonl"

try:
    with open(input_file, encoding="utf-8") as f:
        first_line = f.readline()
        if first_line.strip():
            entry = json.loads(first_line)
            print("Keys in first entry:")
            print(list(entry.keys()))
            print(
                "\nField that will be loaded: 'document_text' in entry?", "document_text" in entry
            )
            if "document_text" in entry:
                text_len = len(entry["document_text"])
                print(f"✓ document_text field found with {text_len} characters")
                print(f"First 200 chars: {entry['document_text'][:200]}")
            else:
                print("✗ document_text field NOT found")
                for key in entry:
                    if key in ["text", "content", "document"]:
                        print(f"  But found alternative field: {key}")
except Exception as e:
    print(f"Error: {e}")
