#!/usr/bin/env python3
import json

input_file = "data/preprocessed/preprocessed_2025-11-03_all_categorized.jsonl"

try:
    with open(input_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                entry = json.loads(line)
                print("First entry keys:", list(entry.keys()))
                print("\nFirst entry (truncated):")
                for key, value in entry.items():
                    if isinstance(value, str) and len(str(value)) > 100:
                        print(f"  {key}: {str(value)[:100]}...")
                    else:
                        print(f"  {key}: {value}")
                break
except Exception as e:
    print(f"Error: {e}")
