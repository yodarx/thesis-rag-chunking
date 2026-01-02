#!/usr/bin/env python3
"""Quick test to verify document loading works"""

import sys

sys.path.insert(0, "/Users/jeremy/IdeaProjects/thesis-rag-chunking")

from src.experiment.silver_generation import load_documents_from_input_file

# Test with the actual file
input_file = "data/preprocessed/preprocessed_2025-11-03_all_categorized.jsonl"
documents = load_documents_from_input_file(input_file)

print(f"Successfully loaded {len(documents)} documents")
if documents:
    print(f"First document length: {len(documents[0])} characters")
    print(f"First 100 chars of first document: {documents[0][:100]}...")
else:
    print("ERROR: No documents loaded!")
    sys.exit(1)
