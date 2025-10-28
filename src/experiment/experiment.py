from typing import Any

from chunking.chunk_fixed import chunk_fixed_size
from chunking.chunk_recursive import chunk_recursive
from chunking.chunk_semantic import chunk_semantic
from chunking.chunk_sentence import chunk_by_sentence


def get_experiments() -> list[dict[str, Any]]:
    """Definiert alle Chunking-Strategien, die getestet werden sollen."""
    return [
        {
            "name": "fixed_512_50",
            "function": chunk_fixed_size,
            "params": {"chunk_size": 512, "chunk_overlap": 50},
        },
        {
            "name": "fixed_256_25",
            "function": chunk_fixed_size,
            "params": {"chunk_size": 256, "chunk_overlap": 25},
        },
        {
            "name": "sentence_s3",
            "function": chunk_by_sentence,
            "params": {"sentences_per_chunk": 3},
        },
        {
            "name": "sentence_s5",
            "function": chunk_by_sentence,
            "params": {"sentences_per_chunk": 5},
        },
        {
            "name": "recursive_512_50",
            "function": chunk_recursive,
            "params": {"chunk_size": 512, "chunk_overlap": 50},
        },
        {
            "name": "recursive_256_25",
            "function": chunk_recursive,
            "params": {"chunk_size": 256, "chunk_overlap": 25},
        },
        {
            "name": "semantic_t0.7",
            "function": chunk_semantic,
            "params": {"similarity_threshold": 0.7},
        },
        {
            "name": "semantic_t0.85",
            "function": chunk_semantic,
            "params": {"similarity_threshold": 0.85},
        },
    ]
