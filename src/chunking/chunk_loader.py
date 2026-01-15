import json
import os
from typing import List, Dict, Any, Callable
from tqdm import tqdm

from src.chunking.chunk_fixed import chunk_fixed_size
from src.chunking.chunk_recursive import chunk_recursive
from src.chunking.chunk_semantic import chunk_semantic
from src.chunking.chunk_sentence import chunk_by_sentence
from src.vectorizer.vectorizer import Vectorizer


def get_chunking_function(name: str) -> Callable[..., List[str]]:
    chunk_functions = {
        "chunk_fixed_size": chunk_fixed_size,
        "chunk_by_sentence": chunk_by_sentence,
        "chunk_recursive": chunk_recursive,
        "chunk_semantic": chunk_semantic,
    }
    return chunk_functions[name]


def load_chunks(
    exp_config: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    vectorizer: Vectorizer,
    cache_dir: str
) -> List[str]:
    """
    Loads chunks from cache or generates them if not present.
    """
    name = exp_config["name"]
    func_name = exp_config["function"]

    # Consistent with build_indicies_SORTED.py logic
    id_str = f"{name}_{func_name}"
    cache_name = f"{name}_{id_str}_chunks.json"
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        print(f"[{name}] Lade Chunks aus Cache...")
        with open(cache_path) as f:
            return json.load(f)

    print(f"[{name}] ⚠️ Cache Miss! Generiere Chunks...")
    chunk_func = get_chunking_function(func_name)
    params = exp_config.get("params", {})

    # Copy params and inject vectorizer if needed
    call_params = params.copy()
    if func_name == "chunk_semantic":
        call_params["chunking_embeddings"] = vectorizer
        if "batch_size" not in call_params:
            call_params["batch_size"] = 1024

    chunks = []

    for d in tqdm(dataset, desc="Chunking"):
        text = d.get("document_text", "")
        chunks.extend(chunk_func(text, **call_params))

    # Ensure cache dir exists
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(chunks, f)

    return chunks

