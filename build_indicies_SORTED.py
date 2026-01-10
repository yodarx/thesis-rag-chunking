import argparse
import json
import os
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime

import faiss
import torch
from tqdm import tqdm

# Imports aus deinem Projekt
from src.chunking.chunk_fixed import chunk_fixed_size
from src.chunking.chunk_recursive import chunk_recursive
from src.chunking.chunk_semantic import chunk_semantic
from src.chunking.chunk_sentence import chunk_by_sentence
from src.experiment.data_loader import load_asqa_dataset
from src.vectorizer.vectorizer import Vectorizer

# --- CONFIGURATION ---
# Verhindert Deadlocks bei High-Performance Loops
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- HELPER FUNCTIONS ---
def get_chunking_function(name: str) -> Callable[..., list[str]]:
    chunk_functions = {
        "chunk_fixed_size": chunk_fixed_size,
        "chunk_by_sentence": chunk_by_sentence,
        "chunk_recursive": chunk_recursive,
        "chunk_semantic": chunk_semantic,
    }
    return chunk_functions[name]


def create_index_name(exp_name: str, model_name: str) -> str:
    return f"{exp_name}_{model_name.replace('/', '_')}"


def load_config(config_path: str) -> dict:
    with open(config_path) as f: return json.load(f)


# --- DYNAMIC BATCH SIZE LOGIC ---
def calculate_dynamic_batch_size(current_max_char_len: int, model_name: str) -> int:
    """
    Optimiert für Nvidia L4 (24GB).
    Fix: Verhindert Tuple-Fehler durch explizites Casting.
    """
    name = model_name.lower()

    # Schätzung: 1 Token ~ 3.0 Zeichen
    est_tokens = max(current_max_char_len / 3.0, 1.0)

    # 1. BUDGETS
    if "minilm" in name:
        token_budget = 6_000_000
    elif "large" in name:
        token_budget = 600_000
    else:
        token_budget = 1_500_000

    # 2. PENALTY
    length_penalty = 1.0
    if est_tokens > 512:
        length_penalty = 1 + (est_tokens / 1024.0)

    adjusted_budget = token_budget / length_penalty

    # Hier sicherstellen, dass es wirklich ein INT ist (kein Tuple)
    optimal_bs = int(adjusted_budget / est_tokens)

    # 3. HARD LIMITS
    # Achte darauf, dass hier am Ende der Zeilen KEINE Kommas stehen!
    max_limit = 64_000 if "minilm" in name else 32_000

    # Safety Clamps
    optimal_bs = max(optimal_bs, 64)
    optimal_bs = min(optimal_bs, max_limit)

    # Large Model Special Case
    if "large" in name:
        optimal_bs = min(optimal_bs, 2048)
        if est_tokens > 512:
            optimal_bs = min(optimal_bs, 256)

    return int(optimal_bs)


def save_artifacts(index, index_dir, chunks, sorted_filename, build_time):
    os.makedirs(index_dir, exist_ok=True)

    print(f"💾 Saving FAISS Index to {index_dir}...")
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    metadata = {
        "indexing_duration": build_time,
        "num_chunks": len(chunks),
        "linked_cache_file": sorted_filename,  # WICHTIG für Retrieval!
        "timestamp": datetime.now().isoformat(),
        "faiss_ntotal": index.ntotal,
        "optimization": "sorted_dynamic_batching_fp16"
    }
    with open(os.path.join(index_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# --- THREADED BUILDER ---
def build_index_dynamic(chunks: list[str], vectorizer: Vectorizer, model_name: str):
    # 1. Init FAISS (FP16)
    print("🔹 Initializing FAISS (FP16)...")
    d = vectorizer.embed_documents(chunks[:1], batch_size=1).shape[1]
    index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
    faiss.omp_set_num_threads(32)  # Nutze alle CPU Cores

    result_queue = queue.Queue(maxsize=500)

    def worker():
        while True:
            emb = result_queue.get()
            if emb is None:
                result_queue.task_done()
                break
            index.add(emb)
            result_queue.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    LOOP_BLOCK_SIZE = 100_000
    total = len(chunks)

    pbar = tqdm(total=total, desc="🚀 Init...", unit="chunk")
    start = time.time()

    try:
        for i in range(0, total, LOOP_BLOCK_SIZE):
            end = min(i + LOOP_BLOCK_SIZE, total)
            batch_text = chunks[i:end]

            # Da sortiert: Der letzte Chunk ist der längste im aktuellen Block!
            longest_char_len = len(batch_text[-1])

            # Berechne optimale Batch Size für diesen Abschnitt
            current_bs = calculate_dynamic_batch_size(longest_char_len, model_name)

            # Update Anzeige (Live-Stats)
            pbar.set_description(f"🚀 Speed | MaxLen: {longest_char_len:3} | BatchSize: {current_bs:5}")

            # Vektorisieren
            embeddings = vectorizer.embed_documents(batch_text, batch_size=current_bs)

            # Ab in die Queue
            result_queue.put(embeddings)
            pbar.update(len(batch_text))
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n⚠️ Abbruch durch User! Index wird NICHT gespeichert.")
        result_queue.put(None)
        t.join()
        return None, 0

    # Clean finish
    result_queue.put(None)
    t.join()
    pbar.close()

    return index, time.time() - start


# --- MAIN LOGIC ---
def process_experiment(exp, config, dataset, vec, out_dir, cache_dir):
    name = exp["name"]
    model_name = config["embedding_model"]

    # 1. Chunks laden/erstellen
    id_str = f"{name}_{exp['function']}"
    cache_name = f"{name}_{id_str}_chunks.json"
    cache_path = os.path.join(cache_dir, cache_name)

    chunks = []
    if os.path.exists(cache_path):
        print(f"[{name}] Lade Chunks aus Cache...")
        with open(cache_path) as f:
            chunks = json.load(f)
    else:
        print(f"[{name}] ⚠️ Cache Miss! Generiere Chunks...")
        chunk_func = get_chunking_function(exp["function"])
        for d in tqdm(dataset, desc="Chunking"):
            chunks.extend(chunk_func(d.get("document_text", ""), **exp["params"]))
        with open(cache_path, "w") as f:
            json.dump(chunks, f)

    # 2. SORTING (Der Speed Boost)
    print(f"⚡ Sorting {len(chunks)} chunks (Global Sort)...")
    chunks.sort(key=len)

    # Sortierte Map speichern (WICHTIG!)
    sorted_name = cache_name.replace(".json", "_SORTED.json")
    print(f"💾 Saving sorted map to {sorted_name}...")
    with open(os.path.join(cache_dir, sorted_name), "w") as f:
        json.dump(chunks, f)

    # 3. Index bauen
    index_dir = os.path.join(out_dir, create_index_name(name, model_name))
    if os.path.exists(os.path.join(index_dir, "index.faiss")):
        print(f"✅ Index {index_dir} existiert bereits. Skip.")
        return

    index, duration = build_index_dynamic(chunks, vec, model_name)

    if index:
        save_artifacts(index, index_dir, chunks, sorted_name, duration)


def main():
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="data/indices")
    parser.add_argument("--cache-dir", default="data/chunks")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_asqa_dataset(config.get("input_file"), config.get("limit"))
    vec = Vectorizer.from_model_name(config["embedding_model"])

    for exp in config["experiments"]:
        process_experiment(exp, config, dataset, vec, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
