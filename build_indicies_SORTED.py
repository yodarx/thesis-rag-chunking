import argparse
import gc  # WICHTIG: Für Garbage Collection
import json
import os
import queue
import threading
import time
from datetime import datetime

from build_chunks import generate_chunks

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import faiss
import numpy as np
import torch
from tqdm import tqdm

# Imports aus deinem Projekt
from src.experiment.data_loader import load_asqa_dataset
from src.vectorizer.vectorizer import Vectorizer


# --- HELPER FUNCTIONS ---
def create_index_name(exp_name: str, model_name: str) -> str:
    return f"{exp_name}_{model_name.replace('/', '_')}"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def calculate_dynamic_batch_size(current_max_char_len: int, model_name: str) -> int:
    """
    Optimiert für Nvidia L4 (24GB).
    KONSERVATIVE EINSTELLUNG FÜR BGE-LARGE.
    """
    name = model_name.lower()

    # Schätzung: 1 Token ~ 3.0 Zeichen
    est_tokens = max(current_max_char_len / 3.0, 1.0)

    if "minilm" in name:
        token_budget = 6_000_000
    elif "large" in name:
        token_budget = 250_000
    else:
        # Base Modelle
        token_budget = 1_200_000

    # 2. PENALTY (Quadratischer Anstieg bei Attention)
    length_penalty = 1.0
    if est_tokens > 512:
        length_penalty = 1 + (est_tokens / 1024.0)

    adjusted_budget = token_budget / length_penalty
    optimal_bs = int(adjusted_budget / est_tokens)

    # 3. HARD LIMITS
    max_limit = 64_000 if "minilm" in name else 16_000

    optimal_bs = max(optimal_bs, 16)  # Minimum safety
    optimal_bs = min(optimal_bs, max_limit)

    # --- SPECIAL CASE: LARGE MODELS ---
    if "large" in name:
        optimal_bs = min(optimal_bs, 512)

        if est_tokens > 200:
            optimal_bs = min(optimal_bs, 256)

        if est_tokens > 450:
            optimal_bs = min(optimal_bs, 128)

    return int(optimal_bs)


def save_artifacts(index, index_dir, chunks, sorted_filename, build_time):
    os.makedirs(index_dir, exist_ok=True)

    print(f"💾 Saving FAISS Index to {index_dir}...")
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    metadata = {
        "indexing_duration": build_time,
        "num_chunks": len(chunks),
        "linked_cache_file": sorted_filename,
        "timestamp": datetime.now().isoformat(),
        "faiss_ntotal": index.ntotal,
        "optimization": "sorted_dynamic_batching_fp16",
    }
    with open(os.path.join(index_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# --- THREADED BUILDER ---
def build_index_dynamic(chunks: list[str], vectorizer: Vectorizer, model_name: str):
    print("🔹 Initializing FAISS (FP16)...")

    # Dummy Call für Dimension & Model Warmup
    dummy = vectorizer.embed_documents(chunks[:1], batch_size=1)
    d = dummy.shape[1] if hasattr(dummy, "shape") else len(dummy[0])

    index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
    faiss.omp_set_num_threads(32)

    result_queue = queue.Queue(maxsize=5)

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

    loop_block_size = 1_000
    total = len(chunks)

    pbar = tqdm(total=total, desc="🚀 Init...", unit="chunk")
    start = time.time()

    try:
        for i in range(0, total, loop_block_size):
            end = min(i + loop_block_size, total)
            batch_text = chunks[i:end]

            # Da sortiert: Der letzte Chunk ist der längste im aktuellen Block!
            longest_char_len = len(batch_text[-1])

            # Berechne optimale Batch Size
            current_bs = calculate_dynamic_batch_size(longest_char_len, model_name)

            pbar.set_description(
                f"🚀 Speed | MaxLen: {longest_char_len:3} | BatchSize: {current_bs:4}"
            )

            # Vektorisieren
            embeddings = vectorizer.embed_documents(batch_text, batch_size=current_bs,convert_to_numpy=True)

            # WICHTIG: Sofort von GPU lösen falls Tensor
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Ab in die Queue
            result_queue.put(embeddings)
            pbar.update(len(batch_text))

            # Cleanup
            del embeddings
            # WICHTIG: Regelmäßiger Cleanup
            if i % (loop_block_size * 2) == 0:
                gc.collect()
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n⚠️ Abbruch durch User! Index wird NICHT gespeichert.")
        result_queue.put(None)
        t.join()
        return None, 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        result_queue.put(None)
        t.join()
        raise e

    # Clean finish
    result_queue.put(None)
    t.join()
    pbar.close()

    return index, time.time() - start


def process_experiment(exp, config, dataset, vec, out_dir, cache_dir):
    name = exp["name"]
    model_name = config["embedding_model"]

    # 1. Chunks laden/erstellen
    chunks = generate_chunks(exp, dataset, vec, cache_dir)

    # Cache Name rebuilding for sorting reference
    id_str = f"{name}_{exp['function']}"
    cache_name = f"{name}_{id_str}_chunks.json"

    # 2. SORTING
    print(f"⚡ Sorting {len(chunks)} chunks (Global Sort)...")
    chunks.sort(key=len)

    sorted_name = cache_name.replace(".json", "_SORTED.json")
    if not os.path.exists(os.path.join(cache_dir, sorted_name)):
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

    # Cleanup nach Experiment
    del index
    del chunks
    gc.collect()
    torch.cuda.empty_cache()


def main():
    torch.set_float32_matmul_precision("medium")
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
