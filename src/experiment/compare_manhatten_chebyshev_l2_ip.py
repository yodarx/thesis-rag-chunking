import csv
import gc
import json
import os
import random

import numpy as np
import torch

# Prevent OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss

# Force single thread for Faiss to prevent contention
faiss.omp_set_num_threads(1)

from sentence_transformers import SentenceTransformer

# --- PROGRESS BAR SETUP ---
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


    def tqdm(iterable, **kwargs):
        return iterable


def tprint(msg):
    if TQDM_AVAILABLE:
        tqdm.write(msg)
    else:
        print(msg)


# --- CONFIGURATION ---
CHUNKS_ROOT = "../../data/chunks"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
SAMPLE_SIZE = 100
CSV_OUTPUT_FILE = "chunk_density_comparison.csv"
BATCH_SIZE = 32  # Adjustable based on GPU VRAM


def get_device():
    # Prioritize CUDA, then MPS (Mac), then CPU
    if torch.cuda.is_available():
        tprint("🚀 Using NVIDIA CUDA")
        return "cuda"
    elif torch.backends.mps.is_available():
        tprint("🚀 Using Apple M1/M2 GPU (MPS)")
        return "mps"
    else:
        tprint("🐢 Using CPU")
        return "cpu"


def load_chunks_text_only(filepath):
    """Loads chunks to get just the text content."""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        tprint(f"    ⚠️ Failed to read file: {e}")
        return []

    texts = []
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        for item in data:
            content = item.get("page_content") or item.get("text") or item.get("content") or ""
            if content.strip():
                texts.append(content)
    elif isinstance(data, list) and isinstance(data[0], str):
        texts = [t for t in data if t.strip()]

    return texts


def process_single_experiment(experiment_name, chunks_file, model, results_accumulator):
    tprint(f"\n🚀 Experiment: {experiment_name}")

    # 1. Load Chunks
    chunk_texts = load_chunks_text_only(chunks_file)
    count = len(chunk_texts)

    if count == 0:
        tprint(f"❌ No chunks found in {experiment_name}")
        return

    tprint(f"   🔹 Vectorizing {count} chunks (Batch Size: {BATCH_SIZE})...")

    # 2. Vectorize ALL chunks (The Corpus)
    # CHANGED: show_progress_bar=True to identify hangs
    try:
        corpus_embeddings = model.encode(
            chunk_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        tprint(f"    ❌ Encoding Error: {e}")
        return

    # Free text memory immediately
    del chunk_texts
    gc.collect()

    # Safety cast
    corpus_embeddings = np.ascontiguousarray(corpus_embeddings, dtype=np.float32)
    d = corpus_embeddings.shape[1]

    # 3. Select Samples
    if count > SAMPLE_SIZE:
        query_indices = random.sample(range(count), SAMPLE_SIZE)
    else:
        query_indices = list(range(count))

    query_vecs = corpus_embeddings[query_indices]
    query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)

    # 4. Metric Loop
    def get_index(name, dim):
        if name == "IP (Cosine)": return faiss.IndexFlatIP(dim)
        if name == "L2 (Euclidean)": return faiss.IndexFlatL2(dim)
        if name == "L1 (Manhattan)": return faiss.IndexFlat(dim, faiss.METRIC_L1)
        if name == "Linf (Chebyshev)": return faiss.IndexFlat(dim, faiss.METRIC_Linf)
        return None

    metric_names = ["IP (Cosine)", "L2 (Euclidean)", "L1 (Manhattan)", "Linf (Chebyshev)"]

    # Calculate one by one to save memory
    for name in metric_names:
        try:
            index = get_index(name, d)
            index.add(corpus_embeddings)

            k = 2 if count >= 2 else 1
            distances, retrieved_ids = index.search(query_vecs, k)

            self_hits = 0
            neighbor_dists = []

            for i, q_idx in enumerate(query_indices):
                # Self retrieval check
                if len(retrieved_ids[i]) > 0 and retrieved_ids[i][0] == q_idx:
                    self_hits += 1

                # Neighbor distance
                if len(distances[i]) >= 2:
                    neighbor_dists.append(distances[i][1])
                elif len(distances[i]) == 1 and retrieved_ids[i][0] != q_idx:
                    neighbor_dists.append(distances[i][0])

            avg_neighbor_dist = float(np.mean(neighbor_dists)) if neighbor_dists else 0.0
            self_retrieval_accuracy = self_hits / len(query_indices) if query_indices else 0

            row = {
                "Experiment": experiment_name,
                "Chunk_Count": count,
                "Distance_Metric": name,
                "Self_Retrieval_Acc": round(self_retrieval_accuracy, 4),
                "Avg_Neighbor_Dist": round(avg_neighbor_dist, 4)
            }
            results_accumulator.append(row)

            # Clean up index
            del index
            gc.collect()

        except Exception as e:
            tprint(f"    ⚠️ Error calculating {name}: {e}")

    # Final cleanup
    del corpus_embeddings
    del query_vecs
    gc.collect()


def main():
    if not os.path.exists(CHUNKS_ROOT):
        tprint(f"❌ Chunks root not found at {CHUNKS_ROOT}")
        return

    experiments = [d for d in os.listdir(CHUNKS_ROOT) if os.path.isdir(os.path.join(CHUNKS_ROOT, d))]
    experiments.sort()

    if not experiments:
        tprint("❌ No experiment folders found.")
        return

    device = get_device()
    tprint(f"🔹 Found {len(experiments)} experiments. Loading Model {MODEL_NAME} onto {device}...")

    model = SentenceTransformer(MODEL_NAME, device=device)
    all_results = []

    for exp in tqdm(experiments, desc="Batch Progress", unit="exp"):
        chunks_file = os.path.join(CHUNKS_ROOT, exp, "chunks.json")
        if os.path.exists(chunks_file):
            process_single_experiment(exp, chunks_file, model, all_results)

    if all_results:
        tprint(f"\n💾 Saving density analysis to {CSV_OUTPUT_FILE}...")
        fieldnames = ["Experiment", "Chunk_Count", "Distance_Metric", "Self_Retrieval_Acc", "Avg_Neighbor_Dist"]
        with open(CSV_OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        tprint("✅ Done.")
    else:
        tprint("⚠️ No results gathered.")


if __name__ == "__main__":
    main()
