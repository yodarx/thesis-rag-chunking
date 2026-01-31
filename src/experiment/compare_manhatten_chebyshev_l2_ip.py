import json
import os
import sys
import random
import csv
import torch
import numpy as np

# Prevent OpenMP conflicts on macOS which can cause SEGFAULTS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
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
SAMPLE_SIZE = 5  # Number of chunks to used as queries for the density test
CSV_OUTPUT_FILE = "chunk_density_comparison.csv"


def get_device():
    if torch.cuda.is_available():
        tprint("🚀 Using NVIDIA CUDA")
        return "cuda"
    else:
        tprint("🐢 Using CPU")
        return "cpu"


def load_chunks_text_only(filepath):
    """Loads chunks to get just the text content, ignoring broken IDs."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

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

    if not chunk_texts:
        tprint(f"❌ No chunks found in {experiment_name}")
        return

    # 2. Vectorize ALL chunks (The Corpus)
    count = len(chunk_texts)
    # Generate embeddings
    corpus_embeddings = model.encode(chunk_texts, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)

    # CRITICAL FIX for SIGSEGV: Ensure arrays are C-contiguous float32 for Faiss
    corpus_embeddings = np.ascontiguousarray(corpus_embeddings, dtype=np.float32)

    d = corpus_embeddings.shape[1]

    # 3. Select Samples for Self-Retrieval Test
    if count > SAMPLE_SIZE:
        query_indices = random.sample(range(count), SAMPLE_SIZE)
    else:
        query_indices = list(range(count))

    # The queries are the chunks themselves
    # CRITICAL FIX: Ensure query slice is also contiguous
    query_vecs = corpus_embeddings[query_indices]
    query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)

    # 4. Define Index Factories
    # We do not reuse indices to avoid state corruption on reset()
    def get_index(name, dim):
        if name == "IP (Cosine)": return faiss.IndexFlatIP(dim)
        if name == "L2 (Euclidean)": return faiss.IndexFlatL2(dim)
        if name == "L1 (Manhattan)": return faiss.IndexFlat(dim, faiss.METRIC_L1)
        if name == "Linf (Chebyshev)": return faiss.IndexFlat(dim, faiss.METRIC_Linf)
        return None

    metric_names = ["IP (Cosine)", "L2 (Euclidean)", "L1 (Manhattan)", "Linf (Chebyshev)"]

    tprint(f"   🔹 Analyzing Vector Space Density (Sample N={len(query_indices)})...")

    for name in metric_names:
        # Create fresh index
        index = get_index(name, d)
        index.add(corpus_embeddings)

        # Search for Top-2 (Target is Self + Nearest Neighbor)
        k = 2
        if count < 2:
            distances, retrieved_ids = index.search(query_vecs, 1)
        else:
            distances, retrieved_ids = index.search(query_vecs, k)

        # METRIC CALCULATIONS
        self_hits = 0
        neighbor_dists = []

        for i, q_idx in enumerate(query_indices):
            # 1. Self-Retrieval Check: Is the top result the chunk itself?
            if len(retrieved_ids[i]) > 0:
                # Top result matches query index
                if retrieved_ids[i][0] == q_idx:
                    self_hits += 1

            # 2. Neighbor Distance Analysis
            # If we found ourselves at 0, the neighbor is at 1.
            if len(distances[i]) >= 2:
                neighbor_dists.append(distances[i][1])
            elif len(distances[i]) == 1 and retrieved_ids[i][0] != q_idx:
                # If we didn't find ourselves, the one we found is a neighbor
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

    # --- SAVE TO CSV ---
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
