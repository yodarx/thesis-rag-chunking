import os
import sys

# --- 1. PATH SETUP (Must be first) ---
# Add the parent directory (src/) to sys.path so we can import 'evaluation.py'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Points to 'src/'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- 2. IMPORT CUSTOM MODULES ---
try:
    # Attempt to import based on file location src/evaluation.py
    from evaluation import calculate_metrics
except ImportError:
    try:
        # Fallback if it is inside a package src/evaluation/evaluation.py
        from evaluation.evaluation import calculate_metrics
    except ImportError:
        print("❌ Error: Could not import 'calculate_metrics'. Ensure 'evaluation.py' is in the 'src' folder.")
        sys.exit(1)

# --- 3. STANDARD & THIRD PARTY IMPORTS ---
import csv
import gc
import json

import numpy as np
import torch

# --- 4. ENVIRONMENT SAFETY (MacOS/Faiss) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss

# Force single thread for Faiss to prevent OpenMP crashes on Apple Silicon
faiss.omp_set_num_threads(1)

from sentence_transformers import SentenceTransformer

# --- 5. PROGRESS BAR SETUP ---
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


# --- 6. CONFIGURATION ---
CHUNKS_ROOT = "../../data/chunks"
GOLD_DATA_PATH = "../../data/preprocessed/gold.jsonl"
INDICES_ROOT = "../../data/indices/compare_distance"
CSV_OUTPUT_FILE = "distance_metric_full_evaluation.csv"
MODEL_NAME = "all-MiniLM-L6-v2"  # Preserved from your active file
BATCH_SIZE = 32
TOP_K = 10


def get_device():
    if torch.cuda.is_available():
        tprint("🚀 Using NVIDIA CUDA")
        return "cuda"
    elif torch.backends.mps.is_available():
        tprint("🚀 Using Apple M1/M2 GPU (MPS)")
        return "mps"
    else:
        tprint("🐢 Using CPU")
        return "cpu"


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_gold_data(filepath):
    """Loads dictionary of questions and expected gold passages."""
    data = []
    if not os.path.exists(filepath):
        tprint(f"❌ Gold data not found at {filepath}")
        return []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                row = json.loads(line)
                q = row.get('question')
                g = row.get('gold_passages')

                # Normalize gold to list
                if isinstance(g, str): g = [g]

                if q and g:
                    data.append({'question': q, 'gold_passages': g})
            except:
                continue
    return data


def load_chunks_text_only(filepath):
    """Loads just the text content of chunks for evaluation matching."""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except:
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


def get_or_create_embeddings(experiment_name, chunk_texts, model):
    """
    Checks if embeddings already exist on disk.
    If yes, loads them. If no, encodes and saves them.
    Explicitly ensures L2 normalization for metric consistency.
    """
    ensure_directory(INDICES_ROOT)
    cache_path = os.path.join(INDICES_ROOT, f"{experiment_name}_embeddings.npy")
    embeddings = None

    if os.path.exists(cache_path):
        tprint(f"   💾 Loading cached embeddings from {cache_path}...")
        try:
            embeddings = np.load(cache_path)
        except Exception as e:
            tprint(f"   ⚠️ Cache corrupt, rebuilding: {e}")
            embeddings = None

    if embeddings is None:
        tprint(f"   ⚙️  Encoding {len(chunk_texts)} chunks...")
        embeddings = model.encode(
            chunk_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=BATCH_SIZE
        )
        # Save raw embeddings
        np.save(cache_path, embeddings)

    # Ensure float32 and contiguous
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # CRITICAL: Enforce L2 normalization so dot-product == cosine similarity
    # This ensures ranking equivalence between IP and L2 distance
    faiss.normalize_L2(embeddings)

    return embeddings


def evaluate_metrics(experiment_name, corpus_embeddings, chunk_texts, gold_data, query_embeddings, results_accumulator):
    d = corpus_embeddings.shape[1]

    # Metric Factory
    metric_map = {
        "IP (Cosine)": faiss.IndexFlatIP(d),
        "L2 (Euclidean)": faiss.IndexFlatL2(d),
        "L1 (Manhattan)": faiss.IndexFlat(d, faiss.METRIC_L1),
        "Linf (Chebyshev)": faiss.IndexFlat(d, faiss.METRIC_Linf)
    }

    n_gold = len(gold_data)

    for metric_name, index in metric_map.items():
        try:
            # 1. Build Index (Instant for Flat indexes)
            index.add(corpus_embeddings)

            # 2. Batch Search
            distances, retrieved_indices = index.search(query_embeddings, TOP_K)

            # 3. Calculate Evaluation Metrics
            sums = {"mrr": 0.0, "ndcg_at_k": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0, "f1_score_at_k": 0.0}

            for i, gold_item in enumerate(gold_data):
                retrieved_ids = retrieved_indices[i]

                # Retrieve actual text for checking against gold
                retrieved_texts = []
                for rid in retrieved_ids:
                    if 0 <= rid < len(chunk_texts):
                        retrieved_texts.append(chunk_texts[rid])

                # Use your existing evaluation.py logic (string matching)
                scores = calculate_metrics(
                    retrieved_chunks=retrieved_texts,
                    gold_passages=gold_item['gold_passages'],
                    k=TOP_K
                )

                for k in sums:
                    sums[k] += scores.get(k, 0.0)

            # 4. Average and Store
            row = {
                "Experiment": experiment_name,
                "Distance_Metric": metric_name,
                "Chunks_Count": len(chunk_texts),
                "MRR": round(sums["mrr"] / n_gold, 4),
                "NDCG@10": round(sums["ndcg_at_k"] / n_gold, 4),
                "Precision@10": round(sums["precision_at_k"] / n_gold, 4),
                "Recall@10": round(sums["recall_at_k"] / n_gold, 4),
                "F1@10": round(sums["f1_score_at_k"] / n_gold, 4)
            }
            results_accumulator.append(row)

            # Cleanup
            del index
            gc.collect()

        except Exception as e:
            tprint(f"      ❌ Error in {metric_name}: {e}")


def main():
    if not os.path.exists(CHUNKS_ROOT):
        tprint(f"❌ Chunks root not found at {CHUNKS_ROOT}")
        return

    # 1. Load Gold Data
    gold_data = load_gold_data(GOLD_DATA_PATH)
    if not gold_data:
        tprint("❌ No gold data found.")
        return
    tprint(f"✅ Loaded {len(gold_data)} Gold Questions.")

    # 2. Get Experiments
    all_experiments = [d for d in os.listdir(CHUNKS_ROOT) if os.path.isdir(os.path.join(CHUNKS_ROOT, d))]

    # Filter based on user request: semantic*, fixed_1024_128, recursive_1024_128, sentence_s10
    experiments = []
    for exp in all_experiments:
        if (exp.startswith("semantic") or
                exp == "fixed_1024_128" or
                exp == "recursive_1024_128" or
                exp == "sentence_s10"):
            experiments.append(exp)

    experiments.sort()
    tprint(f"🔹 Selected {len(experiments)} experiments for analysis: {experiments}")

    # 3. Load Model
    device = get_device()
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 4. Pre-encode Gold Questions (Query Embeddings)
    tprint("🔹 Encoding Gold Questions...")
    questions = [g['question'] for g in gold_data]

    # Check if questions exist
    if len(questions) == 0:
        tprint("❌ No valid questions found in gold data.")
        return

    query_embeddings = model.encode(questions, normalize_embeddings=True, convert_to_numpy=True, batch_size=BATCH_SIZE)
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    # CRITICAL: Explicitly normalize queries for valid Cosine/L2 comparison
    faiss.normalize_L2(query_embeddings)

    all_results = []

    # 5. Main Loop
    for exp in tqdm(experiments, desc="Experiments", unit="exp"):
        chunks_file = os.path.join(CHUNKS_ROOT, exp, "chunks.json")
        if not os.path.exists(chunks_file): continue

        tprint(f"\n🚀 Experiment: {exp}")

        # Load Text (Required for Calculate Metrics)
        chunk_texts = load_chunks_text_only(chunks_file)
        if not chunk_texts: continue

        # Load/Create Cache (Vectors)
        embeddings = get_or_create_embeddings(exp, chunk_texts, model)

        # Ensure contiguous float32 for Faiss
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Run Evaluation
        evaluate_metrics(exp, embeddings, chunk_texts, gold_data, query_embeddings, all_results)

        # Free Memory
        del chunk_texts
        del embeddings
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        if device == "mps": torch.mps.empty_cache()

    # 6. Save Final CSV
    if all_results:
        tprint(f"\n💾 Saving comparison to {CSV_OUTPUT_FILE}...")
        keys = all_results[0].keys()
        with open(CSV_OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        tprint("✅ Done.")


if __name__ == "__main__":
    main()
