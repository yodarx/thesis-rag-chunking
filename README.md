# RAG Chunking Strategy Evaluation Framework

This project provides a comprehensive and modular framework for evaluating the impact of different text chunking
strategies on retrieval performance. It is designed to be extensible, testable, and easy to run.

The framework processes the ASQA dataset, chunks the documents using various methods (Fixed-Size, Sentence, Recursive,
Semantic), indexes them using FAISS, and then evaluates retrieval performance based on the dataset's gold-standard
passages using metrics like MRR, MAP, and NDCG@k.

---

## 🚀 Features

* **Modular Preprocessing:** A standalone script to fetch, parse, and clean ASQA data from Wikipedia.
* **Multiple Chunking Strategies:**
    * Fixed-Size
    * Sentence-based
    * RecursiveCharacterText
    * Semantic (similarity-based)
* **Vector-Based Retrieval:** Uses `sentence-transformers` for embedding and `faiss-cpu` for efficient similarity
  search.
* **Comprehensive Evaluation:** Calculates MRR, MAP, NDCG@k, Precision@k, Recall@k, and F1-Score@k for each strategy.
* **Automated Visualization:** Generates bar charts (Performance vs. Strategy) and scatter plots (Cost-Benefit vs.
  Compute Time) for all metrics.
* **Clean & Testable Code:** Fully type-hinted, modular, and testable code using Dependency Injection and a clean
  separation of concerns.

---

## 📦 Project Structure

The project uses a `src` layout and a main `run.py` script for orchestration.

```

thesis-rag-chunking/
├── run.py                \<-- Main experiment orchestrator
├── requirements.txt      \<-- Project dependencies
├── configs/              # Experiment configuration files (JSON)
│   └── 0_base_experiment.json  # Default experiment config
├── data/
│   └── preprocessed/        \<-- Output of the preprocessor (e.g., asqa\_preprocessed.jsonl)
├── results/              \<-- Output CSVs and plots
├── src/
│   ├── **init**.py       \<-- Makes 'src' a package
│   ├── preprocessor/     \<-- ASQA data downloader and parser
│   ├── chunking/         \<-- All chunking strategy modules (fixed, semantic, etc.)
│   ├── vectorizer/       \<-- SentenceTransformer wrapper
│   ├── experiment/       \<-- Core logic (runner, retriever, results)
│   ├── evaluation/       \<-- Retrieval metrics calculation
│   └── plotting/         \<-- Visualization (matplotlib/seaborn)
└── tests/                \<-- Pytest unit and integration tests

````

---

## 🛠️ Setup & Installation

**Prerequisite:** This project requires **Python 3.12+**.

1. **Clone the Repository:**
   ```bash
   git clone https://git.ffhs.ch/jeremy.rhodes/thesis-rag-chunking.git
   cd thesis-rag-chunking
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   # Create venv
   python3.12 -m venv venv
   
   # Activate venv (macOS/Linux)
   source venv/bin/activate
   
   # (Windows)
   # .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data:**
   The chunking strategies require the NLTK `punkt` tokenizer. Run this command once to download it:
   ```bash
   python -m nltk.downloader punkt
   python -m nltk.downloader punkt_tab
   ```

---

## ⚙️ How to Run the Experiments

All experiment settings (chunking strategies, model, retriever, etc.) are now managed via a config file,
typically `configs/base_experiment.json`.

**To run the main experiment:**

```bash
python run.py --config-json configs/base_experiment.json
```

- You can specify a different config file with `--config-json path/to/your_config.json`.
- The config file used for the run will be copied into the results directory as `experiment_config.json` for
  reproducibility.

**Customizing experiments:**

- Edit `configs/base_experiment.json` to change chunking strategies, model, retriever, or dataset parameters.
- You can create multiple config files in the `configs/` directory for different experiment setups.

---

## 🧩 Experiment Configuration

Experiments are defined in JSON config files (see `configs/`). Each experiment can specify chunking parameters and
options. Example structure:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  // Name of the embedding model used for vectorization
  "retriever_type": "faiss",
  // Type of retriever (e.g., 'faiss')
  "top_k": 5,
  // Number of top results to retrieve for each query
  "input_file": "data/preprocessed/preprocessed_2025-11-03_all.jsonl",
  // Path to the input data file
  "experiments": [
    {
      "name": "fixed_512_50",
      // Unique name for the experiment
      "function": "chunk_fixed_size",
      // Chunking function to use (must match a function in src/chunking)
      "params": {
        "chunk_size": 512,
        "chunk_overlap": 50
      },
      // Parameters for the chunking function
      "log_matches": true
      // (Optional) If true, logs detailed match info for each query
    }
    // ... more experiments ...
  ]
}
```

### Configuration Parameters Explained

- `embedding_model`: The name of the sentence-transformer model used to embed text chunks. Example: `all-MiniLM-L6-v2`.
- `retriever_type`: The retrieval backend to use. Currently, only `faiss` is supported.
- `top_k`: The number of top results to retrieve for each query during evaluation.
- `input_file`: Path to the preprocessed input data file (JSONL format).
- `experiments`: A list of experiment objects, each specifying a chunking strategy and its parameters.
    - `name`: A unique identifier for the experiment (used in result tracking).
    - `function`: The chunking function to use. Must match a function name in `src/chunking/` (
      e.g., `chunk_fixed_size`, `chunk_by_sentence`, `chunk_recursive`, `chunk_semantic`).
    - `params`: A dictionary of parameters for the chunking function. The required parameters depend on the function:
        - For `chunk_fixed_size`: `chunk_size` (int), `chunk_overlap` (int)
        - For `chunk_by_sentence`: `sentences_per_chunk` (int)
        - For `chunk_recursive`: `chunk_size` (int), `chunk_overlap` (int)
        - For `chunk_semantic`: `similarity_threshold` (float)
    - `log_matches`: (Optional, bool) If true, logs detailed match information for each query during evaluation. Default
      is false.

## 📂 Results & Config Tracking

- Results are saved in the `results/` folder, with a timestamped subfolder for each run.
- The experiment config used for a run is automatically copied into the results folder for reproducibility.

## ▶️ Running Experiments

To run an experiment, use:

```bash
python run.py --config-json configs/base_experiment_all.json
```

You can create your own config files in the `configs/` directory and specify them with `--config-json`.

## 🧪 Testing

Run all tests with:

```bash
pytest
```

---

