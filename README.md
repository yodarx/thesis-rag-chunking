# RAG Chunking Strategy Evaluation Framework

This project provides a comprehensive and modular framework for evaluating the impact of different text chunking strategies on retrieval performance. It is designed to be extensible, testable, and easy to run.

The framework processes the ASQA dataset, chunks the documents using various methods (Fixed-Size, Sentence, Recursive, Semantic), indexes them using FAISS, and then evaluates retrieval performance based on the dataset's gold-standard passages using metrics like MRR, MAP, and NDCG@k.

---

## 🚀 Features

* **Modular Preprocessing:** A standalone script to fetch, parse, and clean ASQA data from Wikipedia.
* **Multiple Chunking Strategies:**
    * Fixed-Size
    * Sentence-based
    * RecursiveCharacterText
    * Semantic (similarity-based)
* **Vector-Based Retrieval:** Uses `sentence-transformers` for embedding and `faiss-cpu` for efficient similarity search.
* **Comprehensive Evaluation:** Calculates MRR, MAP, NDCG@k, Precision@k, Recall@k, and F1-Score@k for each strategy.
* **Automated Visualization:** Generates bar charts (Performance vs. Strategy) and scatter plots (Cost-Benefit vs. Compute Time) for all metrics.
* **Clean & Testable Code:** Fully type-hinted, modular, and testable code using Dependency Injection and a clean separation of concerns.

---

## 📦 Project Structure

The project uses a `src` layout and a main `run.py` script for orchestration.

```

thesis-rag-chunking/
├── run.py                \<-- Main experiment orchestrator
├── requirements.txt      \<-- Project dependencies
├── data/
│   └── processed/        \<-- Output of the preprocessor (e.g., asqa\_preprocessed.jsonl)
├── src/
│   ├── **init**.py       \<-- Makes 'src' a package
│   ├── preprocessor/     \<-- ASQA data downloader and parser
│   ├── chunking/         \<-- All chunking strategy modules (fixed, semantic, etc.)
│   ├── vectorizer/       \<-- SentenceTransformer wrapper
│   ├── experiment/       \<-- Core logic (runner, retriever, results)
│   ├── evaluation/       \<-- Retrieval metrics calculation
│   └── plotting/         \<-- Visualization (matplotlib/seaborn)
├── results/              \<-- Output CSVs and plots
├── configs/              # Experiment configuration files (JSON)
│   └── base_experiment.json  # Default experiment config
└── tests/                \<-- Pytest unit and integration tests

````

---

## 🛠️ Setup & Installation

**Prerequisite:** This project requires **Python 3.12+**.

1.  **Clone the Repository:**
    ```bash
    git clone [https://your-repository-url.com/thesis-rag-chunking.git](https://your-repository-url.com/thesis-rag-chunking.git)
    cd thesis-rag-chunking
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create venv
    python3.12 -m venv venv
    
    # Activate venv (macOS/Linux)
    source venv/bin/activate
    
    # (Windows)
    # .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data:**
    The chunking strategies require the NLTK `punkt` tokenizer. Run this command once to download it:
    ```bash
    python -m nltk.downloader punkt
    ```

---

## ⚙️ How to Run the Experiments

All experiment settings (chunking strategies, model, retriever, etc.) are now managed via a config file, typically `configs/base_experiment.json`.

**To run the main experiment:**
```bash
python run.py --config-json configs/base_experiment.json
```
- You can specify a different config file with `--config-json path/to/your_config.json`.
- The config file used for the run will be copied into the results directory as `experiment_config.json` for reproducibility.

**Customizing experiments:**
- Edit `configs/base_experiment.json` to change chunking strategies, model, retriever, or dataset parameters.
- You can create multiple config files in the `configs/` directory for different experiment setups.

---

## 🧩 Experiment Configuration

All experiments are defined in JSON config files located in the `configs/` folder. The default config is `configs/base_experiment.json`.

**Config file structure:**
```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "retriever_type": "faiss",
  "top_k": 5,
  "limit": 100,
  "input_file": "data/preprocessed/preprocessed_2025-10-23_limit_5.jsonl",
  "experiments": [
    {
      "name": "fixed_512_50",
      "function": "chunk_fixed_size",
      "params": {"chunk_size": 512, "chunk_overlap": 50}
    },
    {
      "name": "sentence_s3",
      "function": "chunk_by_sentence",
      "params": {"sentences_per_chunk": 3}
    },
    {
      "name": "semantic_t0.7",
      "function": "chunk_semantic",
      "params": {"similarity_threshold": 0.7}
    }
    // ... more experiments ...
  ]
}
```
- `embedding_model`: Name of the embedding model (e.g., from sentence-transformers)
- `retriever_type`: Retrieval backend (currently only `faiss`)
- `top_k`: Number of top results to retrieve
- `limit`: Limit the number of dataset entries (for quick tests)
- `input_file`: Path to the preprocessed ASQA dataset
- `experiments`: List of chunking strategies to evaluate

Each experiment must specify:
- `name`: A unique name for the experiment
- `function`: The chunking function (must match one of the available functions)
- `params`: Parameters for the chunking function

You can create additional config files in `configs/` for different experiment setups.

---

## ⚙️ Workflow Summary

1. **Preprocess the dataset:**
   ```bash
   python -m src.preprocessor.preprocessor
   ```
   This creates the input file referenced in your config (e.g., `data/preprocessed/preprocessed_2025-10-23_limit_5.jsonl`).

2. **Edit your experiment config:**
   - Open `configs/base_experiment.json` (or create a new config in `configs/`).
   - Adjust chunking strategies, parameters, or dataset path as needed.

3. **Run the experiment:**
   ```bash
   python run.py --config-json configs/base_experiment.json
   ```
   - The config file will be copied into the results folder for reproducibility.
   - All outputs (results, plots, config) are saved in a timestamped subfolder in `results/`.

4. **View results:**
   - Check the results folder for CSVs, plots, and the config used for the run.

---

## 📊 Viewing Results

All outputs from `run.py` are saved in a timestamped folder inside the `results/` directory (e.g., `results/2025-10-23_13-30-00/`).

  * `experiment_config.json`: The exact config used for this run.
  * `_detailed_results.csv`: Metrics for every data point and experiment.
  * `_summary_results.csv`: Aggregated results for each strategy.
  * `*.png`: All generated plots.

The aggregated results are also printed directly to the console at the end of the run.

-----

## 🧪 Running Tests

This project uses `pytest` for testing. The tests are located in the `tests/` directory and mirror the `src/` structure.

To run all tests, ensure your `venv` is active and run the following command from the project root:

```bash
pytest
```
