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
    python3 -m venv venv
    
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

The workflow is a two-step process. All commands **must be run from the project's root directory** (`thesis-rag-chunking/`) for the imports to work correctly.

### Step 1: Run the Preprocessor

First, you must download and process the ASQA dataset. This script runs as a standalone module.

```bash
# This will fetch Wikipedia data and create the .jsonl file in data/processed/
python -m src.preprocessor.preprocessor
````

*Note: This script has a `PREPROCESS_LIMIT` variable at the top, which is useful for quick testing. Set it to `None` to process the entire dataset.*

### Step 2: Run the Main Experiment

After the preprocessor has finished, you can run the main experiment pipeline. This will load the `.jsonl` file, run all chunking experiments, calculate metrics, and save the results.

```bash
python run.py
```

-----

## 📊 Viewing Results

All outputs from `run.py` are saved in a timestamped folder inside the `results/` directory (e.g., `results/2025-10-23_13-30-00/`).

  * `_detailed_results.csv`: A CSV file with the metrics for **every single data point** and experiment.
  * `_summary_results.csv`: A CSV file with the **aggregated (mean) results** for each experiment strategy.
  * `*.png`: All generated plots (bar charts and scatter plots) for easy analysis.

The aggregated results are also printed directly to the console at the end of the run.

-----

## 🧪 Running Tests

This project uses `pytest` for testing. The tests are located in the `tests/` directory and mirror the `src/` structure.

To run all tests, ensure your `venv` is active and run the following command from the project root:

```bash
pytest
```
