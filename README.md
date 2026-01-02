# Thesis RAG Chunking Evaluation Framework

This project provides a framework for evaluating different chunking strategies in Retrieval-Augmented Generation (RAG) systems. It includes tools for data preprocessing, index building, silver standard generation (using LLMs), and running experiments with various configurations.

## Prerequisites

*   **Python 3.12** (Required)
*   **Ollama** (For LLM-based categorization and silver standard generation)

## 1. Environment Setup

### 1.1. Clone the Repository

```bash
git clone <repository-url>
cd thesis-rag-chunking
```

### 1.2. Create a Virtual Environment

**macOS / Linux:**

```bash
python3.12 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
py -3.12 -m venv venv
.\venv\Scripts\activate
```

### 1.3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Ollama Setup

This project uses [Ollama](https://ollama.com/) to run local LLMs for data categorization and generating synthetic "silver standard" datasets.

1.  **Install Ollama**: Download and install from [ollama.com](https://ollama.com/).
2.  **Start Ollama**: Run the application.
3.  **Pull the Model**: The default configuration uses `gpt-oss` (or whichever model you configure). You can pull a model like `llama3` or `mistral` and update your config, or use the one specified.

    ```bash
    ollama pull llama3
    ollama pull qwen2.5:14b
    ```
    *(Note: Ensure the `llm_model` in your config files matches the model you pulled, e.g., "llama3")*

## 3. Development Workflow

This project uses `taskipy` for managing development tasks and `ruff` for formatting/linting.

*   **Run Tests**:
    ```bash
    task test
    # OR directly with pytest
    pytest
    ```

*   **Format Code**:
    ```bash
    task format
    ```

*   **Lint Code**:
    ```bash
    task lint
    ```

## 4. Data Pipeline

The pipeline consists of several stages: Preprocessing -> Categorization -> Indexing -> Silver Standard Generation -> Experiment Execution.

### 4.1. Preprocessing Data

Converts raw data (e.g., from HuggingFace or web scrapes) into the standardized JSONL format.

```bash
python src/preprocessor/main.py
```
*Check the script arguments for input/output paths if necessary.*

### 4.2. Categorizing Data (Optional)

Adds metadata like `category` (Factoid, Inference, Multihop) and `difficulty` to your dataset using an LLM.

```bash
python src/preprocessor/categorize.py --input data/preprocessed/your_file.jsonl --output data/preprocessed/categorized.jsonl
```

### 4.3. Building Indices

Generates FAISS indices for different chunking strategies based on your configuration. Indices are saved in `data/indices/`.

```bash
python build_indices.py configs/0_base_experiment.json
```

### 4.4. Generating Silver Standard

Creates a synthetic evaluation dataset (Question-Answer pairs) from your chunks using an LLM. This is useful for "Silver Standard" evaluation.

**Note:** You can skip this step if you use the `--silver` flag with `run.py`, as it will automatically generate missing silver datasets for each experiment.

*   **Limit**: Controls how many samples to generate.
    *   `10`: Generates 10 samples (good for testing).
    *   `-1`: Generates samples for ALL chunks.
*   **Hops**: Configures multi-hop question generation (default 2 chunks).

```bash
# Run with config settings
python build_silver.py configs/0_base_experiment.json


# Override limit via CLI (e.g., generate for all chunks)
python build_silver.py configs/0_base_experiment.json --limit -1
```

The generated dataset is saved to `data/silver/`.

## 5. Running Experiments

Runs the RAG evaluation pipeline. You can run experiments against a **Gold Standard** (human-annotated) or **Silver Standard** (synthetic) dataset.

### 5.1. Configuration Files

Experiments are defined in JSON files located in `configs/`.

**Structure:**
```json
{
  "embedding_model": "all-MiniLM-L6-v2",  // Model for vectorization
  "retriever_type": "faiss",
  "top_k": 5,                             // Number of chunks to retrieve
  "input_file": "path/to/dataset.jsonl",  // Dataset to evaluate against
  "silver_limit": 10,                     // (For build_silver.py)
  "llm_model": "gpt-oss",                 // (For build_silver.py)
  "hops_count": 2,                        // (For build_silver.py)
  "experiments": [                        // List of chunking strategies to test
    {
      "name": "fixed_512_50",
      "function": "chunk_fixed_size",
      "params": {"chunk_size": 512, "chunk_overlap": 50},
      "input_silver_file": "path/to/custom_silver.jsonl" // Optional: Override auto-generated silver file
    }
  ]
}
```

### 5.2. Running the Evaluation

To run an experiment, point `run.py` to a config file.

**For Gold Standard:**
Ensure `input_file` in the config points to your gold standard dataset.

```bash
python run.py --config-json configs/0_base_experiment.json
```

**For Silver Standard:**
Use the `--silver` flag to run in Silver Standard mode. This mode:
1.  Automatically generates a silver dataset for each experiment (if one doesn't exist).
2.  Uses the generated dataset to evaluate that specific experiment.
3.  Allows overriding the dataset via `input_silver_file` in the experiment config.

```bash
python run.py --config-json configs/0_base_experiment.json --silver
```

**Filtering by Difficulty:**
You can filter the dataset by difficulty (e.g., "Hard", "Medium", "Easy") using the `--difficulty` flag. This works for both Gold and Silver modes, provided the dataset contains a `difficulty` field.

```bash
python run.py --config-json configs/0_base_experiment.json --difficulty Hard
```

*Note: This ensures that each chunking strategy is evaluated against questions generated from its own chunks.*

## 6. Results

Results are saved in the `results/` directory, organized by timestamp. Each run (whether Gold or Silver) creates a new timestamped folder.

*   **Separation**: Gold and Silver standard experiments are run separately, so their results will be in distinct folders (e.g., `results/results/2025-12-18_10-00-00_gold/` for Gold, `results/results/2025-12-18_11-00-00_silver/` for Silver).
*   **Identification**: The configuration file used for the run is copied into the results folder (e.g., `experiment_config_0_base_experiment.json`). You can check the `input_file` in this config to confirm if it was a Gold or Silver run.

Inside each result folder:

*   **Detailed Results**: CSV containing metrics for every single query.
*   **Summary**: CSV with aggregated metrics (MAP, MRR, F1, etc.) per experiment.
*   **Plots**: Visualizations of the metrics.

Example path: `results/results/2025-12-18_10-00-00_gold/`
