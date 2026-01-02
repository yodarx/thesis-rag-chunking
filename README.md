# Thesis RAG Chunking Evaluation Framework

This project provides a framework for evaluating different chunking strategies in Retrieval-Augmented Generation (RAG) systems. It includes tools for data preprocessing, index building, silver standard generation (using LLMs), and running experiments with various configurations.

## Prerequisites

*   **Python 3.12** (Required)
*   **Google API Key** (For silver standard generation with Gemini 3 Pro)
*   **Ollama** (Optional - only for LLM-based categorization if needed)

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

## 2. Google Gemini API Setup

This project uses **Google's Gemini 3 Pro** model to generate synthetic "silver standard" datasets. 

### 2.1. Get Your API Key

1.  **Visit Google AI Studio**: Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2.  **Create/Copy API Key**: Click "Create API key" or copy an existing one
3.  **Set Environment Variable**: Store the key securely in your shell environment

**macOS / Linux:**
```bash
export GEMINI_API_KEY='your-api-key-here'
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY='your-api-key-here'
```

**Windows (CMD):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

### 2.2. Verify Setup

Test your API key:
```bash
python -c "from google import genai; client = genai.Client(); print('✓ Gemini API is configured correctly')"
```

### 2.3. Alternative: Vertex AI

If you prefer to use **Google Cloud Vertex AI** instead:

```bash
# Set environment variables
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'

# The code will automatically use Vertex AI
python build_silver.py configs/0_base_experiment.json
```

---

## 2.4. Ollama (Optional - For Categorization Only)

If you need to run the optional categorization step, you'll need Ollama:

1.  **Install Ollama**: Download from [ollama.com](https://ollama.com/)
2.  **Start Ollama**: Run the application
3.  **Pull a Model**: 
    ```bash
    ollama pull llama3
    ```
4.  **Update Config**: Set `llm_model` in your config file to match the model you pulled

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

Creates a synthetic evaluation dataset (Question-Answer pairs) from your chunks using **Gemini 3 Pro**. This is useful for "Silver Standard" evaluation when human-annotated gold standard data is unavailable.

**Prerequisites:**
- Gemini API key set in environment (see Section 2.1)
- Chunks already indexed (from `build_indices.py`)

**Command:**
```bash
python build_silver.py configs/0_base_experiment.json
```

**Configuration:**

The following settings in your config JSON control silver standard generation:

```json
{
  "silver_limit": 2,      // How many Q&A pairs to generate per experiment
  "hops_count": 2,        // Multi-hop depth (how many chunks to use)
  "embedding_model": "all-MiniLM-L6-v2"  // For context retrieval
}
```

| Setting | Values | Default | Purpose |
|---------|--------|---------|---------|
| `silver_limit` | `-1` or positive int | `10` | `-1`: Generate for ALL chunks. Positive number: Generate exactly that many samples. |
| `hops_count` | Positive int | `2` | Number of chunks to use for multi-hop question generation. Higher = more complex questions. |
| `embedding_model` | Any HuggingFace model | `all-MiniLM-L6-v2` | Used to select relevant context chunks. |

**Example Usage:**

```bash
# Generate 5 samples per experiment
python build_silver.py configs/0_base_experiment.json

# Generate for ALL chunks (set silver_limit: -1 in config, then run)
python build_silver.py configs/0_base_experiment.json
```

**How It Works:**

For each experiment in your config:
1. Loads pre-chunked documenets from `data/indices/{experiment_name}_{embedding_model}/chunks.json`
2. For each sample:
   - Randomly selects `hops_count` chunks as context
   - Sends them to **Gemini 3 Pro** to generate a multi-hop question
   - Extracts the answer from the provided context
3. Saves Q&A pairs to `data/silver/{experiment_name}_{embedding_model}_silver.jsonl`

**Output Format:**

Each line in the output JSONL file:
```json
{
  "sample_id": "uuid",
  "question": "Generated question",
  "answer": "Extracted answer",
  "gold_passages": ["chunk1", "chunk2"],
  "category": "Multihop",
  "difficulty": "Hard"
}
```

**Important Notes:**

- ✅ **Idempotent**: Won't regenerate if the output file already exists (delete to regenerate)
- ✅ **API Efficient**: Uses Gemini 3 Pro's efficient API - verify your quota at [Google AI Console](https://aistudio.google.com/app/apikey)
- ✅ **Error Handling**: Continues processing even if some samples fail; retries up to 10x per sample
- ✅ **Progress Tracking**: Prints status for each experiment

**Troubleshooting:**

| Problem | Solution |
|---------|----------|
| `ImportError: cannot import name 'genai'` | Run `pip install -r requirements.txt` to install google-genai |
| `APIError: 401 Unauthorized` | Verify API key: `echo $GEMINI_API_KEY` |
| `Chunks file not found` | Run `python build_indices.py configs/0_base_experiment.json` first |
| High failure rate in output | Check your chunks are > 100 characters; increase `silver_limit` for more attempts |

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

#### **Gold Standard Evaluation**

Evaluates against human-annotated, high-quality data.
Ensure `input_file` in the config points to your gold standard dataset.

```bash
python run.py --config-json configs/0_base_experiment.json
```

**Use case**: Comprehensive evaluation with reliable ground truth

---

#### **Silver Standard Evaluation (Using Gemini)**

Evaluates against synthetic data generated by **Gemini 3 Pro**. This mode:
1. **Automatically generates** a silver dataset for each experiment (if one doesn't exist)
2. **Uses Gemini 3 Pro** to generate multi-hop Q&A pairs from chunks
3. **Evaluates each chunking strategy** against its own generated questions
4. Allows overriding the dataset via `input_silver_file` in the experiment config

```bash
# Enable Silver Standard mode
python run.py --config-json configs/0_base_experiment.json --silver
```

**Prerequisites:**
- Gemini API key configured (see Section 2)
- Indices built via `build_silver.py` or `build_indices.py`

**How it works:**
- For each experiment, Gemini generates questions from that experiment's chunks
- Since questions are tailored to the chunking strategy, results reflect how well that chunking preserves retrievability
- Useful for rapid prototyping and parameter sensitivity studies

**Use case**: Efficient experimentation without human annotation

---

#### **Difficulty Filtering**

Filter the dataset by difficulty (e.g., "Hard", "Medium", "Easy") using the `--difficulty` flag. This works for both Gold and Silver modes, provided the dataset contains a `difficulty` field.

```bash
python run.py --config-json configs/0_base_experiment.json --difficulty Hard
```

**Example Workflow:**

```bash
# 1. Build indices for all chunking strategies
python build_indices.py configs/0_base_experiment.json

# 2. Generate silver standard questions (Gemini-powered)
python build_silver.py configs/0_base_experiment.json

# 3. Run experiments against silver standard
python run.py --config-json configs/0_base_experiment.json --silver

# 4. Compare results in results/{timestamp}/ folder
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
