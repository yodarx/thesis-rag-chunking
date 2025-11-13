# Thesis RAG Chunking Experiments

This project explores different chunking strategies for Retrieval-Augmented Generation (RAG) systems as part of a thesis
research. It provides a comprehensive framework for comparing fixed-size, recursive, sentence-based, and semantic
chunking approaches with various embedding models and retrieval configurations.

## Project Structure

```
thesis-rag-chunking/
├── src/
│   ├── chunking/
│   │   ├── chunk_fixed.py            # Fixed-size chunking
│   │   ├── chunk_recursive.py        # Recursive character text splitting
│   │   ├── chunk_semantic.py         # Semantic similarity-based chunking
│   │   └── chunk_sentence.py         # Sentence-based chunking
│   ├── experiment/
│   │   ├── data_loader.py            # ASQA dataset loading
│   │   ├── retriever.py              # FAISS-based retrieval
│   │   ├── runner.py                 # Experiment execution
│   │   └── results.py                # Results handling
│   ├── evaluation/
│   │   └── evaluation.py             # Evaluation metrics (MAP, MRR, NDCG, etc.)
│   ├── vectorizer/
│   │   └── vectorizer.py             # Sentence transformer vectorization
│   ├── plotting/
│   │   └── plotting.py               # Results visualization
│   └── preprocessor/
│       └── main.py                   # Data preprocessing utilities
├── configs/
│   ├── 0_base_experiment.json        # Base experiment configuration
│   ├── 1_full_parameter_sweep.json   # Full parameter sweep (all chunking methods)
│   ├── 2_model_sensitivity_bge_large.json  # Model sensitivity analysis
│   ├── 3_top_k_sensitivity_k1.json   # Top-k retrieval with k=1
│   ├── 4_top_k_sensitivity_k3.json   # Top-k retrieval with k=3
│   └── 5_top_k_sensitivity_k10.json  # Top-k retrieval with k=10
├── data/
│   └── preprocessed/                 # Preprocessed datasets (gitignored)
├── docker/                           # Docker configuration and scripts
│   ├── Dockerfile                    # Container definition
│   ├── .dockerignore                 # Docker build exclusions
│   ├── build-and-upload.sh          # Build and push script
│   ├── run_and_stop.sh              # Container entrypoint script
│   └── README.md                     # Docker documentation
├── results/                          # Experiment results (gitignored)
├── tests/                            # Unit tests
├── notebooks/                        # Jupyter notebooks for analysis
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── run.py                            # Main experiment runner
└── README.md
```

## Key Features

- 🔬 **Multiple Chunking Strategies**: Fixed, Recursive, Sentence-based, and Semantic chunking
- 🧪 **Scientific Rigor**: Separate chunking and retrieval embeddings to avoid confounding variables
- 📊 **Comprehensive Metrics**: MAP, MRR, NDCG, Precision, Recall, F1
- 🚀 **Production Ready**: Docker support for RunPod and cloud deployment
- 🔧 **Flexible Configuration**: JSON-based experiment definitions
- 📈 **Automatic Visualization**: Results plotting and statistical analysis
- ⚡ **Parallel Execution**: Run multiple experiments efficiently

## Requirements

### Production Dependencies

The following packages are required to run the experiments:

- **numpy, pandas** - Data manipulation and analysis
- **torch** - Deep learning framework
- **sentence-transformers** - Text embeddings generation
- **scikit-learn** - Machine learning utilities (cosine similarity)
- **faiss-cpu** - Vector similarity search and retrieval
- **langchain-core, langchain-text-splitters** - Text splitting and LLM utilities
- **nltk** - Natural language processing (sentence tokenization)
- **requests, beautifulsoup4** - Web scraping and HTML parsing
- **datasets** - HuggingFace dataset handling
- **matplotlib, seaborn** - Data visualization
- **tqdm** - Progress bars for long-running operations

### Development Dependencies

Additional tools for local development:

- **pytest, pytest-mock** - Testing framework and mocking
- **jupyter, ipython** - Interactive development environment
- **black, flake8, mypy** - Code formatting and linting (optional)

## Installation

### Local Development Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd thesis-rag-chunking

# Install all dependencies (production + development)
pip install -r requirements-dev.txt

# Download required NLTK data
python -m nltk.downloader punkt punkt_tab
```

### Production Setup (Docker)

```bash
# Build and upload Docker container (from project root)
bash docker/build-and-upload.sh
```

## Docker Setup

See the [docker/README.md](docker/README.md) for complete Docker documentation.

### Building the Container

The project includes a complete Docker setup for running experiments on cloud platforms like RunPod. All Docker-related
files are organized in the `docker/` directory.

**Key Docker Features:**

- Python 3.12 slim base image for optimal performance
- Automatic NLTK data download (punkt, punkt_tab) during build
- Production-only dependencies for lean container (~2GB vs ~4GB with dev deps)
- RunPod workspace volume mounting (`/workspace`)
- Automatic results directory creation
- Proper Python path configuration

**Quick Start:**

```bash
# Build and upload to Docker Hub (run from project root)
bash docker/build-and-upload.sh [TAG]

# Build locally without uploading
bash docker/build-and-upload.sh dev
```

For detailed Docker documentation, see [docker/README.md](docker/README.md).

### Using on RunPod

After building and uploading, use this image in RunPod:

```
jsstudentsffhs/thesis-rag-chunking:latest
```

The container automatically uses `/workspace` for persistent storage if available (RunPod), otherwise falls back to
local `results/` directory.

## Usage

### Running Experiments Locally

```bash
# Run the base experiment
python run.py --config-json configs/0_base_experiment.json

# Run a full parameter sweep
python run.py --config-json configs/1_full_parameter_sweep.json

# Run model sensitivity analysis
python run.py --config-json configs/2_model_sensitivity_bge_large.json

# Run tests
pytest tests/
```

### Running on RunPod

The Docker container automatically runs the base experiment when started:

```bash
# Default command runs:
python run.py --config-json configs/0_base_experiment.json
```

You can also access the container shell for custom experiments:
```bash
# Run specific configurations
python run.py --config-json configs/1_full_parameter_sweep.json

# Access results
ls /workspace/results/
```

## Experiment Configuration

### Configuration Files

Experiments are defined in JSON files in the `configs/` directory. Each configuration specifies the embedding model,
retrieval settings, and chunking experiments to run.

**Available Configurations:**

- **0_base_experiment.json** - Quick validation with core chunking methods
- **1_full_parameter_sweep.json** - Comprehensive sweep of all parameters and methods
- **2_model_sensitivity_bge_large.json** - Tests with different embedding models
- **3_top_k_sensitivity_k1.json** - Retrieval analysis with top-1 results
- **4_top_k_sensitivity_k3.json** - Retrieval analysis with top-3 results
- **5_top_k_sensitivity_k10.json** - Retrieval analysis with top-10 results

**Example Configuration:**

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "retriever_type": "faiss",
  "top_k": 5,
  "limit": 1,
  "input_file": "data/preprocessed/preprocessed_2025-11-03_all.jsonl",
  "experiments": [
    {
      "name": "fixed_512_50",
      "function": "chunk_fixed_size",
      "params": {"chunk_size": 512, "chunk_overlap": 50}
    },
    {
      "name": "semantic_t0.8_miniLM",
      "function": "chunk_semantic",
      "params": {
        "similarity_threshold": 0.8,
        "chunking_embeddings": "all-MiniLM-L6-v2"
      }
    }
  ]
}
```

### Available Chunking Strategies

#### 1. Fixed Size (`chunk_fixed_size`)

Creates chunks of fixed character length with optional overlap.

**Parameters:**

- `chunk_size`: Number of characters per chunk (e.g., 128, 256, 512, 1024)
- `chunk_overlap`: Number of overlapping characters between chunks (e.g., 0, 50, 128)

**Example:**

```json
{
  "name": "fixed_512_50",
  "function": "chunk_fixed_size",
  "params": {
    "chunk_size": 512,
    "chunk_overlap": 50
  }
}
```

#### 2. Recursive Character Splitting (`chunk_recursive`)

Uses LangChain's recursive character text splitter with intelligent break points.

**Parameters:**

- `chunk_size`: Target chunk size in characters
- `chunk_overlap`: Overlap between chunks

**Example:**

```json
{
  "name": "recursive_512_50",
  "function": "chunk_recursive",
  "params": {
    "chunk_size": 512,
    "chunk_overlap": 50
  }
}
```

#### 3. Sentence-Based (`chunk_by_sentence`)

Groups consecutive sentences into chunks.

**Parameters:**

- `sentences_per_chunk`: Number of sentences per chunk (e.g., 1, 3, 5, 10)

**Example:**

```json
{
  "name": "sentence_s5",
  "function": "chunk_by_sentence",
  "params": {
    "sentences_per_chunk": 5
  }
}
```

#### 4. Semantic Chunking (`chunk_semantic`)

**New and Improved!** Creates chunks based on semantic similarity between sentences using LangChain's SemanticChunker
with percentile-based breakpoint detection.

**Parameters:**

- `similarity_threshold`: Threshold for semantic similarity (0.0-1.0)
  - Higher values (0.8-0.95) create fewer, larger chunks
  - Lower values (0.5-0.7) create more, smaller chunks
- `chunking_embeddings`: Embedding model for chunking (separate from retrieval embeddings)
  - `"all-MiniLM-L6-v2"` - Fast, lightweight (recommended)
  - `"BAAI/bge-base-en-v1.5"` - Balanced performance
  - `"BAAI/bge-small-en-v1.5"` - Smaller, faster
  - `"sentence-transformers/all-mpnet-base-v2"` - Higher quality

**Example:**

```json
{
  "name": "semantic_t0.8_miniLM",
  "function": "chunk_semantic",
  "params": {
    "similarity_threshold": 0.8,
    "chunking_embeddings": "all-MiniLM-L6-v2"
  }
}
```

**Scientific Design:**
The semantic chunker uses **separate embeddings** for chunking vs. retrieval to maintain scientific rigor and avoid
confounding variables. This allows you to test whether the chunking model affects results independently of the retrieval
model.

**How it works:**

1. Splits text into sentences
2. Computes embeddings for each sentence
3. Calculates similarity between consecutive sentences
4. Creates chunk boundaries when similarity drops below the percentile threshold
5. Results in semantically coherent chunks that respect topic boundaries

### Running Multiple Experiments

```bash
# Run all config files in sequence
python run.py --run-all-configs

# Run specific configuration
python run.py --config-json configs/1_full_parameter_sweep.json

# Run with custom parameters
python run.py --config-json configs/0_base_experiment.json --limit 10
```

### Evaluation Metrics

All experiments automatically compute:

- **MAP (Mean Average Precision)** - Overall retrieval quality across all ranks
- **MRR (Mean Reciprocal Rank)** - Quality of the first relevant result
- **NDCG@K** - Normalized discounted cumulative gain at rank K
- **Precision@K** - Proportion of relevant documents in top K
- **Recall@K** - Proportion of relevant documents retrieved in top K
- **F1@K** - Harmonic mean of Precision@K and Recall@K

## Output and Results

### Directory Structure

```
results/
└── YYYY-MM-DD_HH-MM-SS/
    ├── detailed_results.csv          # Complete experimental results
    ├── *_bar_*.png                  # Visualization plots
    └── experiment_config.json        # Configuration used
```

### Visualization

Automatic generation of:

- Bar charts for all evaluation metrics
- Comparative analysis across chunking strategies
- Statistical significance testing results

## Development Workflow

### 1. Local Development
```bash
# Install development environment
pip install -r requirements-dev.txt
python -m nltk.downloader punkt punkt_tab

# Run experiments
python run.py --config-json configs/0_base_experiment.json

# Run tests
pytest tests/
```

### 2. Docker Testing
```bash
# Build Docker container locally (from project root)
bash docker/build-and-upload.sh dev
```

### 3. Production Deployment

```bash
# Build and upload to Docker Hub (from project root)
bash docker/build-and-upload.sh
```

### 4. RunPod Execution

Use the uploaded Docker image for cloud experiments with persistent storage.

## Dependencies Separation

**Production (`requirements.txt`):** Only packages needed for running experiments
**Development (`requirements-dev.txt`):** Includes production deps + development tools

This separation ensures:

- ✅ Lean Docker containers for production (faster builds, smaller images)
- ✅ Full development environment locally
- ✅ Faster deployment times
- ✅ Reduced security attack surface in production
- ✅ Clear distinction between runtime and development needs

## Technical Details

### RunPod Compatibility

- **Volume mounting**: Automatic detection of `/workspace` for persistent storage
- **Fallback**: Uses local `results/` directory when `/workspace` unavailable
- **Environment**: Optimized for GPU-enabled RunPod instances
- **Dependencies**: All ML libraries pre-installed and configured

### Performance Optimizations

- **FAISS**: CPU-optimized vector similarity search
- **Batch processing**: Efficient vectorization of document chunks
- **Memory management**: Streaming data processing for large datasets
- **Caching**: Intelligent caching of embeddings and models

### Data Pipeline

1. **Dataset Loading**: ASQA dataset via HuggingFace datasets
2. **Chunking**: Multiple strategies applied to documents
3. **Vectorization**: Sentence-transformer embeddings
4. **Indexing**: FAISS index construction for fast retrieval
5. **Evaluation**: Comprehensive metric calculation
6. **Visualization**: Automated plot generation

## Notes

- 📦 NLTK data (punkt, punkt_tab) automatically downloaded during Docker build
- 💾 Container includes volume mounting for persistent workspace storage
- 📊 All experiments output timestamped results to organized directories
- 🔧 Use development requirements file for local work
- ☁️ Docker setup optimized for RunPod cloud platform compatibility
- 🚀 Automatic fallback from `/workspace` to `results/` directory
- 📈 Comprehensive evaluation metrics and visualization included

## Contributing

1. Install development dependencies: `pip install -r requirements-dev.txt`
2. Run tests: `pytest tests/`
3. Format code: `black src/ tests/` (optional)
4. Test Docker build: `./test-docker-locally.sh`
5. Submit pull request with test coverage

## License

[Add your license information here]
