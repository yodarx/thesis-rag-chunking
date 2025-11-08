# Thesis RAG Chunking Experiments

This project explores different chunking strategies for Retrieval-Augmented Generation (RAG) systems as part of a thesis
research.

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
│   ├── 1_full_parameter_sweep.json   # Full parameter sweep
│   └── 2_model_sensitivity_*.json    # Model sensitivity tests
├── data/
│   └── preprocessed/                 # Preprocessed datasets (gitignored)
├── results/                          # Experiment results (gitignored)
├── tests/                            # Unit tests
├── notebooks/                        # Jupyter notebooks for analysis
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies  
├── Dockerfile                        # Docker container configuration
├── build-and-upload.sh              # Docker build and upload script
├── test-docker-locally.sh           # Local Docker testing script
└── README.md
```

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
# Build and upload Docker container
chmod +x build-and-upload.sh
./build-and-upload.sh
```

## Docker Setup

### Building the Container

The project includes a complete Docker setup for running experiments on cloud platforms like RunPod.

**Key Docker Features:**

- Python 3.12 slim base image for optimal performance
- Automatic NLTK data download (punkt, punkt_tab) during build
- Production-only dependencies for lean container (~2GB vs ~4GB with dev deps)
- RunPod workspace volume mounting (`/workspace`)
- Automatic results directory creation
- Proper Python path configuration

**Build and Upload Script (`build-and-upload.sh`):**

```bash
#!/bin/bash
IMAGE_NAME="thesis-rag-chunking"
TAG="latest"
DOCKERHUB_USERNAME="your-dockerhub-username"  # Replace with your username

# Build, tag, and upload to Docker Hub
docker build -t $IMAGE_NAME:$TAG .
docker tag $IMAGE_NAME:$TAG $DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG
docker login
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG
```

**Local Testing Script (`test-docker-locally.sh`):**

- Builds container locally
- Provides interactive shell for manual testing
- Runs default experiment to verify functionality
- Creates local workspace directory for output

### Using on RunPod

After building and uploading, use this image in RunPod:

```
your-dockerhub-username/thesis-rag-chunking:latest
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

Experiments are defined in JSON files in the `configs/` directory:

```json
{
  "experiments": [
    {
      "name": "fixed_512_128",
      "function": "chunk_fixed_size",
      "params": {
        "chunk_size": 512,
        "chunk_overlap": 128
      }
    }
  ]
}
```

### Available Chunking Strategies

1. **Fixed Size** (`chunk_fixed_size`) - Fixed character length chunks
2. **Recursive** (`chunk_recursive`) - Recursive character text splitting
3. **Semantic** (`chunk_semantic`) - Similarity-based semantic chunking
4. **Sentence** (`chunk_by_sentence`) - Sentence boundary-based chunking

### Evaluation Metrics

- **MAP (Mean Average Precision)** - Overall retrieval quality
- **MRR (Mean Reciprocal Rank)** - First relevant result ranking
- **NDCG@K** - Normalized discounted cumulative gain
- **Precision@K, Recall@K, F1@K** - Standard IR metrics

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
# Test Docker container locally
chmod +x test-docker-locally.sh
./test-docker-locally.sh
```

### 3. Production Deployment

```bash
# Build and upload to Docker Hub
chmod +x build-and-upload.sh
./build-and-upload.sh
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
