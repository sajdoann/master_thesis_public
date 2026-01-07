# Embedders Experiments - Project Report

## Overview

The `embedders_experiments` project is a comprehensive framework for evaluating and comparing different text embedding models in retrieval tasks. It provides a standardized pipeline for:

- Loading and chunking document collections
- Embedding documents using various embedding models (local and API-based)
- Building vector databases (ChromaDB or Qdrant)
- Running retrieval queries and evaluating performance
- Tracking costs, performance metrics, and resource usage

The project is designed for research and experimentation, particularly for evaluating embedding models on Czech and multilingual text retrieval tasks.

---

## CLI Script (`cli.py`) - Detailed Explanation

### Purpose

The `cli.py` script provides a command-line interface for running embedding experiments. It serves as the main entry point for executing retrieval experiments with different embedding models and configurations.

### Command Structure

The script uses a subcommand pattern with a single `run` command:

```bash
python -m embedders_experiments run [OPTIONS]
```

### Command-Line Arguments

#### Core Configuration

- **`--embedder`** (default: `"einfra"`)
  - Selects which embedding model to use
  - Choices: All keys from `EMBEDDERS` dictionary (see `constants.py`)
  - Examples: `einfra`, `qwen3`, `gemma2`, `text-embedding-3-small`, etc.

- **`--source`** (default: `"auto"`)
  - Determines how to load the embedding model
  - Options:
    - `"auto"`: Automatically choose based on model type
    - `"api"`: Use API-based embedding (OpenAI, HuggingFace)
    - `"local"`: Use local model loading

- **`--database`** (default: `""`)
  - Vector database storage path
  - Empty string or `":memory:"` for in-memory database
  - Path string for persistent storage (e.g., `"./chroma_db"`)

#### Dataset Configuration

- **`--dataset`** (default: `"embedders_experiments/datasets/med_dareczech"`)
  - Path to dataset directory containing `docs.jsonl`
  - Expected format: JSONL file with `docID`, `source`, and `text` fields

- **`--queries`** (default: `"embedders_experiments/datasets/med_dareczech/queries.jsonl"`)
  - Path to queries file
  - Expected format: JSONL file with `query` and `docID` fields (ground truth)

#### Text Processing

- **`--chunk-size`** (default: `512`)
  - Size of text chunks in characters
  - Documents are split into chunks before embedding

- **`--overlap`** (default: `0`)
  - Number of overlapping characters between consecutive chunks
  - Helps preserve context across chunk boundaries

- **`--first-chunk-only`** (default: `0`)
  - If set to `1`, only the first chunk of each document is used
  - Useful for quick experiments or when documents are short

#### Model Configuration

- **`--device`** (default: `"auto"`)
  - Device for local models: `"auto"`, `"cpu"`, or `"cuda"`
  - `"auto"` uses GPU if available, otherwise CPU

- **`--prompt-mode`** (default: `"none"`)
  - Query prompt formatting strategy
  - Options:
    - `"none"`: No instruction prompts
    - `"custom"`: Use custom instruction text (Czech: "Na základě níže uvedeného dotazu...")
    - `"predefined"`: Use model's built-in prompt (if available)

#### Vector Database Configuration

- **`--distance-metric`** (default: `"cosine"`)
  - Distance metric for similarity search
  - Options:
    - `"cosine"`: Cosine similarity (default, good for normalized embeddings)
    - `"l2"`: Euclidean/L2 distance
    - `"ip"`: Inner product

- **`--batch-size`** (default: `256`)
  - Batch size for adding documents to vector database
  - Smaller batches reduce I/O pressure on network filesystems

#### Output Configuration

- **`--out`** (default: `"embedders_experiments/results/results_med_dareczech.csv"`)
  - Path to output CSV file for aggregated experiment results
  - Per-query results are saved separately in a subdirectory

### What the Script Does

1. **Initialization**
   - Configures logging
   - Parses command-line arguments
   - Validates configuration

2. **Cost Estimation** (for OpenAI models)
   - Estimates API costs before running
   - Checks against budget limit (default: $10 USD)
   - Warns user if cost is significant

3. **Model Loading**
   - Loads the selected embedding model based on `--embedder` and `--source`
   - Supports multiple embedding backends:
     - OpenAI API models
     - SentenceTransformer-compatible models
     - Custom models (Qwen3, Gemma2, EmbeddingGemma, Llama-Embed-Nemotron)
     - HuggingFace Inference API

4. **Vector Database Setup**
   - Creates ChromaDB or Qdrant collection
   - Configures distance metric
   - Sets up embedding function

5. **Document Processing**
   - Loads documents from `docs.jsonl`
   - Chunks documents according to `--chunk-size` and `--overlap`
   - Streams documents to reduce memory usage
   - Adds chunks to vector database in batches

6. **Query Execution**
   - Loads queries from `queries.jsonl`
   - For each query:
     - Embeds the query text
     - Performs similarity search
     - Retrieves top-k results
     - Computes evaluation metrics

7. **Evaluation Metrics**
   - Computes per-query metrics:
     - **MRR@k** (Mean Reciprocal Rank): Measures ranking quality
     - **Recall@k**: Fraction of relevant documents retrieved
     - **Precision@k**: Fraction of retrieved documents that are relevant
   - Aggregates metrics across all queries
   - Saves per-query results to CSV

8. **Results Saving**
   - Appends experiment results to output CSV
   - Includes:
     - Model configuration
     - Performance metrics
     - Timing information (CPU, GPU, total time)
     - Resource usage (GPU memory)
     - Cost information (for API models)
     - Dataset statistics

### Example Usage

```bash
# Run experiment with default settings
python -m embedders_experiments run

# Run with specific embedder and custom chunk size
python -m embedders_experiments run \
    --embedder qwen3 \
    --chunk-size 1024 \
    --overlap 128 \
    --device cuda

# Run with OpenAI API (requires OPENAI_API_KEY)
python -m embedders_experiments run \
    --embedder text-embedding-3-small \
    --source api \
    --database ./chroma_db

# Quick test with first chunk only
python -m embedders_experiments run \
    --first-chunk-only 1 \
    --chunk-size 512
```

---

## Project Architecture

### Core Components

#### 1. **Embedders (`embedders.py`)**
   - Custom embedding function implementations
   - Supports instruction-based query formatting
   - Handles different model architectures:
     - `InstructionalSentenceTransformer`: Base class for instruction-based models
     - `Qwen3EmbeddingFunction`: Qwen3 models with built-in prompts
     - `EmbeddingGemmaEmbeddingFunction`: Google EmbeddingGemma models
     - `GeneralEmbeddingFunction`: General SBERT-style models
     - `LlamaEmbedNemotronEmbeddingFunction`: NVIDIA Llama Embed Nemotron

#### 2. **Vector Database (`vector_db.py`)**
   - Abstract interface for vector databases
   - Supports ChromaDB and Qdrant
   - Handles embedding dimension detection
   - Converts between different distance metrics

#### 3. **Experiments (`experiments.py`)**
   - Main experiment orchestration
   - Manages the full pipeline:
     - Model loading
     - Document chunking and indexing
     - Query execution
     - Metric computation
     - Results saving

#### 4. **Helpers (`helpers.py`)**
   - Text chunking utilities
   - Query execution and evaluation
   - Progress tracking with ETA
   - CSV result saving

#### 5. **Loaders (`loaders.py`)**
   - JSONL file loading
   - Streaming document loader (memory-efficient)
   - Query loading with ground truth

#### 6. **Metrics (`metrics.py`)**
   - Evaluation metric implementations:
     - MRR@k (Mean Reciprocal Rank)
     - Recall@k
     - Precision@k

#### 7. **Cost Tracker (`cost_tracker.py`)**
   - OpenAI API cost estimation
   - Budget limit checking
   - Token counting (approximate and exact)

#### 8. **Constants (`constants.py`)**
   - Registry of available embedders
   - Model name mappings
   - SBERT compatibility checks

---

## Supported Embedding Models

### Local Models (SentenceTransformer-compatible)
- `minilm`: `sentence-transformers/all-MiniLM-L6-v2`
- `e5`: `intfloat/multilingual-e5-large-instruct`
- `fernet`: `fav-kky/FERNET-C5` (Czech)
- `mpnet`: `all-mpnet-base-v2`
- `paraphrase`: `sentence-transformers/paraphrase-xlm-r-multilingual-v1`

### GPU-Required Local Models
- `gemma2`: `BAAI/bge-multilingual-gemma2`
- `qwen3`: `Qwen/Qwen3-Embedding-8B`
- `qwen3-4b`: `Qwen/Qwen3-Embedding-4B`
- `llama-embed-nemotron`: `nvidia/llama-embed-nemotron-8b`

### API Models
- `text-embedding-3-small`: OpenAI embedding API
- `text-embedding-3-large`: OpenAI embedding API
- HuggingFace Inference API (via `--source api`)

---

## Evaluation Metrics

The framework computes standard information retrieval metrics:

### MRR@k (Mean Reciprocal Rank)
- Measures the quality of ranking
- Formula: `1 / rank_of_first_relevant_document`
- Range: [0, 1], higher is better

### Recall@k
- Fraction of relevant documents retrieved in top-k
- Formula: `relevant_retrieved / total_relevant`
- Range: [0, 1], higher is better

### Precision@k
- Fraction of retrieved documents that are relevant
- Formula: `relevant_retrieved / k`
- Range: [0, 1], higher is better

### Default k Values
- Evaluated at: k = 1, 3, 5, 10, 20, 50, 100

---

## Data Format

### Documents (`docs.jsonl`)
Each line is a JSON object:
```json
{
  "docID": "doc_123",
  "source": "source_name",
  "text": "Full document text..."
}
```

### Queries (`queries.jsonl`)
Each line is a JSON object:
```json
{
  "query": "Search query text",
  "docID": "doc_123"  // or ["doc_123", "doc_456"] for multiple relevant docs
}
```

---

## Output Files

### Aggregated Results (`--out` CSV)
Contains one row per experiment with:
- Model configuration (embedder, model_name, source)
- Dataset configuration (chunk_size, overlap, num_docs, num_chunks)
- Performance metrics (MRR@k, Recall@k, Precision@k for all k values)
- Timing information (experiment_time, docs_load_time, embedder_load_time)
- Resource usage (CPU_time, GPU_time, GPU_memory)
- Cost information (estimated_cost, actual_cost, token counts)
- Database configuration (database, distance_metric, prompt_mode)

### Per-Query Results
Saved to: `{out_csv.parent}/{dataset_name}/{dataset_name}_{embedder_key}.csv`

Contains one row per query with:
- Query ID and text
- Correct document IDs
- Retrieved document IDs
- Per-query metrics (MRR@k, Recall@k, Precision@k)
- Query latency
- Sample retrieved documents

---

## Cost Tracking (OpenAI Models)

For OpenAI API models, the framework:
1. **Estimates cost** before running (scans dataset files)
2. **Checks budget limit** (default: $10 USD, configurable via `OPENAI_BUDGET_LIMIT`)
3. **Warns user** if cost exceeds $1 USD
4. **Tracks actual cost** after processing
5. **Saves cost breakdown** in results CSV

### Pricing (as of 2024)
- `text-embedding-3-small`: $0.02 per 1M tokens
- `text-embedding-3-large`: $0.13 per 1M tokens
- `text-embedding-ada-002`: $0.10 per 1M tokens

---

## Performance Tracking

The framework tracks:
- **Total experiment time**: Wall-clock time for entire experiment
- **Document loading time**: Time to chunk and index documents
- **Embedder load time**: Time to initialize the embedding model
- **CPU time**: Process CPU time
- **GPU time**: GPU computation time (if available)
- **GPU memory**: Peak allocated and reserved memory (if available)
- **Query latency**: Average time per query

---

## Key Features

1. **Memory Efficiency**
   - Streaming document loading
   - Batch processing for vector database operations
   - Optional first-chunk-only mode for quick experiments

2. **Flexibility**
   - Multiple embedding backends (local, API)
   - Multiple vector databases (ChromaDB, Qdrant)
   - Configurable distance metrics
   - Custom prompt modes

3. **Comprehensive Evaluation**
   - Multiple evaluation metrics
   - Per-query and aggregated results
   - Detailed performance tracking

4. **Cost Management**
   - Cost estimation before execution
   - Budget limit enforcement
   - Actual cost tracking

5. **Reproducibility**
   - All configuration saved in results CSV
   - Deterministic chunk IDs
   - Consistent evaluation methodology

---

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI embedding models
- `OPENAI_BUDGET_LIMIT`: Budget limit in USD (default: 10.0)
- `CHROMA_HUGGINGFACE_API_KEY` or `HUGGINGFACE_HUB_TOKEN`: For HuggingFace API
- `DISABLE_TQDM`: Disable progress bars (if set to "1")
- `TOKENIZERS_PARALLELISM`: Control tokenizer parallelism

---

## Dependencies

Key dependencies:
- `chromadb`: Vector database
- `sentence-transformers`: Local embedding models
- `torch`: PyTorch for GPU support
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `tqdm`: Progress bars
- `qdrant-client`: Optional, for Qdrant support

---

## Use Cases

1. **Model Comparison**: Compare different embedding models on the same dataset
2. **Hyperparameter Tuning**: Test different chunk sizes, overlap, distance metrics
3. **Cost Analysis**: Evaluate cost-effectiveness of API vs. local models
4. **Performance Benchmarking**: Measure retrieval performance and latency
5. **Research**: Reproducible experiments for academic research

---


