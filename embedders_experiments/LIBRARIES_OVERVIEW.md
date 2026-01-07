# Libraries Overview for embedders_experiments/

This document provides a comprehensive overview of all libraries and dependencies used in the `embedders_experiments/` folder.

## Core Dependencies (from requirements.txt)

### Vector Databases
- **chromadb** (v0.5.4) - Vector database for storing and querying embeddings
- **qdrant-client** (>=1.7.0) - Alternative vector database client (optional, for Qdrant backend)

### Machine Learning & Embeddings
- **sentence-transformers** (v3.0.1) - Library for sentence embeddings and transformer models
- **transformers** (v4.44.2) - Hugging Face transformers library for NLP models
- **torch** (>=2.1.0) - PyTorch deep learning framework

### Data Processing
- **numpy** (v1.26.4) - Numerical computing library
- **pandas** (v2.2.2) - Data manipulation and analysis library

### Utilities
- **python-dotenv** (v1.0.1) - Environment variable management from .env files
- **tqdm** (v4.66.4) - Progress bars for loops

## Additional Dependencies (used but not in requirements.txt)

### NLP & Text Processing
- **nltk** - Natural Language Toolkit (used in `bm25/czech_nlp.py` for Czech stopwords)
- **ufal.udpipe** - UDPipe library for Czech language processing (lemmatization)
- **PyStemmer** - Stemming library for Czech text (alternative to UDPipe)
- **rank_bm25** - BM25 ranking algorithm implementation

### API & HTTP
- **requests** - HTTP library for API calls (used in `generate_wikipedia_questions.py`)

### Optional/Optimization Libraries
- **flash_attn** (optional) - Flash Attention for faster transformer inference (used in Qwen3 and Llama models)
- **tiktoken** (optional) - Token counting for OpenAI models (used in `cost_tracker.py`)

### Other
- **pypika** - SQL query builder (imported in `cli.py` but usage unclear)

## Standard Library Modules Used

### Core Python
- `logging` - Logging framework
- `json` - JSON parsing and serialization
- `csv` - CSV file handling
- `os` - Operating system interface
- `time` - Time-related functions
- `pathlib` - Path manipulation
- `argparse` - Command-line argument parsing
- `typing` - Type hints
- `abc` - Abstract base classes
- `re` - Regular expressions
- `random` - Random number generation
- `math` - Mathematical functions

## Library Usage by Module

### embedders.py
- `torch` - PyTorch for GPU/device management
- `chromadb.EmbeddingFunction` - Base class for embedding functions
- `sentence_transformers.SentenceTransformer` - Main embedding model wrapper
- `flash_attn` (optional) - Flash attention for Qwen3 and Llama models

### experiments.py
- `chromadb` - Vector database operations
- `chromadb.utils.embedding_functions` - Built-in embedding functions (OpenAI, SentenceTransformer, HuggingFace)
- `dotenv` - Environment variable loading
- `torch` - GPU timing and memory tracking

### helpers.py
- `numpy` - Numerical operations for metrics
- `pandas` - DataFrame operations for saving results
- `tqdm` - Progress bars for batch processing
- `csv` - CSV writing

### vector_db.py
- `chromadb` - ChromaDB client and collections
- `qdrant_client` - Qdrant client (optional backend)
- `qdrant_client.models` - Qdrant data models (Distance, VectorParams, PointStruct)

### cost_tracker.py
- `tiktoken` (optional) - Exact token counting for OpenAI models

### bm25/bm25_code.py
- `rank_bm25.BM25Okapi` - BM25 ranking implementation

### bm25/czech_nlp.py
- `nltk` - Czech stopwords
- `ufal.udpipe` - Czech lemmatization
- `PyStemmer` - Czech stemming (alternative)

### generate_wikipedia_questions.py
- `requests` - HTTP requests to EINFRA API
- `dotenv` - Environment variable loading

## Model-Specific Dependencies

### Sentence Transformers Models
- Uses `sentence-transformers` library
- Compatible with standard SBERT models

### Qwen3 Models
- Requires `flash_attn` (optional) for optimized attention
- Uses `sentence-transformers` with custom prompt handling

### EmbeddingGemma Models
- Uses `sentence-transformers` with `trust_remote_code=True`
- Requires specific prompt formatting

### Llama Embed Nemotron Models
- Requires `flash_attn` (optional) for optimized attention
- Uses `sentence-transformers` with custom query/document methods

### OpenAI Models
- Uses OpenAI API (via `chromadb.utils.embedding_functions.OpenAIEmbeddingFunction`)
- Optional `tiktoken` for exact token counting

## Installation Notes

### Required Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies
```bash
# For exact OpenAI token counting
pip install tiktoken

# For optimized attention (Qwen3, Llama models)
pip install flash-attn

# For Czech NLP preprocessing
pip install ufal.udpipe PyStemmer nltk
python -c "import nltk; nltk.download('stopwords')"

# For BM25 retrieval
pip install rank-bm25
```

### Czech NLP Setup
- Download Czech UDPipe model: `czech-pdt-ud-2.5-191206.udpipe` or newer
- Place in one of the search paths (see `czech_nlp.py` for details)
- Or download via: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131

## Environment Variables

The following environment variables are used:
- `OPENAI_API_KEY` - OpenAI API key for embedding models
- `EINFRA_API_KEY` - EINFRA API key for question generation
- `EINFRA_API_URL` - EINFRA API endpoint URL
- `CHROMA_HUGGINGFACE_API_KEY` / `HUGGINGFACE_HUB_TOKEN` - Hugging Face API key
- `OPENAI_BUDGET_LIMIT` - Budget limit for OpenAI API calls (default: $10.00)


