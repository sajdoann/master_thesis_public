from __future__ import annotations

from typing import Dict, Set


# Registry of available embedders by short name
EMBEDDERS: Dict[str, str] = {
    # SBERT-compatible models
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/multilingual-e5-large-instruct",
    "fernet": "fav-kky/FERNET-C5",  # Czech
    "mpnet": "all-mpnet-base-v2",
    "paraphrase": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
    "laBSE": "sentence-transformers/LaBSE",

    # GPU NEEDED
    # Local BGE Gemma
    "gemma2": "BAAI/bge-multilingual-gemma2",
    # Google EmbeddingGemma
    #"embeddinggemma": "google/embeddinggemma-300m",
    # Qwen3 Embedding
    "qwen3": "Qwen/Qwen3-Embedding-8B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
    # NVIDIA Llama Embed Nemotron
    "llama-embed-nemotron": "nvidia/llama-embed-nemotron-8b",

    "multilingual_e5_large_instruct": "intfloat/multilingual-e5-large-instruct",
    #"multilingual_e5_large": "intfloat/multilingual-e5-large",

    # OpenAI models
    "text-embedding-3-small": "text-embedding-3-small",
    "text-embedding-3-large": "text-embedding-3-large",
    #"text-embedding-ada-002": "text-embedding-ada-002",

    #note try: all-distilroberta-v1
}


# Known SBERT-compatible model ids
SBERT_MODELS: Set[str] = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-large-instruct",
    "fav-kky/FERNET-C5",
    "all-mpnet-base-v2",
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
}


def is_sbert_compatible(model_name: str) -> bool:
    """Return True if model_name is compatible with SentenceTransformerEmbeddingFunction."""
    return model_name in SBERT_MODELS


def is_openai_model(embedder_key: str) -> bool:
    """Return True if embedder_key is an OpenAI model."""
    return embedder_key in {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}


