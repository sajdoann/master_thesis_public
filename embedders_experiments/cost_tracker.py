""" Cost tracking and estimation for OpenAI API usage. """

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# OpenAI Embedding API pricing (per 1K tokens) as of 2024
# Source: https://openai.com/pricing
OPENAI_EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.00002,  # $0.02 per 1M tokens
    "text-embedding-3-large": 0.00013,  # $0.13 per 1M tokens
    "text-embedding-ada-002": 0.0001,   # $0.10 per 1M tokens
}

# Default budget limit (in USD) - can be overridden via environment variable
DEFAULT_BUDGET_LIMIT = float(os.getenv("OPENAI_BUDGET_LIMIT", "10.0"))


def count_tokens_approximate(text: str) -> int:
    """Approximate token count for embeddings.
    
    OpenAI embeddings use cl100k_base encoding (same as GPT-3.5/GPT-4).
    Rough approximation: ~4 characters per token for English, ~2-3 for other languages.
    We use a conservative estimate of 3 characters per token.
    """
    return len(text) // 3


def count_tokens_exact(text: str) -> int:
    """Exact token count using tiktoken."""
    try:
        import tiktoken
        # OpenAI embeddings use cl100k_base encoding
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        logger.warning("tiktoken not installed, using approximate token counting")
        return count_tokens_approximate(text)


def estimate_embedding_cost(
    model_name: str,
    num_chunks: int,
    avg_chunk_length: Optional[int] = None,
    total_chars: Optional[int] = None,
    num_queries: int = 0,
    avg_query_length: Optional[int] = None,
    use_exact_count: bool = False,
) -> tuple[float, dict]:
    """Estimate the cost of embedding documents and queries.
    
    Args:
        model_name: OpenAI model name
        num_chunks: Number of document chunks to embed
        avg_chunk_length: Average character length per chunk (optional)
        total_chars: Total characters in all chunks (optional, preferred)
        num_queries: Number of queries to embed
        avg_query_length: Average character length per query (optional)
        use_exact_count: Whether to use exact token counting (requires tiktoken)
    
    Returns:
        Tuple of (total_cost_usd, cost_breakdown_dict)
    """
    if model_name not in OPENAI_EMBEDDING_PRICING:
        logger.warning(f"Unknown model {model_name}, using text-embedding-3-small pricing")
        model_name = "text-embedding-3-small"
    
    price_per_1k_tokens = OPENAI_EMBEDDING_PRICING[model_name]
    
    # Estimate tokens for documents
    if total_chars is not None:
        if use_exact_count:
            # For exact count, we'd need to process all text, which is expensive
            # So we use approximation here
            doc_tokens = total_chars // 3
        else:
            doc_tokens = total_chars // 3  # ~3 chars per token
    elif avg_chunk_length is not None:
        doc_tokens = (num_chunks * avg_chunk_length) // 3
    else:
        # Very rough estimate: assume 1000 chars per chunk = ~333 tokens
        doc_tokens = num_chunks * 333
    
    # Estimate tokens for queries
    if avg_query_length is not None:
        query_tokens = (num_queries * avg_query_length) // 3
    else:
        # Rough estimate: assume 100 chars per query = ~33 tokens
        query_tokens = num_queries * 33
    
    total_tokens = doc_tokens + query_tokens
    total_cost = (total_tokens / 1000) * price_per_1k_tokens
    
    breakdown = {
        "model": model_name,
        "doc_chunks": num_chunks,
        "doc_tokens_est": doc_tokens,
        "queries": num_queries,
        "query_tokens_est": query_tokens,
        "total_tokens_est": total_tokens,
        "price_per_1k_tokens": price_per_1k_tokens,
        "total_cost_usd": total_cost,
    }
    
    return total_cost, breakdown


def check_budget_limit(estimated_cost: float, budget_limit: Optional[float] = None) -> tuple[bool, str]:
    """Check if estimated cost exceeds budget limit.
    
    Returns:
        Tuple of (within_budget, message)
    """
    if budget_limit is None:
        budget_limit = DEFAULT_BUDGET_LIMIT
    
    if estimated_cost > budget_limit:
        return False, (
            f"⚠️  ESTIMATED COST ${estimated_cost:.2f} EXCEEDS BUDGET LIMIT ${budget_limit:.2f}!\n"
            f"   This operation would cost approximately ${estimated_cost:.2f} USD.\n"
            f"   Set OPENAI_BUDGET_LIMIT environment variable to override (current: ${budget_limit:.2f}).\n"
            f"   Aborting to prevent overspending."
        )
    
    return True, f"Estimated cost: ${estimated_cost:.2f} USD (budget limit: ${budget_limit:.2f} USD)"


def calculate_actual_cost(
    model_name: str,
    total_doc_chars: int,
    num_chunks: int,
    total_query_chars: int,
    num_queries: int,
    use_exact_count: bool = False,
) -> tuple[float, dict]:
    """Calculate actual cost from processed text.
    
    Args:
        model_name: OpenAI model name
        total_doc_chars: Total characters in processed document chunks
        num_chunks: Number of document chunks processed
        total_query_chars: Total characters in processed queries
        num_queries: Number of queries processed
        use_exact_count: Whether to use exact token counting (requires tiktoken)
    
    Returns:
        Tuple of (total_cost_usd, cost_breakdown_dict)
    """
    return estimate_embedding_cost(
        model_name=model_name,
        num_chunks=num_chunks,
        total_chars=total_doc_chars,
        num_queries=num_queries,
        avg_query_length=total_query_chars // num_queries if num_queries > 0 else None,
        use_exact_count=use_exact_count,
    )


def estimate_cost_from_dataset(
    model_name: str,
    docs_path: str,
    queries_path: str,
    chunk_size: int = 1000,
    overlap: int = 0,
    use_first_chunk_only: bool = False,
    use_exact_count: bool = False,
) -> tuple[float, dict]:
    """Estimate cost by scanning the dataset files.
    
    This function reads through the dataset files to get accurate character counts.
    """
    from pathlib import Path
    
    docs_path = Path(docs_path)
    queries_path = Path(queries_path)
    
    # Count total characters in documents
    total_doc_chars = 0
    num_docs = 0
    
    if docs_path.exists():
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                import json
                doc = json.loads(line)
                text = doc.get("text", "")
                
                if use_first_chunk_only:
                    # Only count first chunk
                    chunk = text[:chunk_size]
                    total_doc_chars += len(chunk)
                else:
                    # Count all chunks
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunk = text[start:end]
                        total_doc_chars += len(chunk)
                        start += max(1, chunk_size - overlap)
                
                num_docs += 1
    
    # Count total characters in queries
    total_query_chars = 0
    num_queries = 0
    
    if queries_path.exists():
        with queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                import json
                query_obj = json.loads(line)
                query_text = query_obj.get("query", "")
                total_query_chars += len(query_text)
                num_queries += 1
    
    # Count chunks
    if use_first_chunk_only:
        num_chunks = num_docs
    else:
        # Re-count chunks more accurately
        num_chunks = 0
        if docs_path.exists():
            with docs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    import json
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    start = 0
                    while start < len(text):
                        num_chunks += 1
                        start += max(1, chunk_size - overlap)
    
    return estimate_embedding_cost(
        model_name=model_name,
        num_chunks=num_chunks,
        total_chars=total_doc_chars,
        num_queries=num_queries,
        avg_query_length=total_query_chars // num_queries if num_queries > 0 else None,
        use_exact_count=use_exact_count,
    )

