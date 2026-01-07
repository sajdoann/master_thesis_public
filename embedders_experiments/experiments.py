from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions

from .constants import EMBEDDERS, is_sbert_compatible, is_openai_model
from .embedders import GeneralEmbeddingFunction, Qwen3EmbeddingFunction, LlamaEmbedNemotronEmbeddingFunction, \
    EmbeddingGemmaEmbeddingFunction
from .helpers import add_with_eta, add_streaming_with_eta, chunk_text, run_queries, save_experiment_results
from .loaders import load_docs, load_docs_stream, load_queries
from .cost_tracker import (
    estimate_cost_from_dataset,
    calculate_actual_cost,
    check_budget_limit,
    DEFAULT_BUDGET_LIMIT,
)
from .vector_db import create_vector_db

logger = logging.getLogger(__name__)


def run_experiment(
        *,  # ensures keyword only args
        embedder_key: str,
        source: str,
        dataset_dir: Path,
        queries_file: Path,
        database: str,
        chunk_size: int,
        overlap: int,
        out_csv: Path,
        prompt_mode: str = "none",
        device: str = "auto",
        use_first_chunk_only: bool = False,
        distance_metric: str = "cosine",
        batch_size: int = 500,
        db_type: str = "chroma",
) -> Mapping[str, float]:
    """Run a retrieval experiment and append results to CSV.

    Args:
        distance_metric: Distance metric for vector database collection. Options: "cosine", "l2", "ip" (default: "cosine")
            - "cosine": Cosine similarity (default, good for normalized embeddings)
            - "l2": Euclidean/L2 distance
            - "ip": Inner product
        batch_size: Batch size for adding documents to vector database (default: 500).
            Smaller batch sizes reduce I/O pressure and can help avoid disk I/O errors on network filesystems.
        db_type: Type of vector database to use. Options: "chroma", "qdrant" (default: "chroma")

    Returns the computed metrics mapping.
    """
    t_experiment_0 = time.time()
    cpu_time_0 = time.process_time()

    # Initialize GPU time tracking if available
    gpu_time_0 = None
    gpu_time_1 = None
    gpu_memory_stats = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure GPU operations are synchronized
            gpu_time_0 = torch.cuda.Event(enable_timing=True)
            gpu_time_0.record()
            # Reset memory stats to track peak usage during experiment
            torch.cuda.reset_peak_memory_stats()
            gpu_memory_stats = True
    except ImportError:
        pass
    if prompt_mode not in {"none", "custom", "predefined"}:
        raise ValueError(f"Invalid prompt_mode '{prompt_mode}'. Expected one of none/custom/predefined.")
    add_instruction = prompt_mode != "none"
    use_builtin_prompt = prompt_mode == "predefined"

    if embedder_key not in EMBEDDERS:
        raise ValueError(f"Unknown embedder key: {embedder_key}")

    model_name = EMBEDDERS[embedder_key]
    logger.info("Using embedder %s (%s)", embedder_key, model_name)

    docs_path = dataset_dir / "docs.jsonl"
    queries_path = queries_file

    # Cost estimation and budget check for OpenAI models
    estimated_cost = None
    cost_breakdown = None
    if is_openai_model(embedder_key):
        logger.info("Estimating cost for OpenAI embedding API...")
        try:
            estimated_cost, cost_breakdown = estimate_cost_from_dataset(
                model_name=model_name,
                docs_path=str(docs_path),
                queries_path=str(queries_path),
                chunk_size=chunk_size,
                overlap=overlap,
                use_first_chunk_only=use_first_chunk_only,
                use_exact_count=False,  # Set to True if tiktoken is installed
            )

            logger.info("Cost estimation:")
            logger.info(f"  Model: {cost_breakdown['model']}")
            logger.info(f"  Document chunks: {cost_breakdown['doc_chunks']:,}")
            logger.info(f"  Estimated doc tokens: {cost_breakdown['doc_tokens_est']:,}")
            logger.info(f"  Queries: {cost_breakdown['queries']:,}")
            logger.info(f"  Estimated query tokens: {cost_breakdown['query_tokens_est']:,}")
            logger.info(f"  Total estimated tokens: {cost_breakdown['total_tokens_est']:,}")
            logger.info(f"  Estimated cost: ${estimated_cost:.4f} USD")

            # Check budget limit
            within_budget, budget_message = check_budget_limit(estimated_cost)
            logger.info(budget_message)

            if not within_budget:
                print("\n" + "=" * 70)
                print(budget_message)
                print("=" * 70 + "\n")
                raise ValueError(
                    f"Estimated cost ${estimated_cost:.2f} exceeds budget limit. "
                    f"Set OPENAI_BUDGET_LIMIT environment variable to override."
                )

            # Warn if cost is significant
            if estimated_cost > 1.0:
                print(f"\n⚠️  WARNING: This operation will cost approximately ${estimated_cost:.2f} USD")
                print("   Press Ctrl+C to cancel if needed.\n")
            elif estimated_cost > 0.1:
                print(f"\n💡 Note: Estimated cost: ${estimated_cost:.2f} USD\n")

        except Exception as e:
            logger.warning(f"Cost estimation failed: {e}")
            logger.warning("Continuing without cost check. Use caution with large datasets.")

    # Stream queries fully (usually smaller) and docs as a stream for memory safety
    queries = load_queries(queries_path)

    # Silence internal tqdm bars from transformers/sentence-transformers
    # os.environ.setdefault("DISABLE_TQDM", "1")
    # os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Select embedding function
    t0_embedder = time.time()
    try:
        print(f"model_name: {model_name}, embedder key {embedder_key}")
        load_dotenv()

        # 1) OpenAI models (API only)
        if is_openai_model(embedder_key):
            logger.info("Using OpenAI Embedding API")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise EnvironmentError("Missing OpenAI API key (set OPENAI_API_KEY environment variable)")
            embedder = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=model_name,
            )

        # 2) SBERT-compatible models (local only)
        elif is_sbert_compatible(model_name) and source in {"auto", "local"}:
            logger.info("Using SentenceTransformerEmbeddingFunction (SBERT path)")
            embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                normalize_embeddings=True,
            )

        # 3) Qwen local embedding
        elif embedder_key in {"qwen3", "qwen3-4b"} and source in {"auto", "local"}:
            logger.info("Using local QwenEmbeddingFunction")
            embedder = Qwen3EmbeddingFunction(
                model_name=model_name,
                device=device,
                prompt_mode=prompt_mode,
            )

        # 4) Gemma2 local embedding
        elif "gemma2" == embedder_key and source in {"auto", "local"}:
            logger.info(f"Using local {model_name} embedding function")
            embedder = GeneralEmbeddingFunction(
                model_name=model_name,
                prompt_mode=prompt_mode,
                device=device,
            )

        # 5) EmbeddingGemma local embedding
        elif embedder_key == "embeddinggemma" and source in {"auto", "local"}:
            logger.info(f"Using local {model_name} embedding function")
            embedder = EmbeddingGemmaEmbeddingFunction(
                model_name=model_name,
                prompt_mode=prompt_mode,
                device=device,
            )

        # 6) Llama Embed Nemotron local embedding
        elif embedder_key == "llama-embed-nemotron" and source in {"auto", "local"}:
            logger.info(f"Using local {model_name} embedding function")
            embedder = LlamaEmbedNemotronEmbeddingFunction(
                model_name=model_name,
                device=device,
                prompt_mode=prompt_mode,
            )

        # 7) HuggingFace API path
        elif source == "api":
            logger.info("Using Hugging Face Inference API")
            api_key = os.getenv("CHROMA_HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            if not api_key:
                raise EnvironmentError("Missing Hugging Face API key")
            embedder = embedding_functions.HuggingFaceEmbeddingFunction(
                model_name=model_name,
                api_key=api_key,
            )

        else:
            raise ValueError(
                f"No embedding function configured for embedder_key={embedder_key}, "
                f"model_name={model_name}, source={source}. "
                f"OpenAI models: use embedder_key='text-embedding-3-small', 'text-embedding-3-large', or 'text-embedding-ada-002'."
            )

    except Exception:
        logger.exception("Embedder load failed")
        raise
    t1_embedder = time.time()
    t_embedder = t1_embedder - t0_embedder

    # Setup vector database
    t0 = time.time()

    collection_name = f"{dataset_dir.name}__{embedder_key}"
    collection = create_vector_db(
        db_type=db_type,
        database=database,
        collection_name=collection_name,
        embedder=embedder,
        distance_metric=distance_metric,
    )

    # Chunk and add docs (streaming)
    num_docs = 0
    num_chunks = 0

    # Pre-count total chunks to provide a single progress bar with ETA
    def _count_total_chunks() -> int:
        total = 0
        for doc in load_docs_stream(docs_path):
            if use_first_chunk_only:
                total += 1
            else:
                text = doc.get("text", "")
                start = 0
                while start < len(text):
                    total += 1
                    start += max(1, chunk_size - overlap)
        return total

    total_chunks = _count_total_chunks()

    # Track actual characters processed for cost calculation (OpenAI only)
    total_doc_chars_processed = 0

    def iter_chunks():
        nonlocal num_docs, num_chunks, total_doc_chars_processed
        doc_idx = 0
        for doc in load_docs_stream(docs_path):
            num_docs += 1
            orig_id = str(doc["docID"]) if doc.get("docID") is not None else "unknown"
            text = doc.get("text", "")
            chunks = chunk_text(text, size=chunk_size, overlap=overlap)
            if use_first_chunk_only:
                # Only use the first chunk
                if chunks:
                    num_chunks += 1
                    chunk_text_actual = chunks[0]
                    total_doc_chars_processed += len(chunk_text_actual)
                    meta = {"docID": orig_id, "source": str(doc.get("source", "")), "chunk": 0}
                    yield chunk_text_actual, f"{doc_idx}_{orig_id}_0", meta
            else:
                # Use all chunks (original behavior)
                for idx, ch in enumerate(chunks):
                    num_chunks += 1
                    total_doc_chars_processed += len(ch)
                    meta = {"docID": orig_id, "source": str(doc.get("source", "")), "chunk": idx}
                    # Globally unique within dataset run: <doc_idx>_<docID>_<chunk_idx>
                    yield ch, f"{doc_idx}_{orig_id}_{idx}", meta
            doc_idx += 1

    logger.info("Indexing chunks (streaming) to collection")
    added = add_streaming_with_eta(collection, iter_chunks(), batch_size=batch_size, total_items=total_chunks)
    t1 = time.time()
    load_time = t1 - t0

    # Calculate actual cost for OpenAI models
    actual_cost = None
    actual_cost_breakdown = None
    if is_openai_model(embedder_key):
        # Track query characters
        total_query_chars_processed = sum(len(qtext) for qtext, _ in queries)
        num_queries_processed = len(queries)

        try:
            actual_cost, actual_cost_breakdown = calculate_actual_cost(
                model_name=model_name,
                total_doc_chars=total_doc_chars_processed,
                num_chunks=num_chunks,
                total_query_chars=total_query_chars_processed,
                num_queries=num_queries_processed,
                use_exact_count=False,
            )
            logger.info(f"Actual cost calculated: ${actual_cost:.4f} USD")
            logger.info(f"  Processed doc chars: {total_doc_chars_processed:,}")
            logger.info(f"  Processed query chars: {total_query_chars_processed:,}")
            logger.info(f"  Estimated tokens: {actual_cost_breakdown['total_tokens_est']:,}")
        except Exception as e:
            logger.warning(f"Failed to calculate actual cost: {e}")

    # Evaluate
    per_query_csv = (
            out_csv.parent / dataset_dir.name / f"{dataset_dir.name}_{embedder_key}.csv"
    )

    metrics = run_queries(
        queries,
        collection,
        embedder,
        per_query_csv=per_query_csv
    )

    t_experiment_1 = time.time()
    t_experiment = t_experiment_1 - t_experiment_0

    # Calculate CPU time
    cpu_time_1 = time.process_time()
    cpu_time_sec = cpu_time_1 - cpu_time_0
    logger.info(f"CPU time: {cpu_time_sec:.2f} seconds")

    # Calculate GPU time and memory if available
    gpu_time_sec = None
    max_gpu_memory_allocated_mb = None
    max_gpu_memory_reserved_mb = None
    try:
        import torch
        if torch.cuda.is_available() and gpu_time_0 is not None:
            gpu_time_1 = torch.cuda.Event(enable_timing=True)
            gpu_time_1.record()
            torch.cuda.synchronize()  # Wait for all GPU operations to complete
            gpu_time_sec = gpu_time_0.elapsed_time(gpu_time_1) / 1000.0  # Convert ms to seconds
            logger.info(f"GPU time: {gpu_time_sec:.2f} seconds")

            # Get peak memory stats
            if gpu_memory_stats:
                max_gpu_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
                max_gpu_memory_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)  # Convert bytes to MB
                logger.info(f"Max GPU memory allocated: {max_gpu_memory_allocated_mb:.2f} MB")
                logger.info(f"Max GPU memory reserved: {max_gpu_memory_reserved_mb:.2f} MB")
        elif torch.cuda.is_available():
            logger.info("GPU available but timing not initialized")
    except (ImportError, AttributeError):
        logger.debug("PyTorch not available or GPU timing failed")

    prompt_type = None
    if hasattr(embedder, "get_prompt_type"):
        try:
            prompt_type = embedder.get_prompt_type()
        except Exception as exc:
            logger.warning("Failed to retrieve prompt type: %s", exc)

    # Save
    save_experiment_results(
        embedder_key=embedder_key,
        model_name=model_name,
        source=source,
        database=database,
        chunk_size=chunk_size,
        overlap=overlap,
        num_docs=num_docs,
        num_chunks=num_chunks,
        experiment_time=t_experiment,
        docs_load_time_sec=load_time,
        embedder_load_time_sec=t_embedder,
        metrics=metrics,
        out_csv=out_csv,
        use_first_chunk_only=use_first_chunk_only,
        estimated_cost_usd=estimated_cost,
        actual_cost_usd=actual_cost,
        cost_breakdown=actual_cost_breakdown,
        cpu_time_sec=cpu_time_sec,
        gpu_time_sec=gpu_time_sec,
        max_gpu_memory_allocated_mb=max_gpu_memory_allocated_mb,
        max_gpu_memory_reserved_mb=max_gpu_memory_reserved_mb,
        distance_metric=distance_metric,
        prompt_mode=prompt_mode,
        db_type=db_type,
    )

    return metrics