from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_mrr_at_k, compute_recall_at_k, compute_precision_at_k

logger = logging.getLogger(__name__)


def _supports_instructions(embedder_key: str) -> bool:
    """Return True if the embedder supports instruction-based query formatting."""
    return embedder_key in {"qwen3", "gemma2"}


# ------------------------------
# Chunking
# ------------------------------
def chunk_text(text: str, size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += max(1, size - overlap)
    return chunks

# ------------------------------
# Main evaluation function
# ------------------------------
def run_query(collection, embedder, qtext: str, max_k: int):
    """Execute a query using the embedder if available."""
    if hasattr(embedder, "embed_query"):
        try:
            q_emb = embedder.embed_query(qtext)
            return collection.query(query_embeddings=[q_emb], n_results=max_k)
        except Exception:
            return collection.query(query_texts=[qtext], n_results=max_k)
    else:
        return collection.query(query_texts=[qtext], n_results=max_k)


def extract_doc_ids(results) -> List[str]:
    """Extract doc IDs from query results (falling back to metadata)."""
    ids = results["ids"][0]
    metas_out = results.get("metadatas", [[]])[0] if results is not None else []
    retrieved_doc_ids = []

    for i, rid in enumerate(ids):
        doc_id_from_meta = None
        if i < len(metas_out) and isinstance(metas_out[i], dict):
            doc_id_from_meta = metas_out[i].get("docID")
        if doc_id_from_meta is None:
            parts = str(rid).split("_")
            doc_id_from_meta = parts[1] if len(parts) >= 3 else str(rid)
        retrieved_doc_ids.append(str(doc_id_from_meta))
    return retrieved_doc_ids


def print_query_preview(q_idx: int, qtext: str, correct_docs, results):
    """Print formatted preview of retrieved results for inspection."""
    ids = results["ids"][0]
    scores = results["distances"][0]
    docs_out = results["documents"][0]

    print(f"Q{q_idx}: {qtext} | correct docs: {correct_docs}")
    preview_doc = docs_out[0].strip().replace("\n", " ")[:200] + "..."

    for rank, (rid, score, text) in enumerate(zip(ids, scores, docs_out), start=1):
        preview = text.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "..."
        print(f"\t{rank}. ID={rid} | Score={score:.4f} | Text={preview}")
        logger.debug("%d. ID=%s | Score=%.4f | Text=%s", rank, rid, score, preview)

    return preview_doc


# ------------------------------
# Metric aggregation per query
# ------------------------------

def evaluate_single_query(
    q_idx: int,
    qtext: str,
    correct_docs: List[str],
    results,
    ks: Sequence[int],
) -> Tuple[Dict[str, float], List[str]]:
    """Evaluate one query and compute all metrics."""

    docs_out = results["documents"][0]
    retrieved_doc_ids = extract_doc_ids(results)

    # --- Build a preview for every retrieved doc ---
    sample_docs = []
    for text in docs_out:
        cleaned = text.strip().replace("\n", " ")
        if len(cleaned) > 200:
            cleaned = cleaned[:200] + "..."
        sample_docs.append(cleaned)


    # --- Metrics dictionary ---
    query_metrics = {
        "query_id": q_idx,
        "query_text": qtext,
        "correct_docs": ",".join(map(str, correct_docs)),
        "retrieved_doc_ids": ",".join(retrieved_doc_ids[:max(ks)]),
    }

    # --- Ranking metrics ---
    for k in ks:
        query_metrics[f"mrr@{k}"] = compute_mrr_at_k(retrieved_doc_ids, correct_docs, k)

    query_metrics["sample_docs"] = sample_docs

    for k in ks:
        query_metrics[f"recall@{k}"] = compute_recall_at_k(retrieved_doc_ids, correct_docs, k)
        query_metrics[f"precision@{k}"] = compute_precision_at_k(retrieved_doc_ids, correct_docs, k)

    return query_metrics, retrieved_doc_ids


# ------------------------------
# Main orchestration
# ------------------------------

def run_queries(
    queries: Sequence[Tuple[str, Sequence[object] | object]],
    collection,
    embedder,
    per_query_csv: Path,
    ks: Sequence[int] = (1, 3, 5, 10,20,50,100),
):
    """Run all queries, compute per-query and average metrics, and save results."""
    per_query_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, List[float]] = {f"mrr@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})
    metrics.update({f"precision@{k}": [] for k in ks})
    total_q_time = 0.0
    per_query_rows = []

    for q_idx, (qtext, correct_docs) in enumerate(queries):
        if not isinstance(correct_docs, (list, tuple)):
            correct_docs = [correct_docs]

        max_k = max(ks)
        t_start = time.perf_counter()
        results = run_query(collection, embedder, qtext, max_k)
        elapsed = time.perf_counter() - t_start
        total_q_time += elapsed

        logger.info("Query %d took %.3f sec; correct: %s", q_idx, elapsed, correct_docs)
        print_query_preview(q_idx, qtext, correct_docs, results)

        query_metrics, _ = evaluate_single_query(q_idx, qtext, correct_docs, results, ks)
        query_metrics["latency_sec"] = elapsed

        for k in ks:
            metrics[f"mrr@{k}"].append(query_metrics[f"mrr@{k}"])
            metrics[f"recall@{k}"].append(query_metrics[f"recall@{k}"])
            metrics[f"precision@{k}"].append(query_metrics[f"precision@{k}"])

        per_query_rows.append(query_metrics)

    # Aggregate across all queries
    averaged = {m: float(np.mean(v)) for m, v in metrics.items()}
    averaged["avg_latency_sec"] = float(total_q_time / len(queries))

    logger.info("Evaluation Results: %s", {k: round(v, 3) for k, v in averaged.items()})

    # Save per-query results
    df = pd.DataFrame(per_query_rows)
    df.to_csv(per_query_csv, index=False,quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Saved per-query results to {per_query_csv.resolve()}")

    return averaged



def save_experiment_results(
    *,
    embedder_key: str,
    model_name: str,
    source: str,
    database: str,
    chunk_size: int,
    overlap: int,
    num_docs: int,
    num_chunks: int,
    experiment_time: float | None,
    docs_load_time_sec: float | None,
    embedder_load_time_sec: float | None,
    metrics: Mapping[str, float],
    out_csv: Path,
    use_first_chunk_only: bool,
    prompt_mode: str = "none",
    estimated_cost_usd: float | None = None,
    actual_cost_usd: float | None = None,
    cost_breakdown: Mapping[str, object] | None = None,
    cpu_time_sec: float | None = None,
    gpu_time_sec: float | None = None,
    max_gpu_memory_allocated_mb: float | None = None,
    max_gpu_memory_reserved_mb: float | None = None,
    distance_metric: str = "cosine",
    db_type: str = "chroma",
) -> Mapping[str, float]:
    """Append experiment metadata and metrics to CSV at out_csv, ensuring consistent headers."""


    # --- All possible fields here so headers are always consistent ---
    results: Dict[str, object] = {
        "model_name": model_name,
        # Inject your metric values
        **metrics,
        "embedder": embedder_key,
        "source": source,
        "database": database,
        "db_type": db_type,
        "use_first_chunk_only": use_first_chunk_only,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_docs": num_docs,
        "num_chunks": num_chunks,
        "experiment_time": experiment_time or "none",
        "docs_load_time_sec": docs_load_time_sec or "none",
        "embedder_load_time_sec": embedder_load_time_sec or "none",
        "prompt_mode": prompt_mode,
        "estimated_cost_usd": estimated_cost_usd or "none",
        "actual_cost_usd": actual_cost_usd or "none",
        "estimated_tokens": (
            cost_breakdown.get("total_tokens_est") if cost_breakdown else "none"
        ),
        "doc_tokens_est": (
            cost_breakdown.get("doc_tokens_est") if cost_breakdown else "none"
        ),
        "query_tokens_est": (
            cost_breakdown.get("query_tokens_est") if cost_breakdown else "none"
        ),
        "cpu_time_sec": cpu_time_sec or "none",
        "gpu_time_sec": gpu_time_sec or "none",
        "max_gpu_memory_allocated_mb": max_gpu_memory_allocated_mb or "none",
        "max_gpu_memory_reserved_mb": max_gpu_memory_reserved_mb or "none",
        "distance_metric": distance_metric,
    }

    # --- Write CSV ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_csv.exists()

    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    logger.info("Results appended to %s", out_csv)
    return metrics



def add_with_eta(collection, docs: Sequence[str], ids: Sequence[str], metas: Sequence[Mapping[str, str]], batch_size: int = 500) -> None:
    """Add documents to a Chroma collection in batches, logging progress."""
    import math
    from tqdm import tqdm

    total = len(docs)
    num_batches = math.ceil(total / batch_size)
    logger.info("Adding %d chunks in %d batches", total, num_batches)

    for i in tqdm(range(0, total, batch_size), desc="Indexing"):
        batch_docs = docs[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]
        collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)


def add_streaming_with_eta(
    collection,
    items: Iterable[Tuple[str, str, Mapping[str, str]]],
    batch_size: int = 500,
    total_items: int | None = None,
) -> int:
    """Stream items into a Chroma collection in batches.

    items yields tuples of (document_text, id, metadata).
    Returns the total number of items added.
    """
    import math
    from tqdm import tqdm

    buffer_docs: list[str] = []
    buffer_ids: list[str] = []
    buffer_metas: list[Mapping[str, str]] = []
    added = 0

    # If total is unknown, tqdm can work without total
    pbar = tqdm(total=total_items, desc="Indexing")
    try:
        for doc_text, doc_id, meta in items:
            buffer_docs.append(doc_text)
            buffer_ids.append(doc_id)
            buffer_metas.append(meta)
            if len(buffer_docs) >= batch_size:
                collection.add(documents=buffer_docs, ids=buffer_ids, metadatas=buffer_metas)
                added += len(buffer_docs)
                pbar.update(len(buffer_docs))
                buffer_docs.clear()
                buffer_ids.clear()
                buffer_metas.clear()
        # flush remainder
        if buffer_docs:
            collection.add(documents=buffer_docs, ids=buffer_ids, metadatas=buffer_metas)
            added += len(buffer_docs)
            pbar.update(len(buffer_docs))
    finally:
        pbar.close()

    return added