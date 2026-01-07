#!/usr/bin/env python3
"""
BM25 standalone experiment runner for Czech text retrieval.

This script runs BM25 retrieval on a dataset and outputs results in the same
format as embedding-based retrieval experiments for direct comparison.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Handle imports - support both direct execution and module execution
try:
    # Try relative imports first (when run as module)
    from .bm25_code import CzechBM25Retriever
    from ..helpers import save_experiment_results
    from ..loaders import load_docs, load_queries
    from ..metrics import (
        compute_mrr_at_k,
        compute_precision_at_k,
        compute_recall_at_k,
    )
except ImportError:
    # Fall back to absolute imports (when run directly)
    # Add parent directories to path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    sys.path.insert(0, str(project_root))
    
    from embedders_experiments.bm25.bm25_code import CzechBM25Retriever
    from embedders_experiments.helpers import save_experiment_results
    from embedders_experiments.loaders import load_docs, load_queries
    from embedders_experiments.metrics import (
        compute_mrr_at_k,
        compute_precision_at_k,
        compute_recall_at_k,
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def run_bm25_experiment(
    dataset_dir: Path,
    queries_file: Path,
    out_csv: Path,
    mode: str = "udpipe",
    k1: float = 1.5,
    b: float = 0.75,
) -> dict:
    """
    Run BM25 retrieval experiment.

    Args:
        dataset_dir: Path to dataset directory
        queries_file: Path to queries JSONL file
        out_csv: Output CSV path for results
        mode: Preprocessing mode ("udpipe", "stem", or "none")
        k1: BM25 k1 parameter
        b: BM25 b parameter

    Returns:
        Dictionary of computed metrics
    """
    experiment_start = time.time()

    logger.info("=" * 80)
    logger.info("BM25 Standalone Experiment")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Queries: {queries_file}")
    logger.info(f"Output: {out_csv}")
    logger.info(f"Mode: {mode}")
    logger.info(f"BM25 k1: {k1}, b: {b}")

    # Load documents
    docs_path = dataset_dir / "docs.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents file not found: {docs_path}")

    logger.info("Loading documents...")
    docs_raw = load_docs(docs_path)
    logger.info(f"Loaded {len(docs_raw)} documents")

    # Convert to format expected by CzechBM25Retriever
    docs = []
    for doc in docs_raw:
        docs.append({"id": str(doc["docID"]), "text": doc.get("text", "")})

    # Load queries
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    logger.info("Loading queries...")
    queries = load_queries(queries_file)
    logger.info(f"Loaded {len(queries)} queries")

    # Initialize retriever and build index
    logger.info("Initializing BM25 retriever...")
    t0_index = time.time()
    retriever = CzechBM25Retriever(docs=docs, mode=mode, k1=k1, b=b)
    t1_index = time.time()
    index_build_time = t1_index - t0_index
    logger.info(f"Index built in {index_build_time:.2f} seconds")

    # Run queries and compute metrics
    logger.info("Running queries...")
    ks = [1, 3, 5, 10]
    metrics = {f"mrr@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})
    metrics.update({f"precision@{k}": [] for k in ks})
    total_query_time = 0.0

    for q_idx, (qtext, correct_docs) in enumerate(queries):
        # Ensure correct_docs is a list
        if not isinstance(correct_docs, (list, tuple)):
            correct_docs = [correct_docs]
        correct_docs = [str(cd) for cd in correct_docs]

        # Run query
        t_start = time.perf_counter()
        results = retriever.query(query_text=qtext, top_k=10)
        elapsed = time.perf_counter() - t_start
        total_query_time += elapsed

        # Extract retrieved document IDs
        retrieved_doc_ids = [doc_id for doc_id, score in results]

        # Compute metrics for this query
        for k in ks:
            metrics[f"mrr@{k}"].append(
                compute_mrr_at_k(retrieved_doc_ids, correct_docs, k)
            )
            metrics[f"recall@{k}"].append(
                compute_recall_at_k(retrieved_doc_ids, correct_docs, k)
            )
            metrics[f"precision@{k}"].append(
                compute_precision_at_k(retrieved_doc_ids, correct_docs, k)
            )

        if (q_idx + 1) % 10 == 0:
            logger.info(
                f"Processed {q_idx + 1}/{len(queries)} queries "
                f"(avg latency: {total_query_time / (q_idx + 1):.4f}s)"
            )

    # Average metrics across all queries
    averaged_metrics = {m: float(np.mean(v)) for m, v in metrics.items()}
    averaged_metrics["avg_latency_sec"] = float(total_query_time / len(queries))

    logger.info("=" * 80)
    logger.info("Evaluation Results:")
    for k in ks:
        logger.info(
            f"  Precision@{k}: {averaged_metrics[f'precision@{k}']:.4f}, "
            f"Recall@{k}: {averaged_metrics[f'recall@{k}']:.4f}, "
            f"MRR@{k}: {averaged_metrics[f'mrr@{k}']:.4f}"
        )
    logger.info(f"  Average latency: {averaged_metrics['avg_latency_sec']:.4f}s")
    logger.info("=" * 80)

    # Save results
    logger.info("Saving results...")
    experiment_time = time.time() - experiment_start
    save_experiment_results(
        embedder_key="bm25",
        model_name="BM25",
        source="local",
        database="",
        chunk_size=0,
        overlap=0,
        num_docs=len(docs),
        num_chunks=len(docs),  # BM25 has no chunking
        experiment_time=experiment_time,
        docs_load_time_sec=index_build_time,
        embedder_load_time_sec=None,
        metrics=averaged_metrics,
        out_csv=out_csv,
        use_first_chunk_only=False,
        prompt_mode="none",
    )

    logger.info(f"Results saved to {out_csv}")
    return averaged_metrics


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run BM25 retrieval experiment on Czech text dataset"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/tmp_dareczech"),
        help="Path to dataset directory (default: /datasets/med_dareczech)",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to queries file (default: {dataset}/queries.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/results_bm25", # later modify to add +_dataset.csv
        help="Output CSV path (default: embedders_experiments/results/results_bm25.csv)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="udpipe",
        choices=["udpipe", "stem", "none"],
        help='Preprocessing mode: "udpipe" (default) for lemmatization, "stem" for stemming, or "none" for basic tokenization only (no stemming/lemmatization)',
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Trim each document to this many characters before indexing (default: full document)",
    )

    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5, recommended range [1.2-1.5])",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75, recommended range [0.7-0.8])",
    )

    args = parser.parse_args()

    # Resolve paths relative to current working directory
    # Assume script is run from project root
    dataset_dir = args.dataset if args.dataset.is_absolute() else Path.cwd() / args.dataset

    # Default queries file
    if args.queries is None:
        queries_file = dataset_dir / "queries.jsonl"
    else:
        queries_file = args.queries if args.queries.is_absolute() else Path.cwd() / args.queries

    # Output CSV
    out_csv = Path((str(args.out) +  f"_{str(args.dataset).split('/')[-1]}.csv"))


    # Run experiment
    try:
        metrics = run_bm25_experiment(
            dataset_dir=dataset_dir,
            queries_file=queries_file,
            out_csv=out_csv,
            mode=args.mode,
            k1=args.k1,
            b=args.b,
        )
        logger.info("Experiment completed successfully!")
        return 0
    except Exception as e:
        logger.exception("Experiment failed with error:")
        return 1


if __name__ == "__main__":
    exit(main())

