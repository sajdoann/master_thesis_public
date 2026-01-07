from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from pypika.enums import Boolean

from embedders_experiments.constants import EMBEDDERS
from embedders_experiments.experiments import run_experiment
from embedders_experiments.logging_config import configure_logging


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embedder", choices=EMBEDDERS.keys(), default="einfra")
    p.add_argument("--source", choices=["auto", "api", "local"], default="auto")
    p.add_argument("--database", default="") # option :memory:
    p.add_argument("--dataset", type=Path, default=Path("embedders_experiments/datasets/med_dareczech"))
    p.add_argument("--queries", type=Path, default=Path("embedders_experiments/datasets/med_dareczech/queries.jsonl"))
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("embedders_experiments/results/results_med_dareczech.csv"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--prompt-mode",
        choices=["none", "custom", "predefined"],
        default="none",
        help="Choose query prompt: none, custom instruction, or predefined model prompt.",
    )
    p.add_argument("--first_chunk_only", type=int,default=0, help="Use only the first chunk per doc (0 - not true, 1 true just 1st chunk)")
    p.add_argument("--distance-metric", type=str, default="cosine", choices=["cosine", "l2", "ip"],
                   help="Distance metric for ChromaDB collection: cosine (default), l2 (Euclidean), or ip (inner product)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size for adding documents to ChromaDB (default: 256)")
    p.add_argument("--db-type", type=str, default="chroma", choices=["chroma", "qdrant"],
                                      help = "Type of vector database: chroma (default) or qdrant")


def cmd_run(args: argparse.Namespace) -> None:
    run_experiment(
        embedder_key=args.embedder,
        source=args.source,
        dataset_dir=args.dataset,
        queries_file=args.queries,
        database=args.database,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        out_csv=args.out,
        device=args.device,
        prompt_mode=args.prompt_mode,
        use_first_chunk_only=False if not args.first_chunk_only else True,
        distance_metric=args.distance_metric,
        batch_size=args.batch_size,
        db_type=args.db_type,
    )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedders_experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run single experiment")
    _add_common_args(run_p)
    run_p.set_defaults(func=cmd_run)

    return parser


def main(argv: List[str] | None = None) -> None:
    configure_logging(logging.INFO)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


