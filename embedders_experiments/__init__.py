"""Utilities for running and evaluating text embedding experiments.

This package provides:
- Embedding backends and factories (see `embedders`)
- Dataset loaders (see `loaders`)
- Experiment orchestration (see `experiments`)
- Metrics and helpers (see `helpers`)

The public API is intentionally small and typed for clarity.
"""

__all__ = [
    "experiments",
    "embedders",
    "helpers",
    "loaders",
]


