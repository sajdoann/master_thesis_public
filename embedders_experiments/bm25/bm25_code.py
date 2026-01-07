from __future__ import annotations

import logging
from typing import List

from rank_bm25 import BM25Okapi

from .czech_nlp import CzechPreprocessor

logger = logging.getLogger(__name__)


class CzechBM25Retriever:
    """BM25 retriever for Czech text with preprocessing support."""

    def __init__(
        self,
        docs: List[dict],
        mode: str = "udpipe",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize Czech BM25 retriever.

        Args:
            docs: List of dicts with keys "id" (str) and "text" (str)
            mode: Preprocessing mode ("udpipe", "stem", or "none")
            k1: BM25 k1 parameter (default 1.5, recommended range [1.2-1.5])
            b: BM25 b parameter (default 0.75, recommended range [0.7-0.8])
        """
        if not docs:
            raise ValueError("docs list cannot be empty")

        self.mode = mode
        self.k1 = k1
        self.b = b

        # Initialize preprocessor
        logger.info(f"Initializing CzechPreprocessor with mode={mode}")
        self.preprocessor = CzechPreprocessor(mode=mode)

        # Extract document IDs and texts
        self.doc_ids = []
        doc_texts = []

        for doc in docs:
            if "id" not in doc or "text" not in doc:
                raise ValueError("Each doc must have 'id' and 'text' keys")
            self.doc_ids.append(str(doc["id"]))
            doc_texts.append(doc["text"])

        logger.info(f"Preprocessing {len(doc_texts)} documents...")
        # Preprocess all document texts
        tokenized_docs = []
        for i, text in enumerate(doc_texts):
            if i % 100 == 0 and i > 0:
                logger.info(f"Preprocessed {i}/{len(doc_texts)} documents...")
            tokens = self.preprocessor.preprocess(text)
            tokenized_docs.append(tokens)

        logger.info("Building BM25 index...")
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)

        logger.info(f"BM25 index built with {len(self.doc_ids)} documents")

    def query(self, query_text: str, top_k: int = 10) -> List[tuple[str, float]]:
        """
        Query the BM25 index.

        Args:
            query_text: Query text string
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples sorted by descending BM25 score
        """
        # Preprocess query
        query_tokens = self.preprocessor.preprocess(query_text)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Create list of (doc_id, score) tuples
        doc_scores = list(zip(self.doc_ids, scores))

        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return doc_scores[:top_k]

