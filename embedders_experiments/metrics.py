from typing import List


# ------------------------------
# Metric computation helpers
# ------------------------------

def compute_mrr_at_k(retrieved_doc_ids: List[str], correct_docs: List[str], k: int) -> float:
    """Compute Mean Reciprocal Rank (MRR@k) for a single query."""
    for rank, did in enumerate(retrieved_doc_ids[:k], start=1):
        if any(str(did) == str(cd) for cd in correct_docs):
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(retrieved_doc_ids: List[str], correct_docs: List[str], k: int) -> float:
    """Compute Recall@k for a single query."""
    top_doc_ids = retrieved_doc_ids[:k]
    correct_docs_set = {str(cd) for cd in correct_docs}
    # Count unique relevant documents retrieved (handle duplicates)
    unique_relevant_retrieved = len(set(str(did) for did in top_doc_ids) & correct_docs_set)
    total_relevant = len(correct_docs)
    return unique_relevant_retrieved / total_relevant if total_relevant else 0.0


def compute_precision_at_k(retrieved_doc_ids: List[str], correct_docs: List[str], k: int) -> float:
    """Compute Precision@k for a single query."""
    top_doc_ids = retrieved_doc_ids[:k]
    retrieved_relevant = sum(any(str(did) == str(cd) for cd in correct_docs) for did in top_doc_ids)
    return retrieved_relevant / k if k > 0 else 0.0
