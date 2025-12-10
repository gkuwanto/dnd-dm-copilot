"""
Evaluation metrics for RAG system.

Includes:
- Retrieval metrics: Accuracy@k, MRR (Mean Reciprocal Rank)
- Generation metrics: Answer accuracy
"""

from typing import Dict, List, Optional


def calculate_accuracy_at_k(retrieved_ranks: List[Optional[int]], k: int) -> float:
    """
    Calculate Accuracy@k metric.

    Measures the proportion of queries where the correct passage
    was retrieved in the top-k results.

    Args:
        retrieved_ranks: List of ranks where source passage was found
                        (0-indexed), or None if not found in top-k
        k: Cutoff rank for accuracy calculation

    Returns:
        Accuracy@k score between 0.0 and 1.0
    """
    if not retrieved_ranks:
        return 0.0

    correct_at_k = sum(1 for rank in retrieved_ranks if rank is not None and rank < k)
    return correct_at_k / len(retrieved_ranks)


def calculate_mrr(retrieved_ranks: List[Optional[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR is the average of reciprocal ranks of the first correct answer.
    For a rank r (0-indexed), reciprocal rank = 1 / (r + 1).
    If the correct passage is not found, reciprocal rank = 0.

    Args:
        retrieved_ranks: List of ranks where source passage was found
                        (0-indexed), or None if not found

    Returns:
        MRR score between 0.0 and 1.0
    """
    if not retrieved_ranks:
        return 0.0

    reciprocal_ranks = [
        1.0 / (rank + 1) if rank is not None else 0.0 for rank in retrieved_ranks
    ]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_retrieval_metrics(
    retrieved_ranks: List[Optional[int]], k_values: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute all retrieval metrics.

    Args:
        retrieved_ranks: List of ranks where source passage was found
                        (0-indexed), or None if not found
        k_values: List of k values for Accuracy@k (default: [1, 3, 5, 10])

    Returns:
        Dictionary with metric names and values
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = {}

    # Calculate Accuracy@k for each k
    for k in k_values:
        metrics[f"accuracy@{k}"] = calculate_accuracy_at_k(retrieved_ranks, k)

    # Calculate MRR
    metrics["mrr"] = calculate_mrr(retrieved_ranks)

    return metrics


def calculate_answer_accuracy(judgments: List[Dict[str, float | bool]]) -> float:
    """
    Calculate answer accuracy from LLM judgments.

    Args:
        judgments: List of judgment dictionaries with 'correct' (bool) field

    Returns:
        Answer accuracy between 0.0 and 1.0
    """
    if not judgments:
        return 0.0

    correct_count = sum(1 for j in judgments if j["correct"])
    return correct_count / len(judgments)
