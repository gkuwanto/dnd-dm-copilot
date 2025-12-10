"""Tests for evaluation metrics."""

from typing import Dict, List

from dnd_dm_copilot.evaluation.metrics import (
    calculate_accuracy_at_k,
    calculate_answer_accuracy,
    calculate_mrr,
    compute_retrieval_metrics,
)


class TestAccuracyAtK:
    """Tests for Accuracy@k metric."""

    def test_accuracy_at_1_all_correct(self) -> None:
        """Test Accuracy@1 when all retrievals are correct."""
        retrieved_ranks = [0, 0, 0]  # All source passages at rank 0

        accuracy = calculate_accuracy_at_k(retrieved_ranks, k=1)

        assert accuracy == 1.0

    def test_accuracy_at_1_none_correct(self) -> None:
        """Test Accuracy@1 when no retrievals are correct."""
        retrieved_ranks = [5, 10, None]  # All source passages beyond rank 1

        accuracy = calculate_accuracy_at_k(retrieved_ranks, k=1)

        assert accuracy == 0.0

    def test_accuracy_at_3_partial(self) -> None:
        """Test Accuracy@3 with partial correctness."""
        retrieved_ranks = [0, 2, 5, None, 1]  # 3 out of 5 within top-3

        accuracy = calculate_accuracy_at_k(retrieved_ranks, k=3)

        assert accuracy == 0.6

    def test_accuracy_at_10_with_none(self) -> None:
        """Test Accuracy@10 with None values (not found)."""
        retrieved_ranks = [0, 1, 2, None, None]  # 3 out of 5 found

        accuracy = calculate_accuracy_at_k(retrieved_ranks, k=10)

        assert accuracy == 0.6

    def test_accuracy_empty_list(self) -> None:
        """Test that empty list returns 0.0."""
        retrieved_ranks: List[int | None] = []

        accuracy = calculate_accuracy_at_k(retrieved_ranks, k=1)

        assert accuracy == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank (MRR) metric."""

    def test_mrr_all_rank_zero(self) -> None:
        """Test MRR when all retrievals are at rank 0."""
        retrieved_ranks = [0, 0, 0]

        mrr = calculate_mrr(retrieved_ranks)

        assert mrr == 1.0

    def test_mrr_varied_ranks(self) -> None:
        """Test MRR with varied ranks."""
        retrieved_ranks = [0, 1, 4]  # RR = 1.0, 0.5, 0.2

        mrr = calculate_mrr(retrieved_ranks)

        assert abs(mrr - (1.0 + 0.5 + 0.2) / 3) < 1e-6

    def test_mrr_with_none(self) -> None:
        """Test MRR with None values (not found)."""
        retrieved_ranks = [0, None, 2]  # RR = 1.0, 0.0, 0.333...

        mrr = calculate_mrr(retrieved_ranks)

        assert abs(mrr - (1.0 + 0.0 + 1 / 3) / 3) < 1e-6

    def test_mrr_all_none(self) -> None:
        """Test MRR when no passages are found."""
        retrieved_ranks = [None, None, None]

        mrr = calculate_mrr(retrieved_ranks)

        assert mrr == 0.0

    def test_mrr_empty_list(self) -> None:
        """Test MRR with empty list."""
        retrieved_ranks: List[int | None] = []

        mrr = calculate_mrr(retrieved_ranks)

        assert mrr == 0.0


class TestComputeRetrievalMetrics:
    """Tests for computing all retrieval metrics."""

    def test_compute_retrieval_metrics_all_ranks(self) -> None:
        """Test computing metrics with all rank positions."""
        retrieved_ranks = [0, 1, 2, 4, 9]

        metrics = compute_retrieval_metrics(retrieved_ranks)

        assert metrics["accuracy@1"] == 0.2  # Only first one
        assert metrics["accuracy@3"] == 0.6  # First three
        assert metrics["accuracy@5"] == 0.8  # First four (0,1,2,4)
        assert metrics["accuracy@10"] == 1.0  # All five
        assert "mrr" in metrics
        assert 0.0 <= metrics["mrr"] <= 1.0

    def test_compute_retrieval_metrics_with_none(self) -> None:
        """Test computing metrics with None values."""
        retrieved_ranks = [0, None, 2, None, 4]

        metrics = compute_retrieval_metrics(retrieved_ranks)

        assert metrics["accuracy@1"] == 0.2  # Only first
        assert metrics["accuracy@3"] == 0.4  # 0, 2
        assert metrics["accuracy@10"] == 0.6  # 0, 2, 4
        assert "mrr" in metrics

    def test_compute_retrieval_metrics_custom_k_values(self) -> None:
        """Test computing metrics with custom k values."""
        retrieved_ranks = [0, 1, 10, 20]

        metrics = compute_retrieval_metrics(retrieved_ranks, k_values=[1, 5, 15])

        assert "accuracy@1" in metrics
        assert "accuracy@5" in metrics
        assert "accuracy@15" in metrics
        assert "accuracy@3" not in metrics  # Not in custom k_values


class TestAnswerAccuracy:
    """Tests for answer accuracy metric."""

    def test_answer_accuracy_all_correct(self) -> None:
        """Test answer accuracy when all answers are correct."""
        judgments = [
            {"correct": True, "confidence": 1.0},
            {"correct": True, "confidence": 0.9},
            {"correct": True, "confidence": 0.8},
        ]

        accuracy = calculate_answer_accuracy(judgments)

        assert accuracy == 1.0

    def test_answer_accuracy_none_correct(self) -> None:
        """Test answer accuracy when no answers are correct."""
        judgments = [
            {"correct": False, "confidence": 1.0},
            {"correct": False, "confidence": 0.9},
        ]

        accuracy = calculate_answer_accuracy(judgments)

        assert accuracy == 0.0

    def test_answer_accuracy_partial(self) -> None:
        """Test answer accuracy with partial correctness."""
        judgments = [
            {"correct": True, "confidence": 1.0},
            {"correct": False, "confidence": 0.9},
            {"correct": True, "confidence": 0.8},
            {"correct": False, "confidence": 0.7},
        ]

        accuracy = calculate_answer_accuracy(judgments)

        assert accuracy == 0.5

    def test_answer_accuracy_empty_list(self) -> None:
        """Test answer accuracy with empty list."""
        judgments: List[Dict[str, float | bool]] = []

        accuracy = calculate_answer_accuracy(judgments)

        assert accuracy == 0.0

    def test_answer_accuracy_ignores_confidence(self) -> None:
        """Test that confidence scores don't affect accuracy calculation."""
        judgments = [
            {"correct": True, "confidence": 0.1},
            {"correct": False, "confidence": 1.0},
        ]

        accuracy = calculate_answer_accuracy(judgments)

        assert accuracy == 0.5
