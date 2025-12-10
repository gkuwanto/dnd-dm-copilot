"""Tests for RAG pipeline evaluation runner."""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from dnd_dm_copilot.evaluation.run_rag_pipeline import (
    RAGPipelineRunner,
    find_passage_rank,
    load_qa_triplets,
    save_results,
)


class TestLoadQATriplets:
    """Tests for loading QA triplets."""

    def test_load_qa_triplets_valid_file(self, tmp_path: Path) -> None:
        """Test loading QA triplets from valid JSON file."""
        qa_file = tmp_path / "qa_triplets.json"
        triplets = [
            {
                "question": "Q1?",
                "answer": "A1",
                "passage": "P1",
                "metadata": {"source": "test"},
            },
            {
                "question": "Q2?",
                "answer": "A2",
                "passage": "P2",
                "metadata": {"source": "test"},
            },
        ]
        qa_file.write_text(json.dumps(triplets))

        loaded = load_qa_triplets(str(qa_file))

        assert len(loaded) == 2
        assert loaded[0]["question"] == "Q1?"
        assert loaded[1]["question"] == "Q2?"

    def test_load_qa_triplets_missing_file(self) -> None:
        """Test loading QA triplets from missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_qa_triplets("nonexistent.json")


class TestFindPassageRank:
    """Tests for finding passage rank in retrieved results."""

    def test_find_passage_rank_exact_match(self) -> None:
        """Test finding exact passage match at rank 0."""
        source_passage = "Divine Smite deals radiant damage."
        retrieved_passages = [
            {"text": "Divine Smite deals radiant damage.", "score": 0.95},
            {"text": "Sneak Attack deals extra damage.", "score": 0.80},
        ]

        rank = find_passage_rank(source_passage, retrieved_passages)

        assert rank == 0

    def test_find_passage_rank_second_position(self) -> None:
        """Test finding passage at rank 1."""
        source_passage = "Sneak Attack deals extra damage."
        retrieved_passages = [
            {"text": "Divine Smite deals radiant damage.", "score": 0.95},
            {"text": "Sneak Attack deals extra damage.", "score": 0.80},
        ]

        rank = find_passage_rank(source_passage, retrieved_passages)

        assert rank == 1

    def test_find_passage_rank_not_found(self) -> None:
        """Test when passage is not in retrieved results."""
        source_passage = "Action Surge allows extra action."
        retrieved_passages = [
            {"text": "Divine Smite deals radiant damage.", "score": 0.95},
            {"text": "Sneak Attack deals extra damage.", "score": 0.80},
        ]

        rank = find_passage_rank(source_passage, retrieved_passages)

        assert rank is None

    def test_find_passage_rank_empty_retrieved(self) -> None:
        """Test with empty retrieved passages list."""
        source_passage = "Some text"
        retrieved_passages: List[Dict[str, Any]] = []

        rank = find_passage_rank(source_passage, retrieved_passages)

        assert rank is None

    def test_find_passage_rank_case_sensitive(self) -> None:
        """Test that matching is case-sensitive."""
        source_passage = "Divine Smite"
        retrieved_passages = [{"text": "divine smite", "score": 0.95}]

        rank = find_passage_rank(source_passage, retrieved_passages)

        assert rank is None


class TestRAGPipelineRunner:
    """Tests for RAG pipeline runner."""

    @patch("dnd_dm_copilot.evaluation.run_rag_pipeline.FAISSRetriever")
    @patch("dnd_dm_copilot.evaluation.run_rag_pipeline.LFM2Client")
    def test_init_with_paths(
        self, mock_llm_class: MagicMock, mock_retriever_class: MagicMock
    ) -> None:
        """Test initialization with model and index paths."""
        runner = RAGPipelineRunner(
            model_path="models/sbert/",
            index_path="data/indices/mechanics/",
            llm_model_path="models/lfm2.gguf",
        )

        mock_retriever_class.assert_called_once_with(model_path="models/sbert/")
        mock_llm_class.assert_called_once_with(model_path="models/lfm2.gguf")
        assert runner.top_k == 10

    @patch("dnd_dm_copilot.evaluation.run_rag_pipeline.FAISSRetriever")
    @patch("dnd_dm_copilot.evaluation.run_rag_pipeline.LFM2Client")
    def test_run_single_query(
        self, mock_llm_class: MagicMock, mock_retriever_class: MagicMock
    ) -> None:
        """Test running RAG pipeline on single query."""
        # Setup mocks
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "Passage 1", "score": 0.95},
            {"text": "Passage 2", "score": 0.80},
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Generated answer"
        mock_llm_class.return_value = mock_llm

        runner = RAGPipelineRunner(
            model_path="models/sbert/",
            index_path="data/indices/mechanics/",
            llm_model_path="models/lfm2.gguf",
        )

        qa_triplet = {
            "question": "How does Divine Smite work?",
            "answer": "Divine Smite deals radiant damage.",
            "passage": "Passage 1",
            "metadata": {"source": "phb.pdf"},
        }

        result = runner.run_single_query(qa_triplet)

        assert result["question"] == "How does Divine Smite work?"
        assert result["ground_truth_answer"] == "Divine Smite deals radiant damage."
        assert result["generated_answer"] == "Generated answer"
        assert result["retrieved_passages"] == [
            {"text": "Passage 1", "score": 0.95},
            {"text": "Passage 2", "score": 0.80},
        ]
        assert result["source_passage_rank"] == 0
        mock_retriever.search.assert_called_once_with(
            query="How does Divine Smite work?", top_k=10
        )

    @patch("dnd_dm_copilot.evaluation.run_rag_pipeline.FAISSRetriever")
    @patch("dnd_dm_copilot.evaluation.run_rag_pipeline.LFM2Client")
    def test_run_batch(
        self, mock_llm_class: MagicMock, mock_retriever_class: MagicMock
    ) -> None:
        """Test running RAG pipeline on batch of queries."""
        # Setup mocks
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [{"text": "P1", "score": 0.95}],
            [{"text": "P2", "score": 0.90}],
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = ["Answer 1", "Answer 2"]
        mock_llm_class.return_value = mock_llm

        runner = RAGPipelineRunner(
            model_path="models/sbert/",
            index_path="data/indices/mechanics/",
            llm_model_path="models/lfm2.gguf",
        )

        qa_triplets = [
            {"question": "Q1?", "answer": "A1", "passage": "P1", "metadata": {}},
            {"question": "Q2?", "answer": "A2", "passage": "P2", "metadata": {}},
        ]

        results = runner.run_batch(qa_triplets)

        assert len(results) == 2
        assert results[0]["question"] == "Q1?"
        assert results[1]["question"] == "Q2?"

    @patch("dnd_dm_copilot.evaluation.run_rag_pipeline.FAISSRetriever")
    @patch("dnd_dm_copilot.evaluation.run_rag_pipeline.LFM2Client")
    def test_run_batch_with_errors(
        self, mock_llm_class: MagicMock, mock_retriever_class: MagicMock
    ) -> None:
        """Test that batch processing continues on errors."""
        # Setup mocks
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [{"text": "P1", "score": 0.95}],
            Exception("Retrieval error"),
            [{"text": "P3", "score": 0.90}],
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = ["Answer 1", "Answer 3"]
        mock_llm_class.return_value = mock_llm

        runner = RAGPipelineRunner(
            model_path="models/sbert/",
            index_path="data/indices/mechanics/",
            llm_model_path="models/lfm2.gguf",
        )

        qa_triplets = [
            {"question": "Q1?", "answer": "A1", "passage": "P1", "metadata": {}},
            {"question": "Q2?", "answer": "A2", "passage": "P2", "metadata": {}},
            {"question": "Q3?", "answer": "A3", "passage": "P3", "metadata": {}},
        ]

        results = runner.run_batch(qa_triplets, skip_errors=True)

        assert len(results) == 2
        assert results[0]["question"] == "Q1?"
        assert results[1]["question"] == "Q3?"


class TestSaveResults:
    """Tests for saving pipeline results."""

    def test_save_results(self, tmp_path: Path) -> None:
        """Test saving results to JSON file."""
        output_file = tmp_path / "results.json"
        results = [
            {
                "question": "Q1?",
                "generated_answer": "A1",
                "retrieved_passages": [],
                "source_passage_rank": 0,
            },
        ]

        save_results(results, str(output_file))

        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert len(loaded) == 1
        assert loaded[0]["question"] == "Q1?"

    def test_save_results_creates_directory(self, tmp_path: Path) -> None:
        """Test that saving creates parent directory if needed."""
        output_file = tmp_path / "subdir" / "results.json"
        results = [{"question": "Q?"}]

        save_results(results, str(output_file))

        assert output_file.exists()
        assert output_file.parent.exists()
