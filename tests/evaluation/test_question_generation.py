"""Tests for question generation from D&D passages."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnd_dm_copilot.evaluation.generate_questions import (
    QuestionGenerator,
    load_passages,
    sample_passages,
    save_qa_triplets,
)


class TestLoadPassages:
    """Tests for loading passages from JSON corpus."""

    def test_load_passages_valid_file(self, tmp_path: Path) -> None:
        """Test loading passages from valid JSON file."""
        corpus_file = tmp_path / "corpus.json"
        passages = [
            {"text": "Divine Smite allows...", "metadata": {"source": "phb.pdf"}},
            {"text": "Sneak Attack deals...", "metadata": {"source": "phb.pdf"}},
        ]
        corpus_file.write_text(json.dumps(passages))

        loaded = load_passages(str(corpus_file))

        assert len(loaded) == 2
        assert loaded[0]["text"] == "Divine Smite allows..."
        assert loaded[1]["text"] == "Sneak Attack deals..."

    def test_load_passages_missing_file(self) -> None:
        """Test loading passages from missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_passages("nonexistent.json")

    def test_load_passages_invalid_json(self, tmp_path: Path) -> None:
        """Test loading passages from invalid JSON raises error."""
        corpus_file = tmp_path / "invalid.json"
        corpus_file.write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_passages(str(corpus_file))

    def test_load_passages_empty_file(self, tmp_path: Path) -> None:
        """Test loading passages from empty file returns empty list."""
        corpus_file = tmp_path / "empty.json"
        corpus_file.write_text("[]")

        loaded = load_passages(str(corpus_file))

        assert loaded == []


class TestSamplePassages:
    """Tests for sampling random passages from corpus."""

    def test_sample_passages_valid_count(self) -> None:
        """Test sampling valid number of passages."""
        passages = [{"text": f"Passage {i}"} for i in range(100)]

        sampled = sample_passages(passages, n_samples=10, random_state=42)

        assert len(sampled) == 10
        assert all(p in passages for p in sampled)

    def test_sample_passages_more_than_available(self) -> None:
        """Test sampling more passages than available raises error."""
        passages = [{"text": f"Passage {i}"} for i in range(5)]

        with pytest.raises(ValueError, match="Cannot sample"):
            sample_passages(passages, n_samples=10)

    def test_sample_passages_reproducible(self) -> None:
        """Test that sampling with same random_state is reproducible."""
        passages = [{"text": f"Passage {i}"} for i in range(100)]

        sampled1 = sample_passages(passages, n_samples=10, random_state=42)
        sampled2 = sample_passages(passages, n_samples=10, random_state=42)

        assert sampled1 == sampled2

    def test_sample_passages_different_seeds(self) -> None:
        """Test that different random states produce different samples."""
        passages = [{"text": f"Passage {i}"} for i in range(100)]

        sampled1 = sample_passages(passages, n_samples=10, random_state=42)
        sampled2 = sample_passages(passages, n_samples=10, random_state=123)

        assert sampled1 != sampled2


class TestQuestionGenerator:
    """Tests for QuestionGenerator class."""

    @patch("dnd_dm_copilot.evaluation.generate_questions.DeepSeekClient")
    def test_init_with_api_key(self, mock_client_class: MagicMock) -> None:
        """Test initialization with API key."""
        generator = QuestionGenerator(api_key="test_key")

        mock_client_class.assert_called_once_with(api_key="test_key")
        assert generator.model == "deepseek-chat"

    @patch("dnd_dm_copilot.evaluation.generate_questions.DeepSeekClient")
    def test_init_without_api_key(self, mock_client_class: MagicMock) -> None:
        """Test initialization without API key uses environment variable."""
        generator = QuestionGenerator()

        mock_client_class.assert_called_once_with(api_key=None)

    @patch("dnd_dm_copilot.evaluation.generate_questions.DeepSeekClient")
    def test_generate_qa_from_passage(self, mock_client_class: MagicMock) -> None:
        """Test generating QA pair from passage."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = (
            "Answer: Divine Smite deals radiant damage.\n"
            "Question: How does Divine Smite work?"
        )
        mock_client_class.return_value = mock_client

        generator = QuestionGenerator()
        passage = {
            "text": "Divine Smite is a Paladin feature...",
            "metadata": {"source": "phb.pdf"},
        }

        qa_triplet = generator.generate_qa_from_passage(passage)

        assert qa_triplet["question"] == "How does Divine Smite work?"
        assert qa_triplet["answer"] == "Divine Smite deals radiant damage."
        assert qa_triplet["passage"] == passage["text"]
        assert qa_triplet["metadata"] == passage["metadata"]
        mock_client.chat_completion.assert_called_once()

    @patch("dnd_dm_copilot.evaluation.generate_questions.DeepSeekClient")
    def test_generate_qa_from_passage_alt_format(
        self, mock_client_class: MagicMock
    ) -> None:
        """Test generating QA pair from passage with 'passage' field."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = (
            "Answer: Divine Smite deals radiant damage.\n"
            "Question: How does Divine Smite work?"
        )
        mock_client_class.return_value = mock_client

        generator = QuestionGenerator()
        passage = {
            "passage": "Divine Smite is a Paladin feature...",
            "metadata": {"source": "phb.pdf"},
        }

        qa_triplet = generator.generate_qa_from_passage(passage)

        assert qa_triplet["question"] == "How does Divine Smite work?"
        assert qa_triplet["answer"] == "Divine Smite deals radiant damage."
        assert qa_triplet["passage"] == passage["passage"]
        assert qa_triplet["metadata"] == passage["metadata"]
        mock_client.chat_completion.assert_called_once()

    @patch("dnd_dm_copilot.evaluation.generate_questions.DeepSeekClient")
    def test_generate_qa_malformed_response(self, mock_client_class: MagicMock) -> None:
        """Test handling of malformed LLM response."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "This is not a valid format"
        mock_client_class.return_value = mock_client

        generator = QuestionGenerator()
        passage = {"text": "Some text", "metadata": {}}

        with pytest.raises(ValueError, match="Failed to parse"):
            generator.generate_qa_from_passage(passage)

    @patch("dnd_dm_copilot.evaluation.generate_questions.DeepSeekClient")
    @patch("dnd_dm_copilot.evaluation.generate_questions.asyncio.run")
    def test_generate_questions_batch(
        self, mock_asyncio_run: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Test generating questions for batch of passages."""
        # Mock the async run to return pre-made results
        mock_asyncio_run.return_value = [
            {
                "question": "Question 1?",
                "answer": "Answer 1",
                "passage": "Passage 1",
                "metadata": {},
            },
            {
                "question": "Question 2?",
                "answer": "Answer 2",
                "passage": "Passage 2",
                "metadata": {},
            },
        ]

        generator = QuestionGenerator()
        passages = [
            {"text": "Passage 1", "metadata": {}},
            {"text": "Passage 2", "metadata": {}},
        ]

        qa_triplets = generator.generate_questions_batch(passages)

        assert len(qa_triplets) == 2
        assert qa_triplets[0]["question"] == "Question 1?"
        assert qa_triplets[1]["question"] == "Question 2?"
        mock_asyncio_run.assert_called_once()

    @patch("dnd_dm_copilot.evaluation.generate_questions.DeepSeekClient")
    @patch("dnd_dm_copilot.evaluation.generate_questions.asyncio.run")
    def test_generate_questions_batch_with_errors(
        self, mock_asyncio_run: MagicMock, mock_client_class: MagicMock
    ) -> None:
        """Test that batch generation continues on errors."""
        # Mock skipping errors - returns only successful ones
        mock_asyncio_run.return_value = [
            {
                "question": "Question 1?",
                "answer": "Answer 1",
                "passage": "Passage 1",
                "metadata": {},
            },
            {
                "question": "Question 3?",
                "answer": "Answer 3",
                "passage": "Passage 3",
                "metadata": {},
            },
        ]

        generator = QuestionGenerator()
        passages = [
            {"text": "Passage 1", "metadata": {}},
            {"text": "Passage 2", "metadata": {}},
            {"text": "Passage 3", "metadata": {}},
        ]

        qa_triplets = generator.generate_questions_batch(passages, skip_errors=True)

        assert len(qa_triplets) == 2
        assert qa_triplets[0]["question"] == "Question 1?"
        assert qa_triplets[1]["question"] == "Question 3?"


class TestSaveQATriplets:
    """Tests for saving QA triplets to JSON."""

    def test_save_qa_triplets(self, tmp_path: Path) -> None:
        """Test saving QA triplets to file."""
        output_file = tmp_path / "qa_triplets.json"
        qa_triplets = [
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

        save_qa_triplets(qa_triplets, str(output_file))

        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert len(loaded) == 2
        assert loaded[0]["question"] == "Q1?"
        assert loaded[1]["question"] == "Q2?"

    def test_save_qa_triplets_creates_directory(self, tmp_path: Path) -> None:
        """Test that saving creates parent directory if needed."""
        output_file = tmp_path / "subdir" / "qa_triplets.json"
        qa_triplets = [
            {"question": "Q?", "answer": "A", "passage": "P", "metadata": {}}
        ]

        save_qa_triplets(qa_triplets, str(output_file))

        assert output_file.exists()
        assert output_file.parent.exists()
