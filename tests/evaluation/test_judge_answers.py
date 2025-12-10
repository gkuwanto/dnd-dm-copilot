"""Tests for LLM-based answer judging."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnd_dm_copilot.evaluation.judge_answers import (
    AnswerJudge,
    load_rag_results,
    save_judgments,
)


class TestLoadRAGResults:
    """Tests for loading RAG results."""

    def test_load_rag_results_valid_file(self, tmp_path: Path) -> None:
        """Test loading RAG results from valid JSON file."""
        results_file = tmp_path / "rag_results.json"
        results = [
            {
                "question": "Q1?",
                "ground_truth_answer": "A1",
                "generated_answer": "Generated A1",
                "source_passage": "P1",
            },
        ]
        results_file.write_text(json.dumps(results))

        loaded = load_rag_results(str(results_file))

        assert len(loaded) == 1
        assert loaded[0]["question"] == "Q1?"

    def test_load_rag_results_missing_file(self) -> None:
        """Test loading RAG results from missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_rag_results("nonexistent.json")


class TestAnswerJudge:
    """Tests for AnswerJudge class."""

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_init_with_api_key(self, mock_client_class: MagicMock) -> None:
        """Test initialization with API key."""
        judge = AnswerJudge(api_key="test_key")

        mock_client_class.assert_called_once_with(api_key="test_key")
        assert judge.model == "deepseek-chat"

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_judge_answer_correct(self, mock_client_class: MagicMock) -> None:
        """Test judging a correct answer."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = (
            "CORRECT\nConfidence: 0.95\nReasoning: The answer is accurate."
        )
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        result = {
            "question": "How does Divine Smite work?",
            "ground_truth_answer": "Divine Smite deals radiant damage.",
            "generated_answer": "Divine Smite deals radiant damage to enemies.",
            "source_passage": "Divine Smite is a Paladin feature...",
        }

        judgment = judge.judge_answer(result)

        assert judgment["correct"] is True
        assert judgment["confidence"] == 0.95
        assert "reasoning" in judgment
        mock_client.chat_completion.assert_called_once()

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_judge_answer_incorrect(self, mock_client_class: MagicMock) -> None:
        """Test judging an incorrect answer."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = (
            "INCORRECT\nConfidence: 0.85\nReasoning: The answer is wrong."
        )
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        result = {
            "question": "How does Divine Smite work?",
            "ground_truth_answer": "Divine Smite deals radiant damage.",
            "generated_answer": "Divine Smite deals fire damage.",
            "source_passage": "Divine Smite is a Paladin feature...",
        }

        judgment = judge.judge_answer(result)

        assert judgment["correct"] is False
        assert judgment["confidence"] == 0.85

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_judge_answer_malformed_response(
        self, mock_client_class: MagicMock
    ) -> None:
        """Test handling malformed LLM response."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "Invalid format"
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        result = {
            "question": "Q?",
            "ground_truth_answer": "A",
            "generated_answer": "GA",
            "source_passage": "P",
        }

        with pytest.raises(ValueError, match="Failed to parse"):
            judge.judge_answer(result)

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_judge_answer_invalid_confidence(
        self, mock_client_class: MagicMock
    ) -> None:
        """Test handling invalid confidence value."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = (
            "CORRECT\nConfidence: invalid\nReasoning: Test"
        )
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        result = {
            "question": "Q?",
            "ground_truth_answer": "A",
            "generated_answer": "GA",
            "source_passage": "P",
        }

        with pytest.raises(ValueError, match="Invalid confidence"):
            judge.judge_answer(result)

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_judge_batch(self, mock_client_class: MagicMock) -> None:
        """Test judging batch of results."""
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = [
            "CORRECT\nConfidence: 0.95\nReasoning: Good",
            "INCORRECT\nConfidence: 0.85\nReasoning: Bad",
        ]
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        results = [
            {
                "question": "Q1?",
                "ground_truth_answer": "A1",
                "generated_answer": "GA1",
                "source_passage": "P1",
            },
            {
                "question": "Q2?",
                "ground_truth_answer": "A2",
                "generated_answer": "GA2",
                "source_passage": "P2",
            },
        ]

        judgments = judge.judge_batch(results)

        assert len(judgments) == 2
        assert judgments[0]["correct"] is True
        assert judgments[1]["correct"] is False

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_judge_batch_with_errors(self, mock_client_class: MagicMock) -> None:
        """Test that batch judging continues on errors."""
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = [
            "CORRECT\nConfidence: 0.95\nReasoning: Good",
            Exception("API error"),
            "INCORRECT\nConfidence: 0.85\nReasoning: Bad",
        ]
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        results = [
            {
                "question": "Q1?",
                "ground_truth_answer": "A1",
                "generated_answer": "GA1",
                "source_passage": "P1",
            },
            {
                "question": "Q2?",
                "ground_truth_answer": "A2",
                "generated_answer": "GA2",
                "source_passage": "P2",
            },
            {
                "question": "Q3?",
                "ground_truth_answer": "A3",
                "generated_answer": "GA3",
                "source_passage": "P3",
            },
        ]

        judgments = judge.judge_batch(results, skip_errors=True)

        assert len(judgments) == 2
        assert judgments[0]["question"] == "Q1?"
        assert judgments[1]["question"] == "Q3?"

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_parse_judgment_correct(self, mock_client_class: MagicMock) -> None:
        """Test parsing judgment with CORRECT verdict."""
        judge = AnswerJudge()
        response = "CORRECT\nConfidence: 0.90\nReasoning: Matches ground truth."

        correct, confidence, reasoning = judge._parse_judgment(response)

        assert correct is True
        assert confidence == 0.90
        assert reasoning == "Matches ground truth."

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_parse_judgment_incorrect(self, mock_client_class: MagicMock) -> None:
        """Test parsing judgment with INCORRECT verdict."""
        judge = AnswerJudge()
        response = "INCORRECT\nConfidence: 0.80\nReasoning: Wrong answer."

        correct, confidence, reasoning = judge._parse_judgment(response)

        assert correct is False
        assert confidence == 0.80
        assert reasoning == "Wrong answer."


class TestSaveJudgments:
    """Tests for saving judgment results."""

    def test_save_judgments(self, tmp_path: Path) -> None:
        """Test saving judgments to JSON file."""
        output_file = tmp_path / "judgments.json"
        judgments = [
            {
                "question": "Q1?",
                "correct": True,
                "confidence": 0.95,
                "reasoning": "Good",
            },
        ]

        save_judgments(judgments, str(output_file))

        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert len(loaded) == 1
        assert loaded[0]["correct"] is True

    def test_save_judgments_creates_directory(self, tmp_path: Path) -> None:
        """Test that saving creates parent directory if needed."""
        output_file = tmp_path / "subdir" / "judgments.json"
        judgments = [{"question": "Q?", "correct": True}]

        save_judgments(judgments, str(output_file))

        assert output_file.exists()
        assert output_file.parent.exists()
