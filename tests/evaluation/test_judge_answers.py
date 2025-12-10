"""Tests for LLM-based answer judging."""

import asyncio  # noqa: F401
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
        mock_client = MagicMock()
        mock_client.get_async_client.return_value = MagicMock()
        mock_client_class.return_value = mock_client

        judge = AnswerJudge(api_key="test_key")

        mock_client_class.assert_called_once_with(api_key="test_key")
        assert judge.model == "deepseek-chat"
        assert judge.semaphore._value == 50  # default max_concurrent

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_init_with_max_concurrent(self, mock_client_class: MagicMock) -> None:
        """Test initialization with custom max_concurrent."""
        mock_client = MagicMock()
        mock_client.get_async_client.return_value = MagicMock()
        mock_client_class.return_value = mock_client

        judge = AnswerJudge(api_key="test_key", max_concurrent=10)

        assert judge.semaphore._value == 10

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_judge_answer_correct(self, mock_client_class: MagicMock) -> None:
        """Test judging a correct answer."""
        mock_client = MagicMock()
        mock_client.get_async_client.return_value = MagicMock()
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
        mock_client.get_async_client.return_value = MagicMock()
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
        mock_client.get_async_client.return_value = MagicMock()
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
        mock_client.get_async_client.return_value = MagicMock()
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
        # Mock responses
        mock_responses = []
        for verdict, conf, reason in [
            ("CORRECT", 0.95, "Good"),
            ("INCORRECT", 0.85, "Bad"),
        ]:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = (
                f"{verdict}\nConfidence: {conf}\nReasoning: {reason}"
            )
            mock_responses.append(mock_response)

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            side_effect=mock_responses
        )

        mock_client = MagicMock()
        mock_client.get_async_client.return_value = mock_async_client
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
        # Mock responses with an error in the middle
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = (
            "CORRECT\nConfidence: 0.95\nReasoning: Good"
        )

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = (
            "INCORRECT\nConfidence: 0.85\nReasoning: Bad"
        )

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            side_effect=[mock_response1, Exception("API error"), mock_response2]
        )

        mock_client = MagicMock()
        mock_client.get_async_client.return_value = mock_async_client
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
        # Order is non-deterministic with async, so check that we got 2 out of 3
        questions = {j["question"] for j in judgments}
        assert len(questions) == 2
        assert questions.issubset({"Q1?", "Q2?", "Q3?"})

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_parse_judgment_correct(self, mock_client_class: MagicMock) -> None:
        """Test parsing judgment with CORRECT verdict."""
        mock_client = MagicMock()
        mock_client.get_async_client.return_value = MagicMock()
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        response = "CORRECT\nConfidence: 0.90\nReasoning: Matches ground truth."

        correct, confidence, reasoning = judge._parse_judgment(response)

        assert correct is True
        assert confidence == 0.90
        assert reasoning == "Matches ground truth."

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_parse_judgment_incorrect(self, mock_client_class: MagicMock) -> None:
        """Test parsing judgment with INCORRECT verdict."""
        mock_client = MagicMock()
        mock_client.get_async_client.return_value = MagicMock()
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        response = "INCORRECT\nConfidence: 0.80\nReasoning: Wrong answer."

        correct, confidence, reasoning = judge._parse_judgment(response)

        assert correct is False
        assert confidence == 0.80
        assert reasoning == "Wrong answer."

    @pytest.mark.asyncio
    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    async def test_judge_answer_async_correct(
        self, mock_client_class: MagicMock
    ) -> None:
        """Test async judging of a correct answer."""
        # Mock the async client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "CORRECT\nConfidence: 0.95\nReasoning: The answer is accurate."
        )

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_client = MagicMock()
        mock_client.get_async_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        result = {
            "question": "How does Divine Smite work?",
            "ground_truth_answer": "Divine Smite deals radiant damage.",
            "generated_answer": "Divine Smite deals radiant damage to enemies.",
            "source_passage": "Divine Smite is a Paladin feature...",
        }

        judgment = await judge.judge_answer_async(result)

        assert judgment["correct"] is True
        assert judgment["confidence"] == 0.95
        assert "reasoning" in judgment
        mock_async_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    async def test_judge_answer_async_incorrect(
        self, mock_client_class: MagicMock
    ) -> None:
        """Test async judging of an incorrect answer."""
        # Mock the async client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "INCORRECT\nConfidence: 0.85\nReasoning: The answer is wrong."
        )

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_client = MagicMock()
        mock_client.get_async_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        result = {
            "question": "How does Divine Smite work?",
            "ground_truth_answer": "Divine Smite deals radiant damage.",
            "generated_answer": "Divine Smite deals fire damage.",
            "source_passage": "Divine Smite is a Paladin feature...",
        }

        judgment = await judge.judge_answer_async(result)

        assert judgment["correct"] is False
        assert judgment["confidence"] == 0.85

    @pytest.mark.asyncio
    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    async def test_judge_batch_async(self, mock_client_class: MagicMock) -> None:
        """Test async batch judging."""
        # Mock the async client
        mock_responses = []
        for i, verdict in enumerate(["CORRECT", "INCORRECT"]):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = (
                f"{verdict}\nConfidence: 0.9{i}\nReasoning: Test {i}"
            )
            mock_responses.append(mock_response)

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            side_effect=mock_responses
        )

        mock_client = MagicMock()
        mock_client.get_async_client.return_value = mock_async_client
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

        judgments = await judge.judge_batch_async(results)

        assert len(judgments) == 2
        assert judgments[0]["correct"] is True
        assert judgments[1]["correct"] is False

    @pytest.mark.asyncio
    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    async def test_judge_batch_async_with_errors(
        self, mock_client_class: MagicMock
    ) -> None:
        """Test async batch judging continues on errors."""
        # Mock responses with an error in the middle
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = (
            "CORRECT\nConfidence: 0.95\nReasoning: Good"
        )

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = (
            "INCORRECT\nConfidence: 0.85\nReasoning: Bad"
        )

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            side_effect=[mock_response1, Exception("API error"), mock_response2]
        )

        mock_client = MagicMock()
        mock_client.get_async_client.return_value = mock_async_client
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

        judgments = await judge.judge_batch_async(results, skip_errors=True)

        assert len(judgments) == 2
        # Order is non-deterministic with async, so check that we got 2 out of 3
        questions = {j["question"] for j in judgments}
        assert len(questions) == 2
        assert questions.issubset({"Q1?", "Q2?", "Q3?"})

    @patch("dnd_dm_copilot.evaluation.judge_answers.DeepSeekClient")
    def test_judge_batch_calls_async(self, mock_client_class: MagicMock) -> None:
        """Test that judge_batch wrapper properly calls async version."""
        # Mock the async client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "CORRECT\nConfidence: 0.95\nReasoning: Good"
        )

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_client = MagicMock()
        mock_client.get_async_client.return_value = mock_async_client
        mock_client_class.return_value = mock_client

        judge = AnswerJudge()
        results = [
            {
                "question": "Q1?",
                "ground_truth_answer": "A1",
                "generated_answer": "GA1",
                "source_passage": "P1",
            }
        ]

        judgments = judge.judge_batch(results)

        assert len(judgments) == 1
        assert judgments[0]["correct"] is True


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
