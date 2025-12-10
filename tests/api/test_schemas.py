"""Tests for API Pydantic schemas."""

import pytest
from dnd_dm_copilot.api.models.schemas import (
    LoadLoreRequest,
    Passage,
    QueryRequest,
    QueryResponse,
)
from pydantic import ValidationError


class TestQueryRequest:
    """Tests for QueryRequest schema."""

    def test_query_request_valid(self) -> None:
        """Test creating a valid query request."""
        request = QueryRequest(query="How does Divine Smite work?", top_k=3)

        assert request.query == "How does Divine Smite work?"
        assert request.top_k == 3

    def test_query_request_default_top_k(self) -> None:
        """Test that top_k has a default value."""
        request = QueryRequest(query="test query")

        assert request.query == "test query"
        assert request.top_k == 3  # Default value

    def test_query_request_missing_query_raises_error(self) -> None:
        """Test that missing query raises validation error."""
        with pytest.raises(ValidationError):
            QueryRequest()  # type: ignore

    def test_query_request_empty_query_raises_error(self) -> None:
        """Test that empty query raises validation error."""
        with pytest.raises(ValidationError):
            QueryRequest(query="", top_k=3)

    def test_query_request_invalid_top_k_raises_error(self) -> None:
        """Test that invalid top_k raises validation error."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)

        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=-1)


class TestPassage:
    """Tests for Passage schema."""

    def test_passage_valid(self) -> None:
        """Test creating a valid passage."""
        passage = Passage(
            text="Divine Smite deals radiant damage.",
            score=0.95,
            source="phb.pdf",
        )

        assert passage.text == "Divine Smite deals radiant damage."
        assert passage.score == 0.95
        assert passage.source == "phb.pdf"

    def test_passage_with_metadata(self) -> None:
        """Test passage with additional metadata."""
        passage = Passage(
            text="test",
            score=0.9,
            source="doc.pdf",
            metadata={"page": 42, "section": "Combat"},
        )

        assert passage.metadata["page"] == 42
        assert passage.metadata["section"] == "Combat"

    def test_passage_default_metadata(self) -> None:
        """Test that metadata defaults to empty dict."""
        passage = Passage(text="test", score=0.9, source="doc.pdf")

        assert passage.metadata == {}

    def test_passage_score_validation(self) -> None:
        """Test that score is validated."""
        # Score should be between 0 and 1 for cosine similarity
        passage = Passage(text="test", score=0.5, source="doc.pdf")
        assert passage.score == 0.5

        # Negative scores should be allowed (for some distance metrics)
        passage = Passage(text="test", score=-0.1, source="doc.pdf")
        assert passage.score == -0.1


class TestQueryResponse:
    """Tests for QueryResponse schema."""

    def test_query_response_valid(self) -> None:
        """Test creating a valid query response."""
        response = QueryResponse(
            query="How does Divine Smite work?",
            passages=[
                Passage(
                    text="Divine Smite deals radiant damage.",
                    score=0.95,
                    source="phb.pdf",
                )
            ],
            answer="Divine Smite allows paladins to deal extra radiant damage.",
            retrieval_model="fine-tuned-minilm",
        )

        assert response.query == "How does Divine Smite work?"
        assert len(response.passages) == 1
        assert response.passages[0].text == "Divine Smite deals radiant damage."
        assert (
            response.answer
            == "Divine Smite allows paladins to deal extra radiant damage."
        )
        assert response.retrieval_model == "fine-tuned-minilm"

    def test_query_response_default_retrieval_model(self) -> None:
        """Test that retrieval_model has a default value."""
        response = QueryResponse(query="test", passages=[], answer="answer")

        assert response.retrieval_model == "unknown"

    def test_query_response_empty_passages(self) -> None:
        """Test response with no passages."""
        response = QueryResponse(
            query="test",
            passages=[],
            answer="No context available.",
        )

        assert response.passages == []
        assert response.answer == "No context available."

    def test_query_response_to_dict(self) -> None:
        """Test converting response to dict."""
        response = QueryResponse(
            query="test",
            passages=[Passage(text="passage 1", score=0.9, source="doc.pdf")],
            answer="answer",
        )

        data = response.model_dump()

        assert data["query"] == "test"
        assert len(data["passages"]) == 1
        assert data["passages"][0]["text"] == "passage 1"
        assert data["answer"] == "answer"


class TestLoadLoreRequest:
    """Tests for LoadLoreRequest schema."""

    def test_load_lore_request_valid(self) -> None:
        """Test creating a valid load lore request."""
        request = LoadLoreRequest(note_files=["notes0.txt", "notes1.txt", "notes2.md"])

        assert len(request.note_files) == 3
        assert "notes0.txt" in request.note_files
        assert "notes2.md" in request.note_files

    def test_load_lore_request_empty_list_raises_error(self) -> None:
        """Test that empty file list raises validation error."""
        with pytest.raises(ValidationError):
            LoadLoreRequest(note_files=[])

    def test_load_lore_request_missing_files_raises_error(self) -> None:
        """Test that missing note_files raises validation error."""
        with pytest.raises(ValidationError):
            LoadLoreRequest()  # type: ignore

    def test_load_lore_request_single_file(self) -> None:
        """Test load lore request with single file."""
        request = LoadLoreRequest(note_files=["notes.txt"])

        assert len(request.note_files) == 1
        assert request.note_files[0] == "notes.txt"
