"""Tests for mechanics router."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_mechanics_generator(test_client: TestClient):
    """Mock the mechanics RAG generator."""
    from dnd_dm_copilot.api.main import app
    from dnd_dm_copilot.api.routers.mechanics import get_mechanics_generator

    mock_gen = MagicMock()
    mock_gen.generate.return_value = {
        "query": "How does Divine Smite work?",
        "passages": [
            {
                "text": "Divine Smite deals radiant damage.",
                "score": 0.95,
                "source": "phb.pdf",
                "metadata": {},
            }
        ],
        "answer": "Divine Smite allows paladins to deal extra radiant damage.",
    }

    # Override the dependency
    app.dependency_overrides[get_mechanics_generator] = lambda: mock_gen

    yield mock_gen

    # Cleanup
    app.dependency_overrides = {}


def test_query_mechanics_success(
    test_client: TestClient, mock_mechanics_generator
) -> None:
    """Test successful mechanics query."""
    response = test_client.post(
        "/api/v1/mechanics/query",
        json={"query": "How does Divine Smite work?", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "How does Divine Smite work?"
    assert len(data["passages"]) == 1
    assert (
        data["answer"] == "Divine Smite allows paladins to deal extra radiant damage."
    )


def test_query_mechanics_with_default_top_k(
    test_client: TestClient, mock_mechanics_generator
) -> None:
    """Test mechanics query with default top_k."""
    response = test_client.post(
        "/api/v1/mechanics/query",
        json={"query": "What is AC?"},
    )

    assert response.status_code == 200
    # Verify generator was called with default top_k=3
    mock_mechanics_generator.generate.assert_called_once()
    call_kwargs = mock_mechanics_generator.generate.call_args[1]
    assert call_kwargs["top_k"] == 3


def test_query_mechanics_invalid_request(
    test_client: TestClient, mock_mechanics_generator
) -> None:
    """Test mechanics query with invalid request."""
    # Missing query
    response = test_client.post(
        "/api/v1/mechanics/query",
        json={"top_k": 3},
    )

    assert response.status_code == 422  # Validation error


def test_query_mechanics_empty_query(
    test_client: TestClient, mock_mechanics_generator
) -> None:
    """Test mechanics query with empty query string."""
    response = test_client.post(
        "/api/v1/mechanics/query",
        json={"query": "", "top_k": 3},
    )

    assert response.status_code == 422  # Validation error


def test_query_mechanics_invalid_top_k(
    test_client: TestClient, mock_mechanics_generator
) -> None:
    """Test mechanics query with invalid top_k."""
    response = test_client.post(
        "/api/v1/mechanics/query",
        json={"query": "test", "top_k": 0},
    )

    assert response.status_code == 422  # Validation error


def test_query_mechanics_calls_generator(
    test_client: TestClient, mock_mechanics_generator
) -> None:
    """Test that mechanics query calls the generator with correct parameters."""
    test_client.post(
        "/api/v1/mechanics/query",
        json={"query": "How does initiative work?", "top_k": 5},
    )

    mock_mechanics_generator.generate.assert_called_once_with(
        query="How does initiative work?", top_k=5
    )


def test_query_mechanics_returns_passages(
    test_client: TestClient, mock_mechanics_generator
) -> None:
    """Test that mechanics query returns passages with scores."""
    response = test_client.post(
        "/api/v1/mechanics/query",
        json={"query": "test", "top_k": 3},
    )

    data = response.json()
    assert "passages" in data
    assert isinstance(data["passages"], list)
    if len(data["passages"]) > 0:
        passage = data["passages"][0]
        assert "text" in passage
        assert "score" in passage
