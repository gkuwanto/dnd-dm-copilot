"""Tests for health check router."""

from fastapi.testclient import TestClient


def test_health_check(test_client: TestClient) -> None:
    """Test health check endpoint."""
    response = test_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health_check_has_service_info(test_client: TestClient) -> None:
    """Test that health check includes service information."""
    response = test_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "dnd-dm-copilot-api"
