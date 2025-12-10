"""Health check router."""

from typing import Dict

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Status information about the service
    """
    return {
        "status": "healthy",
        "service": "dnd-dm-copilot-api",
    }
