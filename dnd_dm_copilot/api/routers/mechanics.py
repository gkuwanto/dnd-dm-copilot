"""Mechanics RAG router."""

from typing import Optional

from fastapi import APIRouter, Depends

from dnd_dm_copilot.api.models.schemas import QueryRequest, QueryResponse
from dnd_dm_copilot.api.services.generator import RAGGenerator
from dnd_dm_copilot.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/mechanics", tags=["mechanics"])

# Global mechanics generator (will be initialized on startup)
mechanics_generator: Optional[RAGGenerator] = None


def get_mechanics_generator() -> RAGGenerator:
    """
    Dependency to get the mechanics RAG generator.

    Returns:
        RAGGenerator instance for mechanics queries

    Raises:
        RuntimeError: If generator not initialized
    """
    if mechanics_generator is None:
        raise RuntimeError("Mechanics generator not initialized")
    return mechanics_generator


@router.post("/query", response_model=QueryResponse)
async def query_mechanics(
    request: QueryRequest,
    generator: RAGGenerator = Depends(get_mechanics_generator),
) -> QueryResponse:
    """
    Query the mechanics RAG system.

    Args:
        request: Query request with query text and top_k
        generator: Mechanics RAG generator (injected dependency)

    Returns:
        Query response with passages and generated answer
    """
    logger.info(f"Mechanics query: '{request.query}' (top_k={request.top_k})")

    result = generator.generate(query=request.query, top_k=request.top_k)

    return QueryResponse(
        query=result["query"],
        passages=result["passages"],
        answer=result["answer"],
        retrieval_model="fine-tuned-minilm",
    )
