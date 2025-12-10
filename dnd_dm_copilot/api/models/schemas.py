"""Pydantic schemas for API request/response models."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request schema for querying RAG system."""

    query: str = Field(..., min_length=1, description="User query")
    top_k: int = Field(default=3, ge=1, description="Number of passages to retrieve")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate that query is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v


class Passage(BaseModel):
    """Schema for a retrieved passage."""

    text: str = Field(..., description="Passage text content")
    score: float = Field(..., description="Similarity score")
    source: str = Field(..., description="Source file name")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class QueryResponse(BaseModel):
    """Response schema for query results."""

    query: str = Field(..., description="Original query")
    passages: List[Passage] = Field(..., description="Retrieved passages")
    answer: str = Field(..., description="Generated answer")
    retrieval_model: str = Field(
        default="unknown", description="Name of retrieval model used"
    )


class LoadLoreRequest(BaseModel):
    """Request schema for loading lore notes."""

    note_files: List[str] = Field(
        ..., min_length=1, description="List of note file paths to load"
    )
