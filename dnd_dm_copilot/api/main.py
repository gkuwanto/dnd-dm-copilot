"""FastAPI application entry point."""

from fastapi import FastAPI

from dnd_dm_copilot.api.routers import health, mechanics

app = FastAPI(
    title="D&D DM Copilot API",
    description="RAG-based API for D&D Dungeon Master assistance",
    version="0.1.0",
)

# Include routers
app.include_router(health.router)
app.include_router(mechanics.router)
