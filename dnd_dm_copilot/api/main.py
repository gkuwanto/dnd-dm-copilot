"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from dnd_dm_copilot.api.routers import health, mechanics
from dnd_dm_copilot.api.services.generator import RAGGenerator
from dnd_dm_copilot.api.services.retriever import FAISSRetriever
from dnd_dm_copilot.model.lfm2_client import LFM2Client
from dnd_dm_copilot.utils import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Initializes the mechanics RAG generator on startup.
    """
    logger.info("Starting up API server...")

    # Paths
    mechanics_model_path = "models/sbert/"
    mechanics_index_path = "data/indices/mechanics/"
    lfm2_model_path = "models/lfm2/LFM2-1.2B-RAG-Q4_0.gguf"

    # Check if paths exist
    if not Path(mechanics_model_path).exists():
        raise FileNotFoundError(f"Model not found: {mechanics_model_path}")
    if not Path(mechanics_index_path).exists():
        raise FileNotFoundError(f"Index not found: {mechanics_index_path}")
    if not Path(lfm2_model_path).exists():
        raise FileNotFoundError(f"LFM2 model not found: {lfm2_model_path}")

    # Initialize retriever
    logger.info("Initializing FAISS retriever...")
    retriever = FAISSRetriever(model_path=mechanics_model_path)
    retriever.load_index(mechanics_index_path)

    # Initialize LFM2 client
    logger.info("Initializing LFM2 client...")
    llm_client = LFM2Client(model_path=lfm2_model_path)

    # Initialize RAG generator and set global
    logger.info("Initializing RAG generator...")
    mechanics.mechanics_generator = RAGGenerator(
        retriever=retriever, llm_client=llm_client
    )

    logger.info("API server startup complete!")

    yield

    # Cleanup (if needed)
    logger.info("Shutting down API server...")


app = FastAPI(
    title="D&D DM Copilot API",
    description="RAG-based API for D&D Dungeon Master assistance",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for demo
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount(
        "/demo",
        StaticFiles(directory=str(static_path / "demo"), html=True),
        name="demo",
    )


@app.get("/")
async def root():
    """Redirect root to demo page."""
    return RedirectResponse(url="/demo/")


# Include routers
app.include_router(health.router)
app.include_router(mechanics.router)
