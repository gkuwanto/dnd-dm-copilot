"""RAG generator service combining retrieval and LLM generation."""

from typing import Any, Dict

from dnd_dm_copilot.api.services.retriever import FAISSRetriever
from dnd_dm_copilot.model.lfm2_client import LFM2Client
from dnd_dm_copilot.utils import get_logger

logger = get_logger(__name__)


class RAGGenerator:
    """
    RAG generator that combines retrieval and generation.

    This service:
    1. Retrieves relevant passages using FAISS retriever
    2. Generates answers using LFM2 LLM with retrieved context
    """

    def __init__(self, retriever: FAISSRetriever, llm_client: LFM2Client):
        """
        Initialize RAG generator.

        Args:
            retriever: FAISS retriever for passage retrieval
            llm_client: LFM2 client for answer generation
        """
        self.retriever = retriever
        self.llm_client = llm_client
        logger.info("RAG generator initialized")

    def generate(
        self,
        query: str,
        top_k: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate answer to query using RAG pipeline.

        Args:
            query: User query
            top_k: Number of passages to retrieve
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with:
                - query: Original query
                - passages: List of retrieved passages with scores and metadata
                - answer: Generated answer text
        """
        logger.info(f"Generating answer for query: '{query}' (top_k={top_k})")

        # Step 1: Retrieve relevant passages
        passages = self.retriever.search(query=query, top_k=top_k)

        logger.debug(f"Retrieved {len(passages)} passages")

        # Step 2: Extract text for LLM context
        context_texts = [p["text"] for p in passages]

        # Step 3: Generate answer with LLM
        answer = self.llm_client.generate(
            query=query,
            context=context_texts,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(f"Generated answer: {answer[:100]}...")

        # Return structured result
        return {
            "query": query,
            "passages": passages,
            "answer": answer,
        }
