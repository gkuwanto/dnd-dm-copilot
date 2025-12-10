"""LFM2 client using llama-cpp-python for GPU inference."""

from typing import List, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore

from dnd_dm_copilot.utils import get_logger

logger = get_logger(__name__)


class LFM2Client:
    """
    Client for LFM2-1.2B-RAG model using llama-cpp-python.

    Uses GGUF format for efficient GPU inference with llama.cpp.
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        verbose: bool = False,
    ):
        """
        Initialize LFM2 client.

        Args:
            model_path: Path to GGUF model file
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context window size
            verbose: Enable verbose logging

        Raises:
            ValueError: If model file not found
            ImportError: If llama-cpp-python is not installed
        """
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: uv pip install llama-cpp-python"
            )

        logger.info(f"Loading LFM2 model from {model_path}")

        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )

        logger.info("LFM2 model loaded successfully")

    def _format_prompt(self, query: str, context: List[str]) -> str:
        """
        Format prompt with context and query for RAG.

        Args:
            query: User query
            context: List of context passages

        Returns:
            Formatted prompt string
        """
        if context:
            context_str = "\n\n".join(context)
            prompt = (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
        else:
            prompt = f"Question: {query}\n\nAnswer:"

        return prompt

    def generate(
        self,
        query: str,
        context: List[str],
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate answer to query given context passages.

        Args:
            query: User query
            context: List of context passages from retrieval
            temperature: Sampling temperature (0.0 for greedy)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences for generation

        Returns:
            Generated answer text
        """
        prompt = self._format_prompt(query, context)

        logger.debug(f"Generating answer for query: {query[:50]}...")

        # Generate with llama-cpp-python
        output = self.model(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            echo=False,
        )

        # Extract generated text
        # Note: output is a dict when stream=False (not an iterator)
        generated_text: str = str(output["choices"][0]["text"]).strip()  # type: ignore[index] # noqa: E501

        logger.debug(f"Generated answer: {generated_text[:100]}...")

        return generated_text
