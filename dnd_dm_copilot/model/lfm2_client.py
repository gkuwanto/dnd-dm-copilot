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
        n_ctx: int = 128000,
        verbose: bool = False,
        chat_format: str = "chatml",
    ):
        """
        Initialize LFM2 client.

        Args:
            model_path: Path to GGUF model file
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context window size
            verbose: Enable verbose logging
            chat_format: Chat format template (default: llama-2)

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
            chat_format=chat_format,
            flash_attn=True,
        )

        logger.info("LFM2 model loaded successfully")

    def _create_messages(self, query: str, context: List[str]) -> List[dict]:
        """
        Create chat messages with context and query for RAG.

        Args:
            query: User query
            context: List of context passages

        Returns:
            List of message dictionaries for chat completion
        """
        if context:
            context_str = "\n\n".join(context)
            system_message = {
                "role": "system",
                "content": (
                    "You are a helpful Dungeon Master assistant. "
                    "Use the following context to answer questions about D&D rules and mechanics."
                ),
            }
            user_message = {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {query}",
            }
            return [system_message, user_message]
        else:
            system_message = {
                "role": "system",
                "content": "You are a helpful Dungeon Master assistant.",
            }
            user_message = {"role": "user", "content": query}
            return [system_message, user_message]

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
        messages = self._create_messages(query, context)

        logger.debug(f"Generating answer for query: {query[:50]}...")

        # Generate with create_chat_completion
        output = self.model.create_chat_completion(
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        self.model.reset()

        # Extract generated text from chat completion
        generated_text: str = str(output["choices"][0]["message"]["content"]).strip()  # type: ignore[index] # noqa: E501

        logger.debug(f"Generated answer: {generated_text[:100]}...")

        return generated_text
