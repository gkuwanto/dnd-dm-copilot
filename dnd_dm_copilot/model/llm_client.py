"""LLM client for interacting with DeepSeek API."""

import logging
import os
import time
from typing import Any, Optional

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables on module import
load_dotenv()


class LLMClientError(Exception):
    """Custom exception for LLM client errors."""

    pass


class DeepSeekClient:
    """Client for interacting with DeepSeek API with retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        initial_wait: float = 1.0,
    ):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            base_url: Base URL for DeepSeek API
            max_retries: Maximum number of retry attempts
            initial_wait: Initial wait time between retries in seconds

        Raises:
            LLMClientError: If API key is not provided or found
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise LLMClientError("DEEPSEEK_API_KEY not provided.")

        self.base_url = base_url
        self.max_retries = max_retries
        self.initial_wait = initial_wait

        # Initialize clients
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url
        )

    def __enter__(self) -> "DeepSeekClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the clients and release resources."""
        try:
            self.client.close()
        except Exception as e:
            logger.warning(f"Error closing sync client: {e}")

        try:
            # For async client, we would need to use asyncio
            # self.async_client.close()
            pass
        except Exception as e:
            logger.warning(f"Error closing async client: {e}")

    def chat_completion(
        self,
        model: str,
        messages: list,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Make a chat completion request with retry logic.

        Args:
            model: Model name (e.g., "deepseek-chat")
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments to pass to API

        Returns:
            Response text from the model

        Raises:
            LLMClientError: If request fails after all retries
        """
        wait_time = self.initial_wait
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Chat completion attempt(attempt {attempt}/{self.max_retries})"
                )

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                content = response.choices[0].message.content
                if isinstance(content, str):
                    return content
                return str(content)

            except (openai.RateLimitError, openai.Timeout) as e:  # type: ignore
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff

            except openai.APIError as e:
                raise LLMClientError(f"API error: {e}")

        raise LLMClientError(
            f"Chat completion failed after {self.max_retries} attempts: {last_error}"
        )

    def get_default_client(self) -> openai.OpenAI:
        """
        Get the default synchronous client.

        Returns:
            OpenAI client configured for DeepSeek
        """
        return self.client

    def get_async_client(self) -> openai.AsyncOpenAI:
        """
        Get the asynchronous client.

        Returns:
            AsyncOpenAI client configured for DeepSeek
        """
        return self.async_client


# Factory function for backward compatibility
def create_client(
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
) -> DeepSeekClient:
    """
    Factory function to create a DeepSeekClient instance.

    Args:
        api_key: DeepSeek API key
        base_url: Base URL for DeepSeek API

    Returns:
        DeepSeekClient instance
    """
    return DeepSeekClient(api_key=api_key, base_url=base_url)


if __name__ == "__main__":
    try:
        client = create_client()
        print(f"Client: {client}")
        print(f"Sync client: {client.get_default_client()}")
        print(f"Async client: {client.get_async_client()}")
        client.close()
    except LLMClientError as e:
        print(f"Error: {e}")
