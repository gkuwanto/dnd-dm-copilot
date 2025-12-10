"""
Generate evaluation questions from D&D passages using DeepSeek.

This script samples passages from a corpus and generates grounded Q&A pairs
using an answer-first-then-question approach.
"""

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm as atqdm  # type: ignore

from dnd_dm_copilot.model.llm_client import DeepSeekClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_passages(corpus_file: str) -> List[Dict[str, Any]]:
    """
    Load passages from JSON corpus file.

    Args:
        corpus_file: Path to JSON file containing passages

    Returns:
        List of passage dictionaries

    Raises:
        FileNotFoundError: If corpus file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    corpus_path = Path(corpus_file)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        passages: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(passages)} passages from {corpus_file}")
    return passages


def sample_passages(
    passages: List[Dict[str, Any]], n_samples: int, random_state: int = 42
) -> List[Dict[str, Any]]:
    """
    Sample random passages from corpus.

    Args:
        passages: List of passage dictionaries
        n_samples: Number of passages to sample
        random_state: Random seed for reproducibility

    Returns:
        List of sampled passages

    Raises:
        ValueError: If n_samples > len(passages)
    """
    if n_samples > len(passages):
        raise ValueError(
            f"Cannot sample {n_samples} passages from corpus of size {len(passages)}"
        )

    random.seed(random_state)
    sampled = random.sample(passages, n_samples)
    logger.info(f"Sampled {len(sampled)} passages (random_state={random_state})")
    return sampled


def save_qa_triplets(qa_triplets: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save QA triplets to JSON file.

    Args:
        qa_triplets: List of QA triplet dictionaries
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_triplets, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(qa_triplets)} QA triplets to {output_file}")


class QuestionGenerator:
    """Generator for creating QA pairs from D&D passages using DeepSeek."""

    PROMPT_TEMPLATE = """You are an expert D&D rules instructor. Given a passage from a D&D rulebook, generate a factual question-answer pair.  # noqa: E501

First, write a clear, concise answer based ONLY on the information in the passage.
Then, write a natural question that would lead to this answer.

Format your response exactly as:
Answer: <your answer here>
Question: <your question here>

Passage:
{passage}

Generate a question-answer pair:"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_concurrent: int = 50,
    ) -> None:
        """
        Initialize question generator.

        Args:
            api_key: DeepSeek API key (uses DEEPSEEK_API_KEY env var if None)
            model: Model name to use for generation
            max_concurrent: Maximum concurrent API calls
        """
        self.client = DeepSeekClient(api_key=api_key)
        self.async_client = self.client.get_async_client()
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def generate_qa_from_passage(self, passage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a QA pair from a single passage.

        Args:
            passage: Dictionary with 'text' and optional 'metadata' fields

        Returns:
            Dictionary with 'question', 'answer', 'passage', and 'metadata' fields

        Raises:
            ValueError: If LLM response cannot be parsed
            Exception: If API call fails
        """
        # Handle both formats: {"text": "..."} and {"passage": "..."}
        passage_text = passage.get("text") or passage.get("passage", "")
        if not passage_text:
            raise ValueError("Passage must have 'text' or 'passage' field")

        prompt = self.PROMPT_TEMPLATE.format(passage=passage_text)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat_completion(
                model=self.model, messages=messages, temperature=0.7, max_tokens=512
            )

            # Parse response
            answer, question = self._parse_response(response)

            return {
                "question": question,
                "answer": answer,
                "passage": passage_text,
                "metadata": passage.get("metadata", {}),
            }

        except Exception as e:
            logger.error(f"Failed to generate QA for passage: {e}")
            raise

    def _parse_response(self, response: str) -> tuple[str, str]:
        """
        Parse LLM response to extract answer and question.

        Args:
            response: Raw LLM response text

        Returns:
            Tuple of (answer, question)

        Raises:
            ValueError: If response format is invalid
        """
        lines = response.strip().split("\n")

        answer = None
        question = None

        for line in lines:
            line = line.strip()
            if line.startswith("Answer:"):
                answer = line[len("Answer:") :].strip()
            elif line.startswith("Question:"):
                question = line[len("Question:") :].strip()

        if not answer or not question:
            raise ValueError(
                f"Failed to parse answer and question from response: {response}"
            )

        return answer, question

    async def generate_qa_from_passage_async(
        self, passage: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a QA pair from a single passage (async version).

        Args:
            passage: Dictionary with 'text' or 'passage' and optional 'metadata'

        Returns:
            Dictionary with 'question', 'answer', 'passage', and 'metadata'

        Raises:
            ValueError: If LLM response cannot be parsed
            Exception: If API call fails
        """
        # Handle both formats: {"text": "..."} and {"passage": "..."}
        passage_text = passage.get("text") or passage.get("passage", "")
        if not passage_text:
            raise ValueError("Passage must have 'text' or 'passage' field")

        prompt = self.PROMPT_TEMPLATE.format(passage=passage_text)
        messages = [{"role": "user", "content": prompt}]

        async with self.semaphore:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=512,
                )

                content = response.choices[0].message.content
                if not isinstance(content, str):
                    content = str(content)

                # Parse response
                answer, question = self._parse_response(content)

                return {
                    "question": question,
                    "answer": answer,
                    "passage": passage_text,
                    "metadata": passage.get("metadata", {}),
                }

            except Exception as e:
                logger.error(f"Failed to generate QA for passage: {e}")
                raise

    async def generate_questions_batch_async(
        self, passages: List[Dict[str, Any]], skip_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate QA pairs for a batch of passages (async with concurrency).

        Args:
            passages: List of passage dictionaries
            skip_errors: If True, continue on errors; if False, raise on first error

        Returns:
            List of QA triplet dictionaries
        """

        async def process_passage(passage: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                return await self.generate_qa_from_passage_async(passage)
            except Exception as e:
                logger.error(f"Error generating QA for passage: {e}")
                if not skip_errors:
                    raise
                return None

        # Create tasks for all passages
        tasks = [process_passage(p) for p in passages]

        # Run with progress bar
        qa_triplets = []
        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating questions",
        ):
            result = await coro
            if result is not None:
                qa_triplets.append(result)

        logger.info(f"Generated {len(qa_triplets)} QA triplets")
        return qa_triplets

    def generate_questions_batch(
        self, passages: List[Dict[str, Any]], skip_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate QA pairs for a batch of passages (synchronous wrapper).

        Args:
            passages: List of passage dictionaries
            skip_errors: If True, continue on errors; if False, raise on first error

        Returns:
            List of QA triplet dictionaries
        """
        # Run async version
        return asyncio.run(self.generate_questions_batch_async(passages, skip_errors))


def main() -> None:
    """Main entry point for question generation script."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation questions from D&D passages"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to corpus JSON file (e.g., data/processed/mechanics_corpus.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/qa_triplets.json",
        help="Path to output QA triplets JSON file",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of passages to sample for question generation",
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Continue processing on errors instead of stopping",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=50,
        help="Maximum concurrent API calls (default: 50)",
    )

    args = parser.parse_args()

    try:
        # Load corpus
        logger.info(f"Loading corpus from {args.corpus}")
        passages = load_passages(args.corpus)

        # Sample passages
        logger.info(f"Sampling {args.n_samples} passages")
        sampled_passages = sample_passages(
            passages, n_samples=args.n_samples, random_state=args.random_state
        )

        # Generate questions
        logger.info(
            f"Generating questions with {args.max_concurrent} concurrent requests..."
        )
        generator = QuestionGenerator(max_concurrent=args.max_concurrent)
        qa_triplets = generator.generate_questions_batch(
            sampled_passages, skip_errors=args.skip_errors
        )

        # Save results
        save_qa_triplets(qa_triplets, args.output)
        logger.info(f"Successfully generated {len(qa_triplets)} QA triplets")

    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
