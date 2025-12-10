"""
LLM-based judge for evaluating generated answers.

Uses DeepSeek to judge whether generated answers are correct
compared to ground truth answers.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm as atqdm  # type: ignore

from dnd_dm_copilot.model.llm_client import DeepSeekClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_rag_results(results_file: str) -> List[Dict[str, Any]]:
    """
    Load RAG pipeline results from JSON file.

    Args:
        results_file: Path to RAG results JSON file

    Returns:
        List of result dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"RAG results file not found: {results_file}")

    with open(results_path, "r", encoding="utf-8") as f:
        results: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(results)} results from {results_file}")
    return results


def save_judgments(judgments: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save judgment results to JSON file.

    Args:
        judgments: List of judgment dictionaries
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judgments, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(judgments)} judgments to {output_file}")


class AnswerJudge:
    """LLM-based judge for evaluating generated answers."""

    PROMPT_TEMPLATE = """You are an expert D&D rules evaluator. Your task is to judge whether a generated answer is correct compared to the ground truth answer.  # noqa: E501

Question: {question}

Ground Truth Answer: {ground_truth_answer}

Generated Answer: {generated_answer}

Source Passage: {source_passage}

Evaluate whether the generated answer is correct. Consider:
- Does it convey the same core information as the ground truth?
- Is it factually accurate based on the source passage?
- Minor wording differences are acceptable if the meaning is preserved.

Respond in this exact format:
CORRECT or INCORRECT
Confidence: <0.0 to 1.0>
Reasoning: <brief explanation>"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_concurrent: int = 50,
    ) -> None:
        """
        Initialize answer judge.

        Args:
            api_key: DeepSeek API key (uses DEEPSEEK_API_KEY env var if None)
            model: Model name to use for judging
            max_concurrent: Maximum concurrent API calls
        """
        self.client = DeepSeekClient(api_key=api_key)
        self.async_client = self.client.get_async_client()
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def judge_answer(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Judge a single answer.

        Args:
            result: Dict with 'question', 'ground_truth_answer',
                   'generated_answer', and 'source_passage' fields

        Returns:
            Dict with:
                - question: Original question
                - ground_truth_answer: Expected answer
                - generated_answer: LLM-generated answer
                - correct: Boolean judgment
                - confidence: Float 0.0-1.0
                - reasoning: Explanation string

        Raises:
            ValueError: If LLM response cannot be parsed
            Exception: If API call fails
        """
        prompt = self.PROMPT_TEMPLATE.format(
            question=result["question"],
            ground_truth_answer=result["ground_truth_answer"],
            generated_answer=result["generated_answer"],
            source_passage=result["source_passage"],
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat_completion(
                model=self.model, messages=messages, temperature=0.0, max_tokens=256
            )

            # Parse response
            correct, confidence, reasoning = self._parse_judgment(response)

            return {
                "question": result["question"],
                "ground_truth_answer": result["ground_truth_answer"],
                "generated_answer": result["generated_answer"],
                "correct": correct,
                "confidence": confidence,
                "reasoning": reasoning,
            }

        except Exception as e:
            logger.error(
                "Failed to judge answer for question '%s': %s",
                result["question"],
                e,
            )
            raise

    def _parse_judgment(self, response: str) -> tuple[bool, float, str]:
        """
        Parse LLM judgment response.

        Args:
            response: Raw LLM response text

        Returns:
            Tuple of (correct, confidence, reasoning)

        Raises:
            ValueError: If response format is invalid
        """
        lines = response.strip().split("\n")

        verdict = None
        confidence = None
        reasoning = None

        for line in lines:
            line = line.strip()
            if line in ["CORRECT", "INCORRECT"]:
                verdict = line == "CORRECT"
            elif line.startswith("Confidence:"):
                conf_str = line[len("Confidence:") :].strip()
                try:
                    confidence = float(conf_str)
                    if not (0.0 <= confidence <= 1.0):
                        raise ValueError(f"Confidence out of range: {confidence}")
                except ValueError as e:
                    raise ValueError(f"Invalid confidence value: {conf_str}") from e
            elif line.startswith("Reasoning:"):
                reasoning = line[len("Reasoning:") :].strip()

        if verdict is None or confidence is None or reasoning is None:
            raise ValueError(
                f"Failed to parse judgment from response. "
                f"Got verdict={verdict}, confidence={confidence}, reasoning={reasoning}"
            )

        return verdict, confidence, reasoning

    async def judge_answer_async(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Judge a single answer (async version).

        Args:
            result: Dict with 'question', 'ground_truth_answer',
                   'generated_answer', and 'source_passage' fields

        Returns:
            Dict with:
                - question: Original question
                - ground_truth_answer: Expected answer
                - generated_answer: LLM-generated answer
                - correct: Boolean judgment
                - confidence: Float 0.0-1.0
                - reasoning: Explanation string

        Raises:
            ValueError: If LLM response cannot be parsed
            Exception: If API call fails
        """
        prompt = self.PROMPT_TEMPLATE.format(
            question=result["question"],
            ground_truth_answer=result["ground_truth_answer"],
            generated_answer=result["generated_answer"],
            source_passage=result["source_passage"],
        )
        messages = [{"role": "user", "content": prompt}]

        async with self.semaphore:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    temperature=0.0,
                    max_tokens=256,
                )

                content = response.choices[0].message.content
                if not isinstance(content, str):
                    content = str(content)

                # Parse response
                correct, confidence, reasoning = self._parse_judgment(content)

                return {
                    "question": result["question"],
                    "ground_truth_answer": result["ground_truth_answer"],
                    "generated_answer": result["generated_answer"],
                    "correct": correct,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }

            except Exception as e:
                logger.error(
                    "Failed to judge answer for question '%s': %s",
                    result["question"],
                    e,
                )
                raise

    async def judge_batch_async(
        self, results: List[Dict[str, Any]], skip_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Judge a batch of answers (async with concurrency).

        Args:
            results: List of result dictionaries
            skip_errors: If True, continue on errors; if False, raise on first error

        Returns:
            List of judgment dictionaries
        """

        async def process_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                return await self.judge_answer_async(result)
            except Exception as e:
                logger.error(f"Error judging answer for '{result['question']}': {e}")
                if not skip_errors:
                    raise
                return None

        # Create tasks for all results
        tasks = [process_result(r) for r in results]

        # Run with progress bar
        judgments = []
        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Judging answers",
        ):
            result = await coro
            if result is not None:
                judgments.append(result)

        logger.info(f"Judged {len(judgments)} answers")
        return judgments

    def judge_batch(
        self, results: List[Dict[str, Any]], skip_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Judge a batch of answers (synchronous wrapper).

        Args:
            results: List of result dictionaries
            skip_errors: If True, continue on errors; if False, raise

        Returns:
            List of judgment dictionaries
        """
        # Run async version
        return asyncio.run(self.judge_batch_async(results, skip_errors))


def main() -> None:
    """Main entry point for answer judging."""
    parser = argparse.ArgumentParser(
        description="Judge generated answers using LLM evaluator"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to RAG results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/judgments.json",
        help="Path to output judgments JSON file",
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
        # Load RAG results
        logger.info(f"Loading RAG results from {args.results}")
        results = load_rag_results(args.results)

        # Initialize judge
        logger.info(
            f"Initializing judge with {args.max_concurrent} concurrent requests..."
        )
        judge = AnswerJudge(max_concurrent=args.max_concurrent)

        # Judge answers
        logger.info("Judging answers...")
        judgments = judge.judge_batch(results, skip_errors=args.skip_errors)

        # Save judgments
        save_judgments(judgments, args.output)
        logger.info(f"Successfully judged {len(judgments)} answers")

        # Print summary statistics
        correct_count = sum(1 for j in judgments if j["correct"])
        accuracy = correct_count / len(judgments) if judgments else 0.0
        avg_confidence = (
            sum(j["confidence"] for j in judgments) / len(judgments)
            if judgments
            else 0.0
        )

        logger.info("\nSummary:")
        logger.info(f"  Total judged: {len(judgments)}")
        logger.info(f"  Correct: {correct_count}")
        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Avg confidence: {avg_confidence:.3f}")

    except Exception as e:
        logger.error(f"Answer judging failed: {e}")
        raise


if __name__ == "__main__":
    main()
