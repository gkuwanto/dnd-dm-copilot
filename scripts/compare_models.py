"""
Compare baseline model vs fine-tuned model on retrieval evaluation.

This script:
1. Loads the generated QA triplets
2. Builds indices for both baseline and fine-tuned models
3. Runs retrieval evaluation on both
4. Compares metrics and shows improvement
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from dnd_dm_copilot.api.services.retriever import FAISSRetriever
from dnd_dm_copilot.evaluation.metrics import compute_retrieval_metrics
from dnd_dm_copilot.utils import load_json_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """Load corpus and normalize format."""
    data = load_json_pairs(corpus_path)
    passages = []
    for item in data:
        text = item.get("text") or item.get("passage", "")
        if not text:
            continue
        passages.append({"text": text, "metadata": item.get("metadata", {})})
    return passages


def load_qa_triplets(qa_path: str) -> List[Dict[str, Any]]:
    """Load QA triplets."""
    with open(qa_path, "r") as f:
        return json.load(f)


def evaluate_model(
    model_path: str,
    corpus: List[Dict[str, Any]],
    qa_triplets: List[Dict[str, Any]],
    k_values: List[int],
) -> Dict[str, Any]:
    """Evaluate a model on the QA triplets."""
    logger.info(f"Building index for model: {model_path}")
    retriever = FAISSRetriever(model_path)
    retriever.build_index(corpus)

    logger.info("Running retrieval evaluation...")
    retrieved_ranks = []

    for triplet in qa_triplets:
        query = triplet.get("query") or triplet.get("question", "")
        expected_passage = triplet["passage"]

        # Retrieve top-k passages
        results = retriever.search(query, top_k=max(k_values))

        # Find rank of expected passage
        rank = None
        for i, result in enumerate(results):
            if result["text"].strip() == expected_passage.strip():
                rank = i
                break

        retrieved_ranks.append(rank)

    # Compute metrics
    metrics = compute_retrieval_metrics(retrieved_ranks, k_values)
    return metrics


def print_comparison(
    baseline_metrics: Dict[str, Any], finetuned_metrics: Dict[str, Any]
) -> None:
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)
    print(
        f"\n{'Metric':<20} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15} {'% Change':<15}"
    )
    print("-" * 80)

    for key in baseline_metrics:
        if key == "retrieved_ranks":
            continue

        baseline_val = baseline_metrics[key]
        finetuned_val = finetuned_metrics[key]
        improvement = finetuned_val - baseline_val

        if baseline_val > 0:
            pct_change = (improvement / baseline_val) * 100
        else:
            pct_change = float("inf") if improvement > 0 else 0

        print(
            f"{key:<20} {baseline_val:<15.4f} {finetuned_val:<15.4f} "
            f"{improvement:+<15.4f} {pct_change:+<15.1f}%"
        )

    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned model")
    parser.add_argument(
        "--corpus", type=str, required=True, help="Path to corpus JSON file"
    )
    parser.add_argument(
        "--qa_triplets",
        type=str,
        default="data/evaluation/qa_triplets.json",
        help="Path to QA triplets JSON file",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Baseline model name or path",
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="models/mechanics-retrieval",
        help="Fine-tuned model path",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for accuracy@k metrics",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/model_comparison.json",
        help="Output path for comparison results",
    )

    args = parser.parse_args()

    # Load data
    logger.info("Loading corpus and QA triplets...")
    corpus = load_corpus(args.corpus)
    qa_triplets = load_qa_triplets(args.qa_triplets)

    logger.info(f"Corpus size: {len(corpus)} passages")
    logger.info(f"QA triplets: {len(qa_triplets)} questions")

    # Evaluate baseline model
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING BASELINE MODEL")
    logger.info("=" * 80)
    baseline_metrics = evaluate_model(
        args.baseline_model, corpus, qa_triplets, args.k_values
    )

    # Evaluate fine-tuned model
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING FINE-TUNED MODEL")
    logger.info("=" * 80)
    finetuned_metrics = evaluate_model(
        args.finetuned_model, corpus, qa_triplets, args.k_values
    )

    # Print comparison
    print_comparison(baseline_metrics, finetuned_metrics)

    # Save results
    results = {
        "baseline": {
            "model": args.baseline_model,
            "metrics": {
                k: v for k, v in baseline_metrics.items() if k != "retrieved_ranks"
            },
        },
        "finetuned": {
            "model": args.finetuned_model,
            "metrics": {
                k: v for k, v in finetuned_metrics.items() if k != "retrieved_ranks"
            },
        },
        "improvement": {
            k: finetuned_metrics[k] - baseline_metrics[k]
            for k in baseline_metrics
            if k != "retrieved_ranks"
        },
        "corpus_size": len(corpus),
        "n_questions": len(qa_triplets),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
