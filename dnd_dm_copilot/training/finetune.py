#!/usr/bin/env python3
"""
Fine-tuning script for sentence transformer models on D&D mechanics retrieval tasks.

This script fine-tunes a sentence transformer model using MultipleNegativesRankingLoss
for information retrieval tasks. It includes proper data splitting, evaluation,
model saving functionality, and Weights & Biases logging for experiment tracking.

Usage:
    # Basic usage
    uv run -m dnd_dm_copilot.training.finetune
        --dataset dnd-mechanics-dataset.json
        --output_dir output/sbert_retrieval_model

    # With custom wandb project and run name
    uv run -m dnd_dm_copilot.training.finetune
        --dataset dnd-mechanics-dataset.json
        --output_dir output/sbert_retrieval_model \
        --wandb_project "my-project" --wandb_run_name "experiment-1"

    # Disable wandb logging
    uv run -m dnd_dm_copilot.training.finetune
        --dataset dnd-mechanics-dataset.json
        --output_dir output/sbert_retrieval_model \
        --disable_wandb
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import dotenv
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import wandb

dotenv.load_dotenv()


def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """
    Load dataset from JSON file.

    Args:
        dataset_path: Path to the JSON dataset file

    Returns:
        List of dictionaries containing 'query' and 'passage' keys

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset format is invalid
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of dictionaries")

    if len(data) == 0:
        raise ValueError("Dataset is empty")

    # Validate data format on all items or sample if too large
    sample_size = min(100, len(data))
    for i, item in enumerate(data[:sample_size]):
        if not isinstance(item, dict) or "query" not in item or "passage" not in item:
            raise ValueError(
                f"Invalid data format at index {i}."
                " Expected dict with 'query' and 'passage' keys"
            )

    print(f"Loaded {len(data)} query-passage pairs")
    return data


def split_dataset(
    data: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        data: List of query-passage pairs
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation (test gets the remainder)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: train vs (val + test)
    train_data, temp_data = train_test_split(
        data, test_size=(val_ratio + test_ratio), random_state=random_state
    )

    # Second split: val vs test
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=random_state,
    )

    print("Dataset split:")
    print(
        f"  Train: {len(train_data)} samples ({len(train_data) / len(data) * 100:.1f}%)"
    )
    print(
        f"  Validation: {len(val_data)} samples"
        f" ({len(val_data) / len(data) * 100:.1f}%)"
    )
    print(f"  Test: {len(test_data)} samples ({len(test_data) / len(data) * 100:.1f}%)")

    return train_data, val_data, test_data


def create_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """
    Create a sentence transformer model with pooling layer.

    Args:
        model_name: Name of the base transformer model

    Returns:
        Configured SentenceTransformer model
    """
    print(f"Creating model with base: {model_name}")

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    return model


def prepare_training_data(
    train_data: List[Dict[str, str]], batch_size: int = 64
) -> DataLoader:
    """
    Prepare training data in InputExample format for MultipleNegativesRankingLoss.

    Args:
        train_data: List of training query-passage pairs
        batch_size: Batch size for training

    Returns:
        DataLoader for training
    """
    print(f"Preparing training data with batch size: {batch_size}")

    train_examples = []
    for item in train_data:
        train_examples.append(InputExample(texts=[item["query"], item["passage"]]))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)  # type: ignore

    print(f"Created {len(train_examples)} training examples")
    return train_dataloader


def prepare_ir_evaluator(
    data_pairs: List[Dict[str, str]], name: str = "eval"
) -> InformationRetrievalEvaluator:
    """
    Prepare an InformationRetrievalEvaluator from query-passage pairs.

    Args:
        data_pairs: List of query-passage pairs
        name: Name for the evaluator

    Returns:
        Configured InformationRetrievalEvaluator
    """
    queries: Dict[str, str] = {}  # qid -> query
    corpus: Dict[str, str] = {}  # pid -> passage
    relevant_docs: Dict[str, set] = {}  # qid -> set(pid)

    # Use mappings to ensure unique IDs for unique texts
    query_to_qid = {}
    passage_to_pid = {}

    print(f"Preparing evaluator '{name}'...")

    for item in data_pairs:
        query = item["query"]
        passage = item["passage"]

        # Get or create qid
        if query not in query_to_qid:
            qid = f"q{len(queries)}"
            query_to_qid[query] = qid
            queries[qid] = query
            relevant_docs[qid] = set()
        else:
            qid = query_to_qid[query]

        # Get or create pid
        if passage not in passage_to_pid:
            pid = f"p{len(corpus)}"
            passage_to_pid[passage] = pid
            corpus[pid] = passage
        else:
            pid = passage_to_pid[passage]

        # Add positive mapping
        relevant_docs[qid].add(pid)

    print(f"Evaluator '{name}': {len(queries)} queries, {len(corpus)} passages")

    return InformationRetrievalEvaluator(
        queries,
        corpus,
        relevant_docs,
        name=name,
        mrr_at_k=[1, 5, 10, 20],  # Calculate MRR@1, MRR@5, MRR@10, MRR@20
    )


def train_model(
    model: SentenceTransformer,
    train_dataloader: DataLoader,
    evaluator: InformationRetrievalEvaluator,
    output_path: str,
    num_epochs: int = 10,
    evaluation_steps: int = 500,
    warmup_ratio: float = 0.1,
    use_wandb: bool = True,
) -> None:
    """
    Train the sentence transformer model.

    Args:
        model: The model to train
        train_dataloader: Training data loader
        evaluator: Validation evaluator
        output_path: Path to save the trained model
        num_epochs: Number of training epochs
        evaluation_steps: Steps between evaluations
        warmup_ratio: Ratio of training steps for warmup
        use_wandb: Whether to log metrics to wandb
    """
    print(f"Starting training for {num_epochs} epochs...")

    # Calculate warmup steps
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Evaluation every {evaluation_steps} steps")

    # Log training configuration to wandb
    if use_wandb:
        wandb.log(
            {
                "training/total_steps": total_steps,
                "training/warmup_steps": warmup_steps,
                "training/evaluation_steps": evaluation_steps,
                "training/batch_size": train_dataloader.batch_size,
                "training/train_examples": len(train_dataloader.dataset),  # type: ignore
            }
        )

    # Create loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Create custom callback for wandb logging
    class WandbCallback:
        def __init__(self, use_wandb: bool = True):
            self.use_wandb = use_wandb
            self.step = 0

        def __call__(self, score: object, epoch: int, steps: int) -> None:
            if self.use_wandb:
                # The score parameter is typically a float (MRR score)
                # We'll log it as the main validation metric
                wandb.log(
                    {
                        "validation/mrr_score": (
                            score if isinstance(score, (int, float)) else 0
                        ),
                        "training/epoch": epoch,
                        "training/step": steps,
                    }
                )

    wandb_callback = WandbCallback(use_wandb)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True,
        callback=wandb_callback,
    )

    print(f"Training completed. Best model saved to: {output_path}")


def evaluate_test_set(
    model_path: str, test_data: List[Dict[str, str]]
) -> Dict[str, float]:
    """
    Evaluate the trained model on the test set.

    Args:
        model_path: Path to the trained model
        test_data: Test dataset

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Loading best model from: {model_path}")
    model = SentenceTransformer(model_path)

    print("Creating test evaluator...")
    test_evaluator = prepare_ir_evaluator(test_data, name="test")

    print("Evaluating on test set...")
    test_results = test_evaluator(model, output_path=".")

    print("Test Set Evaluation Results:")
    print(test_results)

    return test_results


def setup_wandb_logging(args: argparse.Namespace) -> Optional[object]:
    """
    Initialize Weights & Biases logging.

    Args:
        args: Parsed command line arguments

    Returns:
        wandb.Run object or None if disabled
    """
    if args.disable_wandb:
        return None

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "evaluation_steps": args.evaluation_steps,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "random_state": args.random_state,
            "dataset": args.dataset,
            "output_dir": args.output_dir,
        },
    )
    return wandb.run


def setup_training_environment() -> None:
    """Set up the training environment (CUDA checks, etc)."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU (training will be slower)")
    except ImportError:
        pass


def prepare_train_val_test_splits(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Prepare train, validation, and test data splits.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    data = load_dataset(args.dataset)

    if args.test_dataset:
        print(f"\nUsing separate test dataset: {args.test_dataset}")
        test_data = load_dataset(args.test_dataset)

        # Split main dataset into train and val only (90/10 split of training data)
        train_data, val_data = train_test_split(
            data, test_size=0.1, random_state=args.random_state
        )

        print("Dataset split (with separate test set):")
        print(
            f"  Train: {len(train_data)} samples"
            f"({len(train_data) / len(data) * 100:.1f}%)"
        )
        print(
            f"  Validation: {len(val_data)} samples"
            f" ({len(val_data) / len(data) * 100:.1f}%)"
        )
        print(f"  Test: {len(test_data)} samples (separate dataset)")

    else:
        # Original behavior: split from single dataset
        print("\nUsing single dataset with train/val/test split")
        train_data, val_data, test_data = split_dataset(
            data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_state=args.random_state,
        )

    return train_data, val_data, test_data


def log_dataset_info_to_wandb(
    args: argparse.Namespace,
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]],
    test_data: List[Dict[str, str]],
    baseline_results: Optional[Dict[str, float]] = None,
) -> None:
    """
    Log dataset and experiment information to Weights & Biases.

    Args:
        args: Parsed command line arguments
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        baseline_results: Optional baseline evaluation results
    """
    if args.disable_wandb:
        return

    wandb_log: Dict[str, Any] = {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
    }

    if args.test_dataset:
        wandb_log["test_dataset_path"] = args.test_dataset
    else:
        total = len(train_data) + len(val_data) + len(test_data)
        wandb_log["train_ratio"] = float(len(train_data) / total)  # type: ignore
        wandb_log["val_ratio"] = float(len(val_data) / total)  # type: ignore
        wandb_log["test_ratio"] = float(len(test_data) / total)  # type: ignore

    if baseline_results:
        if isinstance(baseline_results, dict):
            wandb.log({"baseline": baseline_results})
        else:
            wandb.log({"baseline/mrr_score": baseline_results})

    wandb.log(wandb_log)


def main() -> None:
    """Main function to run fine-tuning pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine-tune sentence transformer for D&D mechanics retrieval"
    )

    # Required arguments
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to JSON dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the trained model",
    )

    # Optional arguments
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=None,
        help="Path to separate test dataset JSON file (optional)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model name for fine-tuning",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--evaluation_steps", type=int, default=50, help="Steps between evaluations"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (only used if no test_dataset provided)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of data for validation (only used if no test_dataset provided)",
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dnd-dm-copilot",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--disable_wandb", action="store_true", help="Disable Weights & Biases logging"
    )

    args = parser.parse_args()

    # Setup environment
    setup_training_environment()

    # Initialize wandb
    setup_wandb_logging(args)

    try:
        # Prepare data splits
        train_data, val_data, test_data = prepare_train_val_test_splits(args)

        # Log dataset information
        log_dataset_info_to_wandb(args, train_data, val_data, test_data)

        # Evaluate baseline model on test set
        print("\nEvaluating baseline model on test set...")
        baseline_results = evaluate_test_set(args.model_name, test_data)

        # Log baseline results to wandb
        if not args.disable_wandb:
            if isinstance(baseline_results, dict):
                wandb.log({"baseline": baseline_results})
            else:
                wandb.log({"baseline/mrr_score": baseline_results})

        # Create model
        model = create_model(args.model_name)

        # Prepare training data
        train_dataloader = prepare_training_data(train_data, args.batch_size)

        # Prepare validation evaluator
        evaluator = prepare_ir_evaluator(val_data, name="val")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Train model
        train_model(
            model=model,
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            output_path=args.output_dir,
            num_epochs=args.num_epochs,
            evaluation_steps=args.evaluation_steps,
            use_wandb=not args.disable_wandb,
        )

        # Evaluate on test set
        test_results = evaluate_test_set(args.output_dir, test_data)

        # Log final test results to wandb
        if not args.disable_wandb:
            # Handle both float and dict results
            if isinstance(test_results, dict):
                wandb.log({"test": test_results})
                test_mrr = test_results.get(
                    "test_cosine_mrr@1", test_results.get("mrr_score", 0)
                )
            else:
                # test_results is a float (MRR score)
                wandb.log({"test/mrr_score": test_results})
                test_mrr = test_results

            # Handle baseline results similarly
            if isinstance(baseline_results, dict):
                baseline_mrr = baseline_results.get(
                    "test_cosine_mrr@1", baseline_results.get("mrr_score", 0)
                )
            else:
                baseline_mrr = baseline_results

            # Log model as artifact
            model_artifact = wandb.Artifact(
                name=f"sbert-retrieval-model-{wandb.run.id}",  # type: ignore
                type="model",
                description="Fine-tuned sentence transformer for D&D",
            )
            model_artifact.add_dir(args.output_dir)
            wandb.log_artifact(model_artifact)

            # Log performance comparison
            wandb.log(
                {
                    "comparison/baseline_mrr": baseline_mrr,
                    "comparison/finetuned_mrr": test_mrr,
                    "comparison/improvement": test_mrr - baseline_mrr,
                    "comparison/improvement_percent": (
                        ((test_mrr - baseline_mrr) / baseline_mrr * 100)
                        if baseline_mrr > 0
                        else 0
                    ),
                }
            )

        print("\n" + "=" * 60)
        print("FINE-TUNING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model saved to: {args.output_dir}")
        print(f"Baseline Test results: {baseline_results}")
        print(f"Fine-tuned Test results: {test_results}")

    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        if not args.disable_wandb:
            wandb.finish(exit_code=1)
        raise
    finally:
        # Finish wandb run
        if not args.disable_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
