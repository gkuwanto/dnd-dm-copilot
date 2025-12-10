#!/usr/bin/env python3
"""Push fine-tuned model to Hugging Face Hub."""

import argparse
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer


def push_model_to_hub(
    model_path: str,
    repo_id: str,
    commit_message: str = "Upload fine-tuned D&D mechanics retrieval model",
    private: bool = False,
):
    """
    Push a sentence-transformer model to Hugging Face Hub.

    Args:
        model_path: Local path to the model directory
        repo_id: Hugging Face repo ID (e.g., "garrykuwanto/mechanics-retrieval")
        commit_message: Commit message for the upload
        private: Whether to make the repo private
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = SentenceTransformer(model_path)

    # Push to hub
    print(f"Pushing model to {repo_id}...")
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
    )

    print(f"✓ Model successfully pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Push fine-tuned model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/mechanics-retrieval",
        help="Path to model directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., garrykuwanto/mechanics-retrieval)",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload fine-tuned D&D mechanics retrieval model",
        help="Commit message",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private",
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model path '{args.model_path}' does not exist")
        return 1

    # Check for HF token
    if not os.getenv("HF_TOKEN"):
        print("Warning: HF_TOKEN not found in environment")
        print("Make sure you're logged in with: huggingface-cli login")

    # Push model
    try:
        push_model_to_hub(
            model_path=args.model_path,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            private=args.private,
        )
        return 0
    except Exception as e:
        print(f"Error pushing model: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
