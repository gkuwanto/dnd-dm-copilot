#!/usr/bin/env python3
"""Build FAISS index from corpus."""

import argparse
import sys
from pathlib import Path

from dnd_dm_copilot.api.services.retriever import FAISSRetriever
from dnd_dm_copilot.utils.file_io import load_json_pairs


def build_index(corpus_path: str, model_path: str, output_path: str) -> None:
    """
    Build FAISS index from corpus.

    Args:
        corpus_path: Path to corpus JSON file
        model_path: Path to sentence-transformer model
        output_path: Path to save FAISS index
    """
    print(f"Loading corpus from {corpus_path}...")
    data = load_json_pairs(corpus_path)
    print(f"Loaded {len(data)} items")

    # Convert to retriever format (needs 'text' field)
    print("Converting to retriever format...")
    passages = []
    for item in data:
        # Handle both {"text": "..."} and {"passage": "..."} formats
        text = item.get("text") or item.get("passage", "")
        if not text:
            print(f"Warning: Skipping item with no text or passage field")
            continue

        passages.append({"text": text, "metadata": item.get("metadata", {})})

    print(f"Prepared {len(passages)} passages")

    # Build index
    print(f"Loading model from {model_path}...")
    retriever = FAISSRetriever(model_path)

    print("Building FAISS index...")
    retriever.build_index(passages)

    # Save index
    print(f"Saving index to {output_path}...")
    retriever.save_index(output_path)

    print(f"✓ Index built successfully!")
    print(f"  - {len(passages)} passages indexed")
    print(f"  - Model: {model_path}")
    print(f"  - Index saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from corpus")
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to corpus JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/mechanics-retrieval",
        help="Path to sentence-transformer model (default: models/mechanics-retrieval)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/indices/mechanics/",
        help="Path to save FAISS index (default: data/indices/mechanics/)",
    )

    args = parser.parse_args()

    # Check corpus exists
    if not Path(args.corpus).exists():
        print(f"Error: Corpus file not found: {args.corpus}")
        return 1

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return 1

    try:
        build_index(args.corpus, args.model, args.output)
        return 0
    except Exception as e:
        print(f"Error building index: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
