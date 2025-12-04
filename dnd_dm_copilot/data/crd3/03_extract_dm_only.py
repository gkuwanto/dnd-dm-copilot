#!/usr/bin/env python3
"""
Extract only DM response pairs from filtered CRD3 data
These pairs are most relevant for training a DM copilot
"""

import json
import os


def extract_dm_pairs(input_file: str, output_file: str):
    """Extract only pairs where the passage is a DM response"""
    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)

    print(f"Loaded {len(all_pairs):,} training pairs")

    # Filter for DM response pairs only
    dm_pairs = [p for p in all_pairs if "DM responds:" in p["passage"]]

    print(
        f"\nExtracted {len(dm_pairs):,} DM response pairs ({len(dm_pairs) / len(all_pairs) * 100:.1f}%)"
    )

    # Calculate statistics
    avg_query = sum(len(p["query"].split()) for p in dm_pairs[:10000]) / max(
        min(10000, len(dm_pairs)), 1
    )
    avg_passage = sum(len(p["passage"].split()) for p in dm_pairs[:10000]) / max(
        min(10000, len(dm_pairs)), 1
    )

    print("\nDM Response Pair Statistics:")
    print(f"  Average query length: {avg_query:.1f} words")
    print(f"  Average passage length: {avg_passage:.1f} words")

    # Save DM-only pairs
    print(f"\nSaving DM-only pairs to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dm_pairs, f, indent=2, ensure_ascii=False)

    print(f"✓ DM-only dataset saved: {output_file}")

    # Show examples
    print("\nExample DM Response Pairs:")
    import random

    random.seed(42)
    samples = random.sample(dm_pairs, min(3, len(dm_pairs)))

    for i, sample in enumerate(samples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Query: {sample['query'][:200]}...")
        print(f"Passage: {sample['passage'][:200]}...")


def main():
    input_file = "crd3_training_pairs_filtered.json"
    output_file = "crd3_training_pairs_dm_only.json"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Please run 02_filter_data.py first.")
        return

    extract_dm_pairs(input_file, output_file)


if __name__ == "__main__":
    main()
