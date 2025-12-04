#!/usr/bin/env python3
"""
Filter CRD3 training pairs to improve data quality
Removes low-quality pairs (very short passages, dice rolls, etc.)
"""

import json
from typing import Dict, List


def is_likely_dice_roll(text: str) -> bool:
    """Check if text is likely just a dice roll or number"""
    # Remove common prefixes
    for prefix in [
        "MATT:",
        "TRAVIS:",
        "LAURA:",
        "LIAM:",
        "TALIESIN:",
        "SAM:",
        "MARISHA:",
        "ASHLEY:",
    ]:
        text = text.replace(prefix, "").strip()

    # Check if it's just a number or simple dice notation
    words = text.split()
    if len(words) == 1:
        # Single word that's a number
        if words[0].replace(".", "").replace(",", "").isdigit():
            return True
        # Simple dice notation like "d20" or "1d6"
        if "d" in words[0].lower() and any(c.isdigit() for c in words[0]):
            return True

    return False


def is_filler_response(text: str) -> bool:
    """Check if text is a filler response like 'Yeah', 'Okay', 'Mm-hmm'"""
    filler_words = {
        "yeah",
        "yes",
        "yep",
        "yup",
        "ok",
        "okay",
        "sure",
        "right",
        "mm-hmm",
        "uh-huh",
        "mhm",
        "nope",
        "no",
        "nah",
        "oh",
        "ah",
        "um",
        "uh",
        "hmm",
        "hm",
    }

    # Remove speaker names
    for prefix in [
        "MATT:",
        "TRAVIS:",
        "LAURA:",
        "LIAM:",
        "TALIESIN:",
        "SAM:",
        "MARISHA:",
        "ASHLEY:",
    ]:
        text = text.replace(prefix, "").strip()

    # Check if it's just filler
    cleaned = text.lower().strip(".,!?")
    return cleaned in filler_words


def filter_training_pairs(
    pairs: List[Dict],
    min_passage_words: int = 5,
    min_query_words: int = 5,
    remove_dice_rolls: bool = True,
    remove_fillers: bool = True,
    keep_dm_responses: bool = True,
) -> tuple[List[Dict], Dict[str, int]]:
    """Filter training pairs based on quality criteria"""
    filtered_pairs = []

    stats = {
        "total": len(pairs),
        "removed_short_passage": 0,
        "removed_short_query": 0,
        "removed_dice_roll": 0,
        "removed_filler": 0,
        "kept": 0,
    }

    for pair in pairs:
        query = pair["query"]
        passage = pair["passage"]

        # Always keep high-quality DM response pairs
        if keep_dm_responses and "DM responds:" in passage:
            passage_text = passage.replace("DM responds:", "").strip()
            if len(passage_text.split()) >= min_passage_words:
                filtered_pairs.append(pair)
                stats["kept"] += 1
                continue

        # Check query length
        query_words = len(query.split())
        if query_words < min_query_words:
            stats["removed_short_query"] += 1
            continue

        # Check passage length
        passage_words = len(passage.split())
        if passage_words < min_passage_words:
            stats["removed_short_passage"] += 1
            continue

        # Check for dice rolls
        if remove_dice_rolls and is_likely_dice_roll(passage):
            stats["removed_dice_roll"] += 1
            continue

        # Check for filler responses
        if remove_fillers and is_filler_response(passage):
            stats["removed_filler"] += 1
            continue

        # Pair passed all filters
        filtered_pairs.append(pair)
        stats["kept"] += 1

    return filtered_pairs, stats


def print_statistics(
    stats: Dict, filtered_pairs: List[Dict], original_pairs: List[Dict]
) -> None:
    """Print filtering statistics"""
    print("\n" + "=" * 60)
    print("FILTERING STATISTICS")
    print("=" * 60)
    print(f"Original pairs: {stats['total']:,}")
    print(f"Filtered pairs: {stats['kept']:,}")
    print(
        f"Removed: {stats['total'] - stats['kept']:,}"
        "({(stats['total'] - stats['kept']) / stats['total'] * 100:.1f}%)"
    )
    print()
    print("Removal reasons:")
    print(f"  Short passage (<5 words): {stats['removed_short_passage']:,}")
    print(f"  Short query (<5 words): {stats['removed_short_query']:,}")
    print(f"  Dice rolls: {stats['removed_dice_roll']:,}")
    print(f"  Filler responses: {stats['removed_filler']:,}")
    print()

    # Calculate average lengths
    avg_query_before = (
        sum(len(p["query"].split()) for p in original_pairs[:10000]) / 10000
    )
    avg_passage_before = (
        sum(len(p["passage"].split()) for p in original_pairs[:10000]) / 10000
    )
    avg_query_after = sum(
        len(p["query"].split()) for p in filtered_pairs[:10000]
    ) / max(min(10000, len(filtered_pairs)), 1)
    avg_passage_after = sum(
        len(p["passage"].split()) for p in filtered_pairs[:10000]
    ) / max(min(10000, len(filtered_pairs)), 1)

    print(f"Average query length: {avg_query_before:.1f} → {avg_query_after:.1f} words")
    print(
        f"Average passage length: {avg_passage_before:.1f} "
        f"→ {avg_passage_after:.1f} words"
    )
    print("=" * 60)


def main() -> None:
    input_file = "crd3_training_pairs_no_llm.json"
    output_file = "crd3_training_pairs_filtered.json"

    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        training_pairs = json.load(f)

    print(f"Loaded {len(training_pairs):,} training pairs")
    print("\nFiltering pairs...")

    # Filter with default settings
    filtered_pairs, stats = filter_training_pairs(
        training_pairs,
        min_passage_words=5,
        min_query_words=5,
        remove_dice_rolls=True,
        remove_fillers=True,
        keep_dm_responses=True,
    )

    # Print statistics
    print_statistics(stats, filtered_pairs, training_pairs)

    # Save filtered data
    print(f"\nSaving filtered data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_pairs, f, indent=2, ensure_ascii=False)

    print(f"✓ Filtered dataset saved: {output_file}")

    # Show some example filtered pairs
    print("\nExample filtered pairs:")
    import random

    random.seed(42)
    samples = random.sample(filtered_pairs, min(3, len(filtered_pairs)))

    for i, sample in enumerate(samples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Query: {sample['query'][:200]}...")
        print(f"Passage: {sample['passage'][:200]}...")


if __name__ == "__main__":
    main()
