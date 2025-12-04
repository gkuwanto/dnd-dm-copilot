#!/usr/bin/env python3
"""
Filter CRD3 DM-only pairs for highest quality
Aims for ~40k pairs (similar magnitude to dnd-mechanics dataset)
"""

import json
import re
from typing import Dict, List


def has_meaningful_content(text: str) -> bool:
    """Check if text has meaningful DM narration or rules content"""
    # Remove DM responds prefix
    text = text.replace("DM responds:", "").strip()

    # Must be substantial
    if len(text.split()) < 10:
        return False

    # Look for indicators of low-quality responses
    low_quality_patterns = [
        r"^\d+\.$",  # Just numbers
        r"^(Okay|All right|Sure|Yes|No|Yeah|Yep)[\.,!]?\s*$",  # Simple acknowledgments
        r"^You (can|do|are|have)[\.,]?\s*$",  # Incomplete statements
    ]

    for pattern in low_quality_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False

    return True


def has_dm_narration(text: str) -> bool:
    """Check if text contains DM narration (world-building, descriptions)"""
    narration_indicators = [
        "you see",
        "you notice",
        "you hear",
        "you find",
        "the room",
        "the air",
        "the sound",
        "the smell",
        "as you",
        "in front of you",
        "around you",
        "suddenly",
        "slowly",
        "carefully",
        "darkness",
        "light",
        "shadow",
        "figure",
        "creature",
        "monster",
        "beast",
        "door",
        "hallway",
        "chamber",
        "corridor",
    ]

    text_lower = text.lower()
    return any(indicator in text_lower for indicator in narration_indicators)


def has_rules_content(text: str) -> bool:
    """Check if text contains D&D rules or mechanics"""
    rules_indicators = [
        "roll",
        "check",
        "saving throw",
        "save",
        "attack",
        "damage",
        "hit points",
        "hp",
        "ac",
        "armor class",
        "spell",
        "cast",
        "concentration",
        "advantage",
        "disadvantage",
        "action",
        "bonus action",
        "reaction",
        "ki point",
        "spell slot",
        "rage",
        "proficiency",
        "modifier",
        "d20",
        "d6",
        "d8",
        "d10",
        "d12",
    ]

    text_lower = text.lower()
    return any(indicator in text_lower for indicator in rules_indicators)


def is_conversational_filler(text: str) -> bool:
    """Check if text is mostly conversational filler without DM content"""
    # Remove speaker prefix
    text = text.replace("Player says:", "").strip()

    # Short queries are often just filler
    if len(text.split()) < 5:
        return True

    # Check for pure meta-conversation (about the game, not in-game)
    meta_only_patterns = [
        r"can i (do|use|make|take|have)",
        r"do i (get|have|need|roll)",
        r"how (do|does|did)",
        r"what (is|does|do|did)",
        r"^(okay|all right|sure|yes|no|yeah)",
    ]

    text_lower = text.lower()
    for pattern in meta_only_patterns:
        if re.match(pattern, text_lower):
            return True

    return False


def calculate_quality_score(pair: Dict) -> float:
    """Calculate quality score for a query-passage pair"""
    query = pair["query"]
    passage = pair["passage"]

    score = 0.0

    # Passage quality (more important)
    if has_meaningful_content(passage):
        score += 2.0

    if has_dm_narration(passage):
        score += 1.5

    if has_rules_content(passage):
        score += 1.5

    # Passage length bonus (prefer substantial responses)
    passage_words = len(passage.split())
    if passage_words >= 30:
        score += 1.0
    elif passage_words >= 20:
        score += 0.5

    # Query quality (less important but still matters)
    if not is_conversational_filler(query):
        score += 1.0

    query_words = len(query.split())
    if query_words >= 10:
        score += 0.5

    return score


def filter_high_quality_pairs(
    pairs: List[Dict],
    target_count: int = 40000,
    min_quality_score: float = 3.0,
) -> List[Dict]:
    """Filter for highest quality pairs"""
    print("Calculating quality scores...")

    # Score all pairs
    scored_pairs = []
    for pair in pairs:
        score = calculate_quality_score(pair)
        if score >= min_quality_score:
            scored_pairs.append((score, pair))

    print(f"Pairs with score >= {min_quality_score}: {len(scored_pairs):,}")

    # Sort by quality score (highest first)
    scored_pairs.sort(reverse=True, key=lambda x: x[0])

    # Take top N pairs
    top_pairs = [pair for score, pair in scored_pairs[:target_count]]

    return top_pairs


def print_quality_analysis(pairs: List[Dict]) -> None:
    """Print analysis of selected pairs"""
    narration_count = sum(1 for p in pairs if has_dm_narration(p["passage"]))
    rules_count = sum(1 for p in pairs if has_rules_content(p["passage"]))

    avg_query_len = sum(len(p["query"].split()) for p in pairs) / len(pairs)
    avg_passage_len = sum(len(p["passage"].split()) for p in pairs) / len(pairs)

    print("\nQuality Analysis:")
    print(
        f"  Pairs with DM narration: {narration_count:,}"
        f" ({narration_count / len(pairs) * 100:.1f}%)"
    )
    print(
        f"  Pairs with rules content: {rules_count:,}"
        f" ({rules_count / len(pairs) * 100:.1f}%)"
    )
    print(f"  Average query length: {avg_query_len:.1f} words")
    print(f"  Average passage length: {avg_passage_len:.1f} words")


def main() -> None:
    input_file = "crd3_training_pairs_dm_only.json"
    output_file = "crd3_training_pairs_high_quality.json"
    target_count = 40000

    print(f"Loading DM-only pairs from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        dm_pairs = json.load(f)

    print(f"Loaded {len(dm_pairs):,} DM response pairs")
    print(f"Target: {target_count:,} high-quality pairs\n")

    # Filter for high quality
    high_quality_pairs = filter_high_quality_pairs(dm_pairs, target_count=target_count)

    print(f"\n{'=' * 60}")
    print(f"Selected {len(high_quality_pairs):,} highest quality pairs")
    print(
        f"Reduction: {len(dm_pairs):,} → {len(high_quality_pairs):,}"
        f" ({len(high_quality_pairs) / len(dm_pairs) * 100:.1f}%)"
    )
    print(f"{'=' * 60}")

    # Analyze quality
    print_quality_analysis(high_quality_pairs)

    # Save filtered data
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(high_quality_pairs, f, indent=2, ensure_ascii=False)

    print(f"✓ High-quality dataset saved: {output_file}")

    # Show examples
    print("\nExample high-quality pairs:")
    import random

    random.seed(42)
    samples = random.sample(high_quality_pairs[:1000], min(3, len(high_quality_pairs)))

    for i, sample in enumerate(samples, 1):
        score = calculate_quality_score(sample)
        print(f"\n--- Example {i} (Quality Score: {score:.1f}) ---")
        print(f"Query: {sample['query'][:150]}...")
        print(f"Passage: {sample['passage'][:150]}...")


if __name__ == "__main__":
    main()
