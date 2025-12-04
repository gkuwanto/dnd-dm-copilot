#!/usr/bin/env python3
"""
Comprehensive FIREBALL training pair creation
Includes all possible ways of creating training pairs from FIREBALL data
"""

import gc
import json
import os
import re
from typing import Dict, List

import dotenv
import huggingface_hub

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def load_fireball_dataset_batch(start_idx: int = 0, batch_size: int = 1000) -> list:
    """Load FIREBALL dataset in batches to manage memory"""
    try:
        from datasets import load_dataset

        dataset = load_dataset("lara-martin/FIREBALL", split="train")

        # Return a batch of sessions
        end_idx = min(start_idx + batch_size, len(dataset))
        return [session for session in dataset.select(range(start_idx, end_idx))]
    except Exception as e:
        print(f"Error loading FIREBALL dataset: {e}")
        return []


def get_dataset_size() -> int:
    """Get the total size of the FIREBALL dataset"""
    try:
        from datasets import load_dataset

        dataset = load_dataset("lara-martin/FIREBALL", split="train")
        return len(dataset)
    except Exception as e:
        print(f"Error getting dataset size: {e}")
        return 0


def normalize_field(field: object) -> list:
    """Normalize field to list format"""
    if isinstance(field, str):
        return [field] if field else []
    elif isinstance(field, list):
        return field
    else:
        return []


def create_utterance_command_pairs(session: Dict) -> List[Dict]:
    """Create pairs from utterances and commands"""
    pairs = []

    before_utterances = normalize_field(session.get("before_utterances", []))
    commands_norm = normalize_field(session.get("commands_norm", ""))
    utterance_history = normalize_field(session.get("utterance_history", []))

    # Pair 1: Before utterances -> Commands
    for utterance in before_utterances:
        for command in commands_norm:
            if utterance and command:
                pairs.append(
                    {
                        "query": f"Player says: {utterance}",
                        "passage": f"Command: {command}",
                    }
                )

    # Pair 2: Utterance history -> Commands
    for utterance in utterance_history:
        for command in commands_norm:
            if utterance and command:
                pairs.append(
                    {"query": f"Context: {utterance}", "passage": f"Action: {command}"}
                )

    return pairs


def create_command_result_pairs(session: Dict) -> List[Dict]:
    """Create pairs from commands and automation results"""
    pairs = []

    commands_norm = normalize_field(session.get("commands_norm", ""))
    automation_results = normalize_field(session.get("automation_results", ""))

    # Basic command -> result pairs
    for command in commands_norm:
        for result in automation_results:
            if command and result:
                pairs.append(
                    {"query": f"Command: {command}", "passage": f"Result: {result}"}
                )

    return pairs


def create_spell_specific_pairs(session: Dict) -> List[Dict]:
    """Create pairs specifically for spell casting"""
    pairs = []

    commands_norm = normalize_field(session.get("commands_norm", ""))
    automation_results = normalize_field(session.get("automation_results", ""))

    for command in commands_norm:
        for result in automation_results:
            if command and result:
                # Spell casting patterns
                if "cast" in command.lower() or "spell" in command.lower():
                    if "casts" in result.lower() or "spell" in result.lower():
                        pairs.append(
                            {
                                "query": f"Spell command: {command}",
                                "passage": f"Spell result: {result}",
                            }
                        )

                # Specific spell patterns
                if any(
                    spell in command.lower()
                    for spell in [
                        "fireball",
                        "heal",
                        "cure",
                        "magic",
                        "armor",
                        "shield",
                    ]
                ):
                    pairs.append(
                        {
                            "query": f"Spell casting: {command}",
                            "passage": f"Spell effect: {result}",
                        }
                    )

    return pairs


def create_attack_specific_pairs(session: Dict) -> List[Dict]:
    """Create pairs specifically for attacks"""
    pairs = []

    commands_norm = normalize_field(session.get("commands_norm", ""))
    automation_results = normalize_field(session.get("automation_results", ""))

    for command in commands_norm:
        for result in automation_results:
            if command and result:
                # Attack patterns
                if any(
                    word in command.lower()
                    for word in ["attack", "hit", "strike", "weapon"]
                ):
                    if any(
                        word in result.lower()
                        for word in ["hit", "miss", "damage", "attack"]
                    ):
                        pairs.append(
                            {
                                "query": f"Attack command: {command}",
                                "passage": f"Attack result: {result}",
                            }
                        )

                # Damage patterns
                if "damage" in result.lower() or re.search(r"\d+", result):
                    pairs.append(
                        {
                            "query": f"Combat action: {command}",
                            "passage": f"Damage dealt: {result}",
                        }
                    )

    return pairs


def create_dice_roll_pairs(session: Dict) -> List[Dict]:
    """Create pairs from dice rolls"""
    pairs = []

    commands_norm = normalize_field(session.get("commands_norm", ""))
    automation_results = normalize_field(session.get("automation_results", ""))

    for command in commands_norm:
        for result in automation_results:
            if command and result:
                # Dice roll patterns
                if re.search(r"\d+d\d+", command):  # Contains dice notation
                    if re.search(r"\d+", result):  # Contains numbers
                        pairs.append(
                            {
                                "query": f"Dice roll: {command}",
                                "passage": f"Roll result: {result}",
                            }
                        )

                # Skill check patterns
                if any(
                    skill in command.lower()
                    for skill in ["roll", "check", "save", "skill"]
                ):
                    pairs.append(
                        {
                            "query": f"Skill check: {command}",
                            "passage": f"Check result: {result}",
                        }
                    )

    return pairs


def create_combat_state_pairs(session: Dict) -> List[Dict]:
    """Create pairs from combat state changes"""
    pairs: List[Dict] = []

    combat_before = session.get("combat_state_before", [])
    combat_after = session.get("combat_state_after", [])

    if not combat_before or not combat_after:
        return pairs

    # Compare character states
    for i, (before_char, after_char) in enumerate(zip(combat_before, combat_after)):
        if not isinstance(before_char, dict) or not isinstance(after_char, dict):
            continue

        char_name = before_char.get("name", f"Character {i}")

        # HP changes
        before_hp = before_char.get("hp", "")
        after_hp = after_char.get("hp", "")

        if before_hp and after_hp and before_hp != after_hp:
            pairs.append(
                {
                    "query": f"Character HP: {char_name} - {before_hp}",
                    "passage": f"Updated HP: {char_name} - {after_hp}",
                }
            )

        # Effects changes
        before_effects = before_char.get("effects", "")
        after_effects = after_char.get("effects", "")

        if before_effects != after_effects:
            pairs.append(
                {
                    "query": f"Character effects: {char_name} - {before_effects}",
                    "passage": f"New effects: {char_name} - {after_effects}",
                }
            )

        # Class and race information
        char_class = before_char.get("class", "")
        char_race = before_char.get("race", "")

        if char_class and char_race:
            pairs.append(
                {
                    "query": f"Character info: {char_name}",
                    "passage": f"Class: {char_class}, Race: {char_race}",
                }
            )

        # Spells and abilities
        spells = before_char.get("spells", "")
        attacks = before_char.get("attacks", "")

        if spells:
            pairs.append(
                {
                    "query": f"Character spells: {char_name}",
                    "passage": f"Available spells: {spells}",
                }
            )

        if attacks:
            pairs.append(
                {
                    "query": f"Character attacks: {char_name}",
                    "passage": f"Available attacks: {attacks}",
                }
            )

    return pairs


def create_caster_target_pairs(session: Dict) -> List[Dict]:
    """Create pairs from caster and target information"""
    pairs = []

    caster_after = session.get("caster_after", {})
    targets_after = session.get("targets_after", [])

    if caster_after and targets_after:
        caster_name = caster_after.get("name", "Unknown")

        for target in targets_after:
            if isinstance(target, dict):
                target_name = target.get("name", "Unknown")

                pairs.append(
                    {
                        "query": f"Caster: {caster_name}",
                        "passage": f"Target: {target_name}",
                    }
                )

                # Caster effects
                caster_effects = caster_after.get("effects", "")
                if caster_effects:
                    pairs.append(
                        {
                            "query": f"Caster effects: {caster_name}",
                            "passage": f"Active effects: {caster_effects}",
                        }
                    )

                # Target state
                target_hp = target.get("hp", "")
                if target_hp:
                    pairs.append(
                        {
                            "query": f"Target state: {target_name}",
                            "passage": f"HP: {target_hp}",
                        }
                    )

    return pairs


def create_contextual_pairs(session: Dict) -> List[Dict]:
    """Create pairs with contextual information"""
    pairs = []

    before_utterances = normalize_field(session.get("before_utterances", []))
    utterance_history = normalize_field(session.get("utterance_history", []))
    commands_norm = normalize_field(session.get("commands_norm", ""))
    automation_results = normalize_field(session.get("automation_results", ""))

    # Combine context from multiple sources
    all_context = before_utterances + utterance_history

    for context in all_context:
        for command in commands_norm:
            if context and command:
                pairs.append(
                    {
                        "query": f"Game context: {context}",
                        "passage": f"Player action: {command}",
                    }
                )

        for result in automation_results:
            if context and result:
                pairs.append(
                    {
                        "query": f"Game context: {context}",
                        "passage": f"Game outcome: {result}",
                    }
                )

    return pairs


def create_sequence_pairs(session: Dict) -> List[Dict]:
    """Create pairs from sequences of actions"""
    pairs = []

    before_utterances = normalize_field(session.get("before_utterances", []))
    utterance_history = normalize_field(session.get("utterance_history", []))
    commands_norm = normalize_field(session.get("commands_norm", ""))
    automation_results = normalize_field(session.get("automation_results", ""))

    # Create sequences
    all_actions = (
        before_utterances + utterance_history + commands_norm + automation_results
    )

    for i in range(len(all_actions) - 1):
        current_action = all_actions[i]
        next_action = all_actions[i + 1]

        if current_action and next_action:
            pairs.append(
                {
                    "query": f"Previous action: {current_action}",
                    "passage": f"Next action: {next_action}",
                }
            )

    return pairs


def create_specialized_pairs(session: Dict) -> List[Dict]:
    """Create specialized pairs for specific game mechanics"""
    pairs = []

    commands_norm = normalize_field(session.get("commands_norm", ""))
    automation_results = normalize_field(session.get("automation_results", ""))

    for command in commands_norm:
        for result in automation_results:
            if command and result:
                # Initiative and turn order
                if "init" in command.lower() or "initiative" in command.lower():
                    pairs.append(
                        {
                            "query": f"Initiative command: {command}",
                            "passage": f"Initiative result: {result}",
                        }
                    )

                # Saving throws
                if "save" in command.lower() or "saving" in command.lower():
                    pairs.append(
                        {
                            "query": f"Saving throw: {command}",
                            "passage": f"Save result: {result}",
                        }
                    )

                # Ability checks
                if any(
                    ability in command.lower()
                    for ability in ["str", "dex", "con", "int", "wis", "cha"]
                ):
                    pairs.append(
                        {
                            "query": f"Ability check: {command}",
                            "passage": f"Check result: {result}",
                        }
                    )

                # Movement and positioning
                if any(
                    word in command.lower()
                    for word in ["move", "position", "location", "teleport"]
                ):
                    pairs.append(
                        {
                            "query": f"Movement command: {command}",
                            "passage": f"Movement result: {result}",
                        }
                    )

    return pairs


def process_batch_comprehensive(sessions: List[Dict]) -> List[Dict]:
    """Process a batch of sessions with all pair creation methods"""
    all_pairs = []

    for session in sessions:
        # Create all types of pairs
        utterance_pairs = create_utterance_command_pairs(session)
        all_pairs.extend(utterance_pairs)

        command_pairs = create_command_result_pairs(session)
        all_pairs.extend(command_pairs)

        spell_pairs = create_spell_specific_pairs(session)
        all_pairs.extend(spell_pairs)

        attack_pairs = create_attack_specific_pairs(session)
        all_pairs.extend(attack_pairs)

        dice_pairs = create_dice_roll_pairs(session)
        all_pairs.extend(dice_pairs)

        combat_pairs = create_combat_state_pairs(session)
        all_pairs.extend(combat_pairs)

        caster_pairs = create_caster_target_pairs(session)
        all_pairs.extend(caster_pairs)

        contextual_pairs = create_contextual_pairs(session)
        all_pairs.extend(contextual_pairs)

        sequence_pairs = create_sequence_pairs(session)
        all_pairs.extend(sequence_pairs)

        specialized_pairs = create_specialized_pairs(session)
        all_pairs.extend(specialized_pairs)

    return all_pairs


def save_pairs_to_json(pairs: List[Dict], filename: str) -> None:
    """Save pairs to JSON file"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(pairs)} pairs to {filename}")


def main() -> None:
    # Get dataset size
    total_sessions = get_dataset_size()
    print(f"FIREBALL dataset size: {total_sessions:,} sessions")

    # Configuration
    batch_size = 1000  # Process 1000 sessions at a time
    max_batches = 153  # Limit to first 50 batches for testing

    total_pairs = []
    processed_sessions = 0

    print(f"Processing in batches of {batch_size} sessions")
    print(f"Maximum batches: {max_batches}")
    print("Using comprehensive pair creation methods:")
    print("  - Utterance-Command pairs")
    print("  - Command-Result pairs")
    print("  - Spell-specific pairs")
    print("  - Attack-specific pairs")
    print("  - Dice roll pairs")
    print("  - Combat state pairs")
    print("  - Caster-Target pairs")
    print("  - Contextual pairs")
    print("  - Sequence pairs")
    print("  - Specialized pairs")

    for batch_num in range(max_batches):
        start_idx = batch_num * batch_size

        if start_idx >= total_sessions:
            break

        print(
            f"\nProcessing batch {batch_num + 1}/{max_batches}"
            f" (sessions {start_idx}-{start_idx + batch_size - 1})"
        )

        # Load batch
        sessions = load_fireball_dataset_batch(start_idx, batch_size)
        if not sessions:
            print("No more sessions to process")
            break

        # Process batch with comprehensive methods
        batch_pairs = process_batch_comprehensive(sessions)
        total_pairs.extend(batch_pairs)
        processed_sessions += len(sessions)

        print(f"  Processed {len(sessions)} sessions")
        print(f"  Created {len(batch_pairs)} pairs from this batch")
        print(f"  Total pairs so far: {len(total_pairs)}")

        # Save intermediate results every 10 batches
        if (batch_num + 1) % 10 == 0:
            intermediate_file = (
                f"fireball_comprehensive_batch_{batch_num + 1}_pairs.json"
            )
            save_pairs_to_json(total_pairs, intermediate_file)
            print(f"  Saved intermediate results to {intermediate_file}")

        # Clear memory
        del sessions
        del batch_pairs
        gc.collect()

    # Save final results
    print("\nProcessing complete!")
    print(f"Total sessions processed: {processed_sessions:,}")
    print(f"Total training pairs created: {len(total_pairs):,}")

    # Save to files
    save_pairs_to_json(total_pairs, "fireball_comprehensive_pairs.json")

    print("Saved to fireball_comprehensive_pairs.json")

    huggingface_hub.create_repo(
        "garrykuwanto/fireball_comprehensive_pairs",
        repo_type="dataset",
        token=HF_TOKEN,
        exist_ok=True,
    )

    # Upload to garrykuwanto/fireball_comprehensive_pairs
    huggingface_hub.upload_file(
        path_or_fileobj="fireball_comprehensive_pairs.json",
        path_in_repo="fireball_comprehensive_pairs.json",
        repo_id="garrykuwanto/fireball_comprehensive_pairs",
        repo_type="dataset",
        token=HF_TOKEN,
    )

    print("Uploaded to garrykuwanto/fireball_comprehensive_pairs")

    # Show some examples
    print("\nExample pairs:")
    for i, pair in enumerate(total_pairs[:5]):
        print(f"\nExample {i + 1}:")
        print(f"Query: {pair['query']}")
        print(f"Passage: {pair['passage']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
