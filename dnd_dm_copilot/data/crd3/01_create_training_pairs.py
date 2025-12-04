#!/usr/bin/env python3
"""
Create training pairs from CRD3 data without LLM inference
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import dotenv
import huggingface_hub

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def load_crd3_data(data_dir: str) -> List[Dict]:
    """Load CRD3 data from aligned data directory"""
    episodes = []

    # Load from c=2 directory (you can also use c=3, c=4)
    c2_dir = os.path.join(data_dir, "data", "aligned data", "c=2")

    for filename in os.listdir(c2_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(c2_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    episode_name = filename.split("_")[0]
                    episodes.append({"episode_name": episode_name, "data": data})
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return episodes


def extract_dialogues_from_episode(episode_data: List[Dict]) -> List[Tuple[str, str]]:
    """Extract dialogues from episode data"""
    dialogues = []

    for item in episode_data:
        for turn in item["TURNS"]:
            if len(turn["NAMES"]) > 1:
                speaker = "in unison " + " ".join(turn["NAMES"])
            else:
                speaker = turn["NAMES"][0]

            utterance = " ".join(turn["UTTERANCES"])
            dialogues.append((speaker, utterance))

    return dialogues


def create_dialogue_context_pairs(
    dialogues: List[Tuple[str, str]], context_window: int = 3
) -> List[Dict]:
    """Create pairs using dialogue context"""
    pairs = []

    for i in range(len(dialogues) - context_window):
        # Context as query
        context = " ".join(
            [
                f"{name}: {utterance}"
                for name, utterance in dialogues[i : i + context_window]
            ]
        )

        # Next utterance as passage
        next_name, next_utterance = dialogues[i + context_window]
        target = f"{next_name}: {next_utterance}"

        pairs.append({"query": f"Dialogue context: {context}", "passage": target})

    return pairs


def create_speaker_interaction_pairs(dialogues: List[Tuple[str, str]]) -> List[Dict]:
    """Create pairs based on speaker interactions"""
    pairs = []

    for i in range(len(dialogues) - 1):
        current_speaker, current_utterance = dialogues[i]
        next_speaker, next_utterance = dialogues[i + 1]

        if current_speaker != next_speaker:
            pairs.append(
                {
                    "query": f"{current_speaker} says: {current_utterance}",
                    "passage": f"{next_speaker} responds: {next_utterance}",
                }
            )

    return pairs


def create_chunk_dialogue_pairs(episode_data: List[Dict]) -> List[Dict]:
    """Create pairs using chunk summaries as context"""
    pairs = []

    for item in episode_data:
        chunk_summary = item["CHUNK"]

        for turn in item["TURNS"]:
            speaker = (
                turn["NAMES"][0]
                if len(turn["NAMES"]) == 1
                else "in unison " + " ".join(turn["NAMES"])
            )
            utterance = " ".join(turn["UTTERANCES"])

            pairs.append(
                {
                    "query": f"Scene context: {chunk_summary}",
                    "passage": f"{speaker}: {utterance}",
                }
            )

    return pairs


def create_turn_sequence_pairs(
    dialogues: List[Tuple[str, str]], sequence_length: int = 5
) -> List[Dict]:
    """Create pairs from turn sequences"""
    pairs = []

    for i in range(len(dialogues) - sequence_length):
        sequence = dialogues[i : i + sequence_length]

        # Split sequence in half
        mid_point = sequence_length // 2
        query_turns = sequence[:mid_point]
        passage_turns = sequence[mid_point:]

        query = " ".join([f"{name}: {utterance}" for name, utterance in query_turns])
        passage = " ".join(
            [f"{name}: {utterance}" for name, utterance in passage_turns]
        )

        pairs.append({"query": query, "passage": passage})

    return pairs


def create_matt_dm_pairs(dialogues: List[Tuple[str, str]]) -> List[Dict]:
    """Create pairs specifically for DM responses (Matt's lines)"""
    pairs = []

    for i in range(len(dialogues) - 1):
        current_speaker, current_utterance = dialogues[i]
        next_speaker, next_utterance = dialogues[i + 1]

        # If next speaker is Matt (DM), create a pair
        if next_speaker == "MATT":
            pairs.append(
                {
                    "query": f"Player says: {current_utterance}",
                    "passage": f"DM responds: {next_utterance}",
                }
            )

    return pairs


def create_training_dataset(
    episodes: List[Dict], methods: Optional[List[str]] = None
) -> List[Dict]:
    """Create training dataset using multiple methods"""
    if methods is None:
        methods = [
            "dialogue_context",
            "speaker_interaction",
            "chunk_dialogue",
            "turn_sequence",
            "matt_dm",
        ]

    all_pairs = []

    for episode in episodes:
        episode_name = episode["episode_name"]
        episode_data = episode["data"]

        print(f"Processing episode: {episode_name}")

        # Extract dialogues
        dialogues = extract_dialogues_from_episode(episode_data)

        # Create pairs using different methods
        if "dialogue_context" in methods:
            pairs = create_dialogue_context_pairs(dialogues)
            all_pairs.extend(pairs)
            print(f"  Created {len(pairs)} dialogue context pairs")

        if "speaker_interaction" in methods:
            pairs = create_speaker_interaction_pairs(dialogues)
            all_pairs.extend(pairs)
            print(f"  Created {len(pairs)} speaker interaction pairs")

        if "chunk_dialogue" in methods:
            pairs = create_chunk_dialogue_pairs(episode_data)
            all_pairs.extend(pairs)
            print(f"  Created {len(pairs)} chunk dialogue pairs")

        if "turn_sequence" in methods:
            pairs = create_turn_sequence_pairs(dialogues)
            all_pairs.extend(pairs)
            print(f"  Created {len(pairs)} turn sequence pairs")

        if "matt_dm" in methods:
            pairs = create_matt_dm_pairs(dialogues)
            all_pairs.extend(pairs)
            print(f"  Created {len(pairs)} Matt DM pairs")

    return all_pairs


def main():
    # Load CRD3 data
    # get dir from where call
    #
    episodes = load_crd3_data(os.getcwd())

    print(f"Loaded {len(episodes)} episodes")

    # Create training pairs using multiple methods
    training_pairs = create_training_dataset(
        episodes,
        methods=[
            "dialogue_context",
            "speaker_interaction",
            "chunk_dialogue",
            "matt_dm",
        ],
    )

    print(f"Total training pairs created: {len(training_pairs)}")

    json_output = "crd3_training_pairs_no_llm.json"
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)

    huggingface_hub.create_repo(
        "garrykuwanto/crd3_training_pairs",
        repo_type="dataset",
        token=HF_TOKEN,
        exist_ok=True,
    )

    # Upload to garrykuwanto/crd3_training_pairs
    huggingface_hub.upload_file(
        path_or_fileobj=json_output,
        path_in_repo=json_output,
        repo_id="garrykuwanto/crd3_training_pairs",
        repo_type="dataset",
        token=HF_TOKEN,
    )

    print(f"Saved as JSON: {json_output}")
    print("Uploaded to garrykuwanto/crd3_training_pairs")


if __name__ == "__main__":
    main()
