#!/usr/bin/env python3
"""
Create training pairs from Reddit campaign notes using DeepSeek LLM.

This pipeline:
1. Loads Reddit notes from text files
2. Chunks content by paragraphs
3. Generates 5 questions per chunk using DeepSeek (async concurrent)
4. Creates query-passage pairs for retrieval evaluation
5. Saves checkpoints periodically and can resume from interruptions
6. Saves locally and uploads to HuggingFace
"""

import asyncio
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import dotenv
import huggingface_hub
import openai

# Load environment
dotenv.load_dotenv()

# Configuration
DATA_DIR = os.path.dirname(__file__)
OUTPUT_FILE = "reddit_training_pairs.json"
CHECKPOINT_FILE = "reddit_checkpoint.json"
HF_REPO = "garrykuwanto/reddit-notes-dataset"
MIN_CHUNK_LENGTH = 50
MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N chunks
CONCURRENT_REQUESTS = 10  # Number of concurrent API calls

SYSTEM_PROMPT = """You are helping create training data for a D&D Dungeon Master copilot system.
The copilot helps DMs search through their campaign notes during gameplay.

Your task: Given a passage from DM campaign notes, generate exactly 5 questions that a DM might ask
when they need to retrieve this specific information during a game session.

Guidelines:
- Questions should be natural and conversational
- Questions should reflect real gameplay scenarios (e.g., "What was the NPC's name?", "Which faction does this character belong to?")
- Cover different aspects: characters, plot, world-building, relationships, mechanics
- Vary question complexity and specificity
- Return ONLY a JSON array of 5 strings, nothing else

Example output format:
["What was the name of the cleric's uncle?", "Which clan does Barhador belong to?", "What is the Eglan sect?", "Who is Barhador's mother?", "What deity does the war cleric worship?"]
"""


def validate_environment() -> None:
    """Validate required environment variables, exit if DEEPSEEK_API_KEY missing."""
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        print("ERROR: DEEPSEEK_API_KEY not found in environment")
        print("Please set it in your .env file")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not found, will skip HuggingFace upload")


def load_reddit_notes(data_dir: str) -> str:
    """Load all notes*.txt files and concatenate."""
    all_text = []
    for i in range(8):
        filepath = os.path.join(data_dir, f"notes{i}.txt")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                all_text.append(f.read())
            print(f"Loaded {filepath}")
    return "\n\n".join(all_text)


def chunk_by_paragraph(text: str, min_length: int = 50) -> List[str]:
    """Split text into paragraphs, filter by minimum length."""
    chunks = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        # Filter out: empty, too short, separator lines
        if chunk and len(chunk) >= min_length and not all(c in "= \n\t" for c in chunk):
            chunks.append(chunk)
    return chunks


def parse_response(response_text: str) -> List[str]:
    """Parse DeepSeek response with fallback."""
    try:
        # Primary: JSON parsing
        questions = json.loads(response_text)
        if isinstance(questions, list):
            return [str(q) for q in questions][:5]
    except json.JSONDecodeError:
        # Fallback: Regex extraction
        questions = re.findall(r'"([^"]+)"', response_text)
        if questions:
            return questions[:5]

    # Last resort: return empty, skip chunk
    return []


def load_checkpoint() -> Tuple[List[Dict[str, str]], int]:
    """Load checkpoint if exists, return (pairs, last_processed_index)."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                pairs = checkpoint.get("pairs", [])
                last_index = checkpoint.get("last_index", -1)
                print(
                    f"Resuming from checkpoint: {last_index + 1} chunks already processed, {len(pairs)} pairs"
                )
                return pairs, last_index
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch")
    return [], -1


def save_checkpoint(pairs: List[Dict[str, str]], last_index: int) -> None:
    """Save checkpoint with current progress."""
    checkpoint = {"pairs": pairs, "last_index": last_index, "total_pairs": len(pairs)}
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


async def generate_queries_for_chunk_async(
    chunk: str,
    chunk_index: int,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, List[str]]:
    """Generate 5 questions using DeepSeek with retry logic (async)."""
    async with semaphore:  # Limit concurrent requests
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"""Passage from DM notes:

{chunk}

Generate exactly 5 questions a DM would ask to find this passage. Return only a JSON array of strings.""",
                        },
                    ],
                    temperature=0.7,
                    max_tokens=500,
                )

                # Parse response
                content = response.choices[0].message.content
                questions = parse_response(content)
                if questions:
                    return chunk_index, questions
                else:
                    # If parsing failed, try next attempt
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    return chunk_index, []

            except openai.RateLimitError:
                wait_time = 2**attempt
                print(f"Rate limit on chunk {chunk_index}, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                print(f"Error on chunk {chunk_index}, attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    return chunk_index, []
                await asyncio.sleep(2**attempt)

    return chunk_index, []


async def create_training_pairs_async(
    chunks: List[str],
    client: openai.AsyncOpenAI,
    start_index: int = 0,
    existing_pairs: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Create query-passage pairs for all chunks using async concurrency."""
    pairs = existing_pairs if existing_pairs else []
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # Process chunks in batches for checkpointing
    chunks_to_process = chunks[start_index + 1 :]
    total_chunks = len(chunks)

    # Create tasks for all remaining chunks
    tasks = [
        generate_queries_for_chunk_async(chunk, start_index + 1 + i, client, semaphore)
        for i, chunk in enumerate(chunks_to_process)
    ]

    # Process tasks and collect results
    completed = 0
    for coro in asyncio.as_completed(tasks):
        chunk_index, queries = await coro
        chunk = chunks[chunk_index]

        # Add pairs for this chunk
        for query in queries:
            pairs.append({"query": query, "passage": chunk})

        completed += 1

        # Progress update
        if completed % 10 == 0:
            print(
                f"Processed {start_index + 1 + completed}/{total_chunks} chunks ({len(pairs)} pairs)"
            )

        # Checkpoint save
        if completed % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(pairs, start_index + completed)
            print(f"Checkpoint saved at {start_index + completed + 1} chunks")

    return pairs


def save_to_json(pairs: List[Dict], filename: str) -> None:
    """Save pairs to JSON with utf-8 encoding."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)


def upload_to_huggingface(filepath: str, repo_name: str, token: str) -> None:
    """Upload to HuggingFace Hub."""
    huggingface_hub.create_repo(
        repo_name,
        repo_type="dataset",
        token=token,
        exist_ok=True,
    )

    huggingface_hub.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=os.path.basename(filepath),
        repo_id=repo_name,
        repo_type="dataset",
        token=token,
    )


async def main_async():
    """Main execution (async)."""
    print("=" * 60)
    print("Reddit Notes Ingestion Pipeline (Async + Checkpointing)")
    print("=" * 60)

    # Validate environment
    validate_environment()

    # Initialize DeepSeek async client
    client = openai.AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )

    # Load checkpoint if exists
    existing_pairs, last_index = load_checkpoint()

    # Load notes
    print("\nLoading Reddit notes...")
    text = load_reddit_notes(DATA_DIR)

    # Chunk by paragraph
    print("\nChunking by paragraph...")
    chunks = chunk_by_paragraph(text, MIN_CHUNK_LENGTH)
    print(f"Created {len(chunks)} chunks")

    if last_index >= len(chunks) - 1:
        print("\nAll chunks already processed!")
        pairs = existing_pairs
    else:
        # Generate pairs
        print(
            f"\nGenerating queries for {len(chunks) - last_index - 1} remaining chunks..."
        )
        print(f"Using {CONCURRENT_REQUESTS} concurrent requests")
        pairs = await create_training_pairs_async(
            chunks, client, last_index, existing_pairs
        )

    # Save final result
    print(f"\nSaving {len(pairs)} pairs to {OUTPUT_FILE}...")
    save_to_json(pairs, OUTPUT_FILE)

    # Clean up checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint file cleaned up")

    # Upload to HuggingFace
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"\nUploading to {HF_REPO}...")
        upload_to_huggingface(OUTPUT_FILE, HF_REPO, hf_token)
    else:
        print("\nSkipping HuggingFace upload (HF_TOKEN not found)")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Total pairs created: {len(pairs)}")
    print("=" * 60)


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
