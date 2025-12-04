"""Process D&D mechanics dataset and create training pairs."""

import logging
from typing import Any, Dict, List

from datasets import load_dataset, Dataset

from dnd_dm_copilot.utils import save_json_pairs, load_config, get_logger

logger = get_logger(__name__)


def load_dnd_mechanics_dataset() -> Dataset:
    """
    Load dnd-mechanics-dataset from Hugging Face.

    Returns:
        Dataset containing D&D mechanics Q&A pairs

    Raises:
        RuntimeError: If dataset cannot be loaded
    """
    try:
        logger.info("Loading D&D mechanics dataset from Hugging Face...")
        dataset = load_dataset("m0no1/dnd-mechanics-dataset", split="train")
        logger.info(f"Successfully loaded {len(dataset)} pairs")
        return dataset

    except Exception as e:
        logger.error(f"Error loading dnd-mechanics-dataset: {e}")
        raise RuntimeError(f"Failed to load D&D mechanics dataset: {e}") from e


def preprocess_dataset(dataset: Dataset) -> List[Dict[str, str]]:
    """
    Preprocess dataset by renaming fields to standardized format.

    Args:
        dataset: Dataset with 'instruction' and 'output' fields

    Returns:
        List of dictionaries with 'query' and 'passage' fields

    Raises:
        ValueError: If required fields are missing
    """
    if not dataset or len(dataset) == 0:
        logger.warning("Empty dataset provided to preprocess_dataset")
        return []

    processed_dataset = []

    for idx, item in enumerate(dataset):
        try:
            if "instruction" not in item or "output" not in item:
                logger.warning(f"Skipping item {idx}: missing required fields")
                continue

            processed_dataset.append({
                "query": item["instruction"],
                "passage": item["output"]
            })

        except (KeyError, TypeError) as e:
            logger.warning(f"Error processing item {idx}: {e}")
            continue

    logger.info(f"Successfully processed {len(processed_dataset)} pairs")
    return processed_dataset


def main(
    output_file: str = "dnd-mechanics-dataset.json",
    repo_id: str = "garrykuwanto/dnd-mechanics-dataset",
    upload: bool = True,
) -> None:
    """
    Main function to load, process, and save D&D mechanics dataset.

    Args:
        output_file: Path where JSON file will be saved
        repo_id: HuggingFace repository ID for uploading
        upload: Whether to upload to HuggingFace

    Raises:
        RuntimeError: If processing fails
    """
    try:
        # Load dataset
        dataset = load_dnd_mechanics_dataset()

        # Preprocess
        processed_dataset = preprocess_dataset(dataset)

        if not processed_dataset:
            raise RuntimeError("No data to process")

        # Save to file
        save_json_pairs(processed_dataset, output_file)
        logger.info(f"Saved {len(processed_dataset)} pairs to {output_file}")

        # Upload to HuggingFace if requested
        if upload:
            try:
                config = load_config()
                from dnd_dm_copilot.utils import upload_to_huggingface

                logger.info(f"Uploading to {repo_id}...")
                upload_to_huggingface(output_file, repo_id, config.hf_token)
                logger.info("Upload complete")

            except Exception as e:
                logger.warning(f"Failed to upload to HuggingFace: {e}")
                logger.info("Data was saved locally but not uploaded")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
