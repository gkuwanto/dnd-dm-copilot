"""HuggingFace utilities for uploading datasets and models."""

import logging
import time

from huggingface_hub import upload_file

logger = logging.getLogger(__name__)


class HuggingFaceError(Exception):
    """Custom exception for HuggingFace operations."""

    pass


def upload_to_huggingface(
    file_path: str,
    repo_id: str,
    token: str,
    repo_type: str = "dataset",
    max_retries: int = 3,
    initial_wait: float = 1.0,
) -> None:
    """
    Upload a file to HuggingFace with retry logic and exponential backoff.

    Args:
        file_path: Local path to the file to upload
        repo_id: HuggingFace repository ID (format: "username/repo-name")
        token: HuggingFace API token
        repo_type: Type of repository ("dataset" or "model")
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time between retries in seconds

    Raises:
        HuggingFaceError: If upload fails after all retries
    """
    if repo_type not in ("dataset", "model"):
        raise ValueError(f"repo_type must be 'dataset' or 'model', got '{repo_type}'")

    wait_time = initial_wait
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Uploading {file_path} to {repo_id}"
                f" (attempt {attempt + 1}/{max_retries})"
            )

            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path.split("/")[-1],
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
            )

            logger.info(f"Successfully uploaded {file_path} to {repo_id}")
            return

        except Exception as e:
            last_error = e
            logger.warning(f"Upload attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff

    raise HuggingFaceError(
        f"Failed to upload {file_path} after {max_retries} attempts: {last_error}"
    )
