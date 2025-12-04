"""Base class for data processors with common patterns and error handling."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dnd_dm_copilot.utils import save_json_pairs, load_config, upload_to_huggingface

logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


class BaseDataProcessor(ABC):
    """Base class for standardized data processing pipeline."""

    def __init__(
        self,
        output_file: str,
        repo_id: str = "",
        upload: bool = True,
        config: Any = None,
    ):
        """
        Initialize processor.

        Args:
            output_file: Path where processed data will be saved
            repo_id: Optional HuggingFace repository ID for uploading
            upload: Whether to upload to HuggingFace
            config: Optional Config object (defaults to loading from environment)
        """
        self.output_file = output_file
        self.repo_id = repo_id
        self.upload_enabled = upload
        self.config = config or load_config()

    @abstractmethod
    def load_data(self) -> Any:
        """Load raw data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def process_data(self, raw_data: Any) -> List[Dict[str, str]]:
        """Process raw data into query-passage pairs. Must be implemented by subclasses."""
        pass

    def validate_pairs(self, pairs: List[Dict[str, str]]) -> None:
        """
        Validate that pairs have the correct structure.

        Args:
            pairs: List of query-passage pairs

        Raises:
            DataProcessingError: If validation fails
        """
        if not pairs:
            raise DataProcessingError("No pairs to validate")

        if not all(isinstance(pair, dict) for pair in pairs):
            raise DataProcessingError("All pairs must be dictionaries")

        if not all("query" in pair and "passage" in pair for pair in pairs):
            raise DataProcessingError("All pairs must have 'query' and 'passage' keys")

        logger.info(f"Validation passed for {len(pairs)} pairs")

    def save_data(self, pairs: List[Dict[str, str]]) -> None:
        """
        Save processed data to JSON file.

        Args:
            pairs: List of query-passage pairs

        Raises:
            DataProcessingError: If save fails
        """
        try:
            save_json_pairs(pairs, self.output_file)
            logger.info(f"Saved {len(pairs)} pairs to {self.output_file}")
        except Exception as e:
            raise DataProcessingError(f"Failed to save data: {e}") from e

    def upload_data(self) -> None:
        """
        Upload processed data to HuggingFace if configured.

        Logs warning but doesn't fail if upload doesn't work.
        """
        if not self.upload_enabled or not self.repo_id:
            logger.debug("Upload disabled or no repository ID provided")
            return

        try:
            logger.info(f"Uploading to {self.repo_id}...")
            upload_to_huggingface(self.output_file, self.repo_id, self.config.hf_token)
            logger.info("Upload complete")
        except Exception as e:
            logger.warning(f"Failed to upload to HuggingFace: {e}")
            logger.info("Data was saved locally but not uploaded")

    def process(self) -> List[Dict[str, str]]:
        """
        Execute the full processing pipeline.

        Returns:
            List of processed query-passage pairs

        Raises:
            DataProcessingError: If processing fails
        """
        try:
            logger.info("Starting data processing pipeline...")

            # Load
            logger.info("Loading raw data...")
            raw_data = self.load_data()

            # Process
            logger.info("Processing data...")
            pairs = self.process_data(raw_data)

            # Validate
            logger.info("Validating processed data...")
            self.validate_pairs(pairs)

            # Save
            self.save_data(pairs)

            # Upload
            if self.upload_enabled:
                self.upload_data()

            logger.info("Processing pipeline complete")
            return pairs

        except DataProcessingError:
            raise
        except Exception as e:
            logger.error(f"Processing pipeline failed: {e}")
            raise DataProcessingError(f"Unexpected error in processing: {e}") from e
