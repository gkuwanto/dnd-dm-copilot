"""Tests for base data processor."""

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from dnd_dm_copilot.data.base_processor import (
    BaseDataProcessor,
    DataProcessingError,
)


class ConcreteProcessor(BaseDataProcessor):
    """Concrete implementation for testing."""

    def __init__(self, *args, load_return=None, process_return=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_return = load_return or []
        self.process_return = process_return or []

    def load_data(self) -> Any:
        return self.load_return

    def process_data(self, raw_data: Any) -> List[Dict[str, str]]:
        return self.process_return


class TestBaseDataProcessor:
    """Tests for BaseDataProcessor class."""

    def test_initialization(self, temp_dir):
        """Test processor initialization."""
        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(output_file=output_file, upload=False)

        assert processor.output_file == output_file
        assert processor.upload_enabled is False

    def test_initialization_with_config(self, temp_dir, mock_config):
        """Test initialization with custom config."""
        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(
            output_file=output_file, config=mock_config, upload=False
        )

        assert processor.config is mock_config

    def test_validate_pairs_success(self, sample_query_passage_pairs):
        """Test successful pair validation."""
        processor = ConcreteProcessor(output_file="test.json", upload=False)

        # Should not raise
        processor.validate_pairs(sample_query_passage_pairs)

    def test_validate_pairs_empty(self):
        """Test validation with empty pairs."""
        processor = ConcreteProcessor(output_file="test.json", upload=False)

        with pytest.raises(DataProcessingError, match="No pairs to validate"):
            processor.validate_pairs([])

    def test_validate_pairs_invalid_type(self):
        """Test validation with non-dict items."""
        processor = ConcreteProcessor(output_file="test.json", upload=False)

        with pytest.raises(DataProcessingError, match="must be dictionaries"):
            processor.validate_pairs([{"query": "test"}, "not a dict"])

    def test_validate_pairs_missing_query(self):
        """Test validation with missing query field."""
        processor = ConcreteProcessor(output_file="test.json", upload=False)

        with pytest.raises(
            DataProcessingError, match="must have 'query' and 'passage' keys"
        ):
            processor.validate_pairs([{"passage": "test"}])

    def test_validate_pairs_missing_passage(self):
        """Test validation with missing passage field."""
        processor = ConcreteProcessor(output_file="test.json", upload=False)

        with pytest.raises(
            DataProcessingError, match="must have 'query' and 'passage' keys"
        ):
            processor.validate_pairs([{"query": "test"}])

    @patch("dnd_dm_copilot.data.base_processor.save_json_pairs")
    def test_save_data_success(self, mock_save, sample_query_passage_pairs, temp_dir):
        """Test successful data saving."""
        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(output_file=output_file, upload=False)

        processor.save_data(sample_query_passage_pairs)

        mock_save.assert_called_once_with(sample_query_passage_pairs, output_file)

    @patch("dnd_dm_copilot.data.base_processor.save_json_pairs")
    def test_save_data_failure(self, mock_save, sample_query_passage_pairs, temp_dir):
        """Test data saving failure."""
        mock_save.side_effect = Exception("Save failed")

        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(output_file=output_file, upload=False)

        with pytest.raises(DataProcessingError, match="Failed to save data"):
            processor.save_data(sample_query_passage_pairs)

    @patch("dnd_dm_copilot.data.base_processor.upload_to_huggingface")
    def test_upload_data_success(self, mock_upload, temp_dir, mock_config):
        """Test successful data upload."""
        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(
            output_file=output_file,
            repo_id="test/repo",
            upload=True,
            config=mock_config,
        )

        processor.upload_data()

        mock_upload.assert_called_once_with(
            output_file, "test/repo", mock_config.hf_token
        )

    @patch("dnd_dm_copilot.data.base_processor.upload_to_huggingface")
    def test_upload_data_disabled(self, mock_upload, temp_dir):
        """Test that upload is skipped when disabled."""
        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(output_file=output_file, upload=False)

        processor.upload_data()

        mock_upload.assert_not_called()

    @patch("dnd_dm_copilot.data.base_processor.upload_to_huggingface")
    def test_upload_data_no_repo_id(self, mock_upload, temp_dir):
        """Test that upload is skipped when no repo_id."""
        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(output_file=output_file, repo_id="", upload=True)

        processor.upload_data()

        mock_upload.assert_not_called()

    @patch("dnd_dm_copilot.data.base_processor.upload_to_huggingface")
    def test_upload_data_failure(self, mock_upload, temp_dir, mock_config):
        """Test that upload failure is handled gracefully."""
        mock_upload.side_effect = Exception("Upload failed")

        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(
            output_file=output_file,
            repo_id="test/repo",
            upload=True,
            config=mock_config,
        )

        # Should not raise, just log warning
        processor.upload_data()

        mock_upload.assert_called_once()

    @patch("dnd_dm_copilot.data.base_processor.save_json_pairs")
    @patch("dnd_dm_copilot.data.base_processor.upload_to_huggingface")
    def test_process_full_pipeline(
        self, mock_upload, mock_save, sample_query_passage_pairs, temp_dir, mock_config
    ):
        """Test full processing pipeline."""
        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(
            output_file=output_file,
            repo_id="test/repo",
            upload=True,
            config=mock_config,
            load_return={"raw": "data"},
            process_return=sample_query_passage_pairs,
        )

        result = processor.process()

        assert result == sample_query_passage_pairs
        mock_save.assert_called_once()
        mock_upload.assert_called_once()

    def test_process_load_failure(self, temp_dir):
        """Test pipeline handles load failure."""

        class FailingProcessor(ConcreteProcessor):
            def load_data(self):
                raise Exception("Load failed")

        processor = FailingProcessor(output_file=str(temp_dir / "out.json"))

        with pytest.raises(DataProcessingError, match="Unexpected error in processing"):
            processor.process()

    def test_process_validation_failure(self, temp_dir):
        """Test pipeline handles validation failure."""
        output_file = str(temp_dir / "output.json")
        processor = ConcreteProcessor(
            output_file=output_file,
            upload=False,
            load_return={"raw": "data"},
            process_return=[{"invalid": "pair"}],  # Missing query/passage
        )

        with pytest.raises(DataProcessingError):
            processor.process()
