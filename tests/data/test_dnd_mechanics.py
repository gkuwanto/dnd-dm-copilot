"""Tests for D&D mechanics data processor."""

from unittest.mock import MagicMock, patch

import pytest

# Import module with numeric prefix using __import__
dnd_mechanics_module = __import__(
    "dnd_dm_copilot.data.dnd_mechanics.01_create_training_pairs",
    fromlist=["load_dnd_mechanics_dataset", "preprocess_dataset", "main"],
)
load_dnd_mechanics_dataset = dnd_mechanics_module.load_dnd_mechanics_dataset
preprocess_dataset = dnd_mechanics_module.preprocess_dataset
main = dnd_mechanics_module.main


class TestLoadDndMechanicsDataset:
    """Tests for load_dnd_mechanics_dataset function."""

    def test_load_success(self):
        """Test successful dataset loading."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        with patch.object(dnd_mechanics_module, "load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset

            result = load_dnd_mechanics_dataset()

            assert result is mock_dataset
            mock_load_dataset.assert_called_once_with(
                "m0no1/dnd-mechanics-dataset", split="train"
            )

    def test_load_failure(self):
        """Test dataset loading failure."""
        with patch.object(dnd_mechanics_module, "load_dataset") as mock_load_dataset:
            mock_load_dataset.side_effect = Exception("Failed to load")

            with pytest.raises(
                RuntimeError, match="Failed to load D&D mechanics dataset"
            ):
                load_dnd_mechanics_dataset()


class TestPreprocessDataset:
    """Tests for preprocess_dataset function."""

    def test_preprocess_success(self):
        """Test successful dataset preprocessing."""
        mock_dataset = [
            {"instruction": "Query 1", "output": "Passage 1"},
            {"instruction": "Query 2", "output": "Passage 2"},
        ]

        result = preprocess_dataset(mock_dataset)

        assert len(result) == 2
        assert result[0] == {"query": "Query 1", "passage": "Passage 1"}
        assert result[1] == {"query": "Query 2", "passage": "Passage 2"}

    def test_preprocess_empty_dataset(self):
        """Test preprocessing empty dataset."""
        result = preprocess_dataset([])

        assert result == []

    def test_preprocess_missing_instruction(self):
        """Test preprocessing with missing instruction field."""
        mock_dataset = [
            {"instruction": "Query 1", "output": "Passage 1"},
            {"output": "Passage 2"},  # Missing instruction
            {"instruction": "Query 3", "output": "Passage 3"},
        ]

        result = preprocess_dataset(mock_dataset)

        # Should skip the invalid item
        assert len(result) == 2
        assert result[0]["query"] == "Query 1"
        assert result[1]["query"] == "Query 3"

    def test_preprocess_missing_output(self):
        """Test preprocessing with missing output field."""
        mock_dataset = [
            {"instruction": "Query 1", "output": "Passage 1"},
            {"instruction": "Query 2"},  # Missing output
        ]

        result = preprocess_dataset(mock_dataset)

        # Should skip the invalid item
        assert len(result) == 1
        assert result[0]["query"] == "Query 1"

    def test_preprocess_invalid_item(self):
        """Test preprocessing with invalid item type."""
        mock_dataset = [
            {"instruction": "Query 1", "output": "Passage 1"},
            "not a dict",
            {"instruction": "Query 3", "output": "Passage 3"},
        ]

        result = preprocess_dataset(mock_dataset)

        # Should skip the invalid item
        assert len(result) == 2

    def test_preprocess_none_dataset(self):
        """Test preprocessing with None dataset."""
        result = preprocess_dataset(None)

        assert result == []


class TestMain:
    """Tests for main function."""

    def test_main_success_with_upload(self):
        """Test successful main execution with upload."""
        mock_dataset = [
            {"instruction": "Query 1", "output": "Passage 1"},
            {"instruction": "Query 2", "output": "Passage 2"},
        ]

        mock_config_obj = MagicMock()
        mock_config_obj.hf_token = "test-token"

        with patch.object(
            dnd_mechanics_module, "load_dnd_mechanics_dataset"
        ) as mock_load, patch.object(
            dnd_mechanics_module, "load_config"
        ) as mock_config, patch.object(
            dnd_mechanics_module, "save_json_pairs"
        ) as mock_save, patch(
            "dnd_dm_copilot.utils.upload_to_huggingface"
        ) as mock_upload:
            mock_load.return_value = mock_dataset
            mock_config.return_value = mock_config_obj

            main(output_file="test.json", repo_id="test/repo", upload=True)

            # Verify load was called
            mock_load.assert_called_once()

            # Verify save was called with processed data
            assert mock_save.call_count == 1
            saved_data = mock_save.call_args[0][0]
            assert len(saved_data) == 2
            assert saved_data[0]["query"] == "Query 1"
            assert saved_data[0]["passage"] == "Passage 1"

            # Verify upload was called
            mock_upload.assert_called_once_with("test.json", "test/repo", "test-token")

    def test_main_success_without_upload(self):
        """Test successful main execution without upload."""
        mock_dataset = [{"instruction": "Query 1", "output": "Passage 1"}]

        with patch.object(
            dnd_mechanics_module, "load_dnd_mechanics_dataset"
        ) as mock_load, patch.object(
            dnd_mechanics_module, "save_json_pairs"
        ) as mock_save:
            mock_load.return_value = mock_dataset

            main(output_file="test.json", upload=False)

            mock_load.assert_called_once()
            mock_save.assert_called_once()

    def test_main_empty_dataset(self):
        """Test main with empty processed dataset."""
        with patch.object(
            dnd_mechanics_module, "load_dnd_mechanics_dataset"
        ) as mock_load:
            mock_load.return_value = []

            with pytest.raises(RuntimeError, match="No data to process"):
                main(output_file="test.json", upload=False)

    def test_main_upload_failure(self):
        """Test that upload failure doesn't crash main."""
        mock_dataset = [{"instruction": "Query 1", "output": "Passage 1"}]

        mock_config_obj = MagicMock()
        mock_config_obj.hf_token = "test-token"

        with patch.object(
            dnd_mechanics_module, "load_dnd_mechanics_dataset"
        ) as mock_load, patch.object(
            dnd_mechanics_module, "load_config"
        ) as mock_config, patch.object(
            dnd_mechanics_module, "save_json_pairs"
        ) as mock_save, patch(
            "dnd_dm_copilot.utils.upload_to_huggingface"
        ) as mock_upload:
            mock_load.return_value = mock_dataset
            mock_config.return_value = mock_config_obj
            mock_upload.side_effect = Exception("Upload failed")

            # Should not raise, just log warning
            main(output_file="test.json", repo_id="test/repo", upload=True)

            mock_save.assert_called_once()
            mock_upload.assert_called_once()

    def test_main_load_failure(self):
        """Test main handles load failure."""
        with patch.object(
            dnd_mechanics_module, "load_dnd_mechanics_dataset"
        ) as mock_load:
            mock_load.side_effect = RuntimeError("Load failed")

            with pytest.raises(RuntimeError):
                main(output_file="test.json", upload=False)
