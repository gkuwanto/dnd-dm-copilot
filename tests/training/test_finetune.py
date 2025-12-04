"""Tests for training pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from dnd_dm_copilot.training.finetune import (
    load_dataset,
    split_dataset,
    create_model,
    prepare_training_data,
    prepare_ir_evaluator,
)


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_dataset_success(self, temp_json_file, sample_query_passage_pairs):
        """Test successful dataset loading."""
        data = load_dataset(str(temp_json_file))

        assert isinstance(data, list)
        assert len(data) == len(sample_query_passage_pairs)
        assert all("query" in item and "passage" in item for item in data)

    def test_load_dataset_file_not_found(self):
        """Test that non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset("/nonexistent/dataset.json")

    def test_load_dataset_invalid_format(self, temp_dir):
        """Test that invalid format raises error."""
        # Not a list
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text('{"query": "test"}')

        with pytest.raises(ValueError, match="Dataset must be a list"):
            load_dataset(str(invalid_file))

    def test_load_dataset_empty(self, temp_dir):
        """Test that empty dataset raises error."""
        empty_file = temp_dir / "empty.json"
        empty_file.write_text("[]")

        with pytest.raises(ValueError, match="Dataset is empty"):
            load_dataset(str(empty_file))

    def test_load_dataset_missing_fields(self, temp_dir):
        """Test that missing required fields raises error."""
        bad_file = temp_dir / "bad.json"
        bad_file.write_text('[{"query": "test"}]')  # Missing "passage"

        with pytest.raises(ValueError, match="Invalid data format"):
            load_dataset(str(bad_file))


class TestSplitDataset:
    """Tests for split_dataset function."""

    def test_split_dataset_success(self, sample_query_passage_pairs):
        """Test successful dataset splitting."""
        train, val, test = split_dataset(
            sample_query_passage_pairs, train_ratio=0.6, val_ratio=0.2, random_state=42
        )

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(sample_query_passage_pairs)

    def test_split_dataset_default_ratios(self, sample_query_passage_pairs):
        """Test splitting with default ratios (80/10/10)."""
        train, val, test = split_dataset(sample_query_passage_pairs)

        total = len(sample_query_passage_pairs)
        # With small datasets, exact ratios aren't always possible, so use larger tolerance
        assert len(train) / total == pytest.approx(0.8, abs=0.1)
        assert len(val) / total == pytest.approx(0.1, abs=0.1)
        assert len(test) / total == pytest.approx(0.1, abs=0.1)

    def test_split_dataset_invalid_ratios(self, sample_query_passage_pairs):
        """Test that invalid ratios raise error."""
        with pytest.raises(
            ValueError, match="train_ratio.*val_ratio must be less than 1.0"
        ):
            split_dataset(sample_query_passage_pairs, train_ratio=0.9, val_ratio=0.2)

    def test_split_dataset_reproducibility(self, sample_query_passage_pairs):
        """Test that same random_state produces same split."""
        train1, val1, test1 = split_dataset(sample_query_passage_pairs, random_state=42)
        train2, val2, test2 = split_dataset(sample_query_passage_pairs, random_state=42)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2


class TestCreateModel:
    """Tests for create_model function."""

    @patch("dnd_dm_copilot.training.finetune.models.Transformer")
    @patch("dnd_dm_copilot.training.finetune.models.Pooling")
    @patch("dnd_dm_copilot.training.finetune.SentenceTransformer")
    def test_create_model_default(self, mock_st, mock_pool, mock_transformer):
        """Test model creation with default parameters."""
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.get_word_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_transformer_instance

        model = create_model()

        assert model is not None
        mock_transformer.assert_called_once()

    @patch("dnd_dm_copilot.training.finetune.models.Transformer")
    @patch("dnd_dm_copilot.training.finetune.models.Pooling")
    @patch("dnd_dm_copilot.training.finetune.SentenceTransformer")
    def test_create_model_custom_name(self, mock_st, mock_pool, mock_transformer):
        """Test model creation with custom model name."""
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.get_word_embedding_dimension.return_value = 768
        mock_transformer.return_value = mock_transformer_instance

        custom_model = "bert-base-uncased"
        _model = create_model(model_name=custom_model)

        mock_transformer.assert_called_once_with(custom_model)


class TestPrepareTrainingData:
    """Tests for prepare_training_data function."""

    @patch("dnd_dm_copilot.training.finetune.DataLoader")
    @patch("dnd_dm_copilot.training.finetune.InputExample")
    def test_prepare_training_data_success(
        self, mock_example, mock_dataloader, sample_query_passage_pairs
    ):
        """Test successful training data preparation."""
        mock_dataloader_instance = MagicMock()
        mock_dataloader.return_value = mock_dataloader_instance

        dataloader = prepare_training_data(sample_query_passage_pairs, batch_size=2)

        assert dataloader is mock_dataloader_instance
        assert mock_example.call_count == len(sample_query_passage_pairs)

    @patch("dnd_dm_copilot.training.finetune.DataLoader")
    def test_prepare_training_data_empty(self, mock_dataloader):
        """Test preparation with empty training data."""
        mock_dataloader_instance = MagicMock()
        mock_dataloader.return_value = mock_dataloader_instance

        dataloader = prepare_training_data([], batch_size=32)

        assert dataloader is mock_dataloader_instance


class TestPrepareIREvaluator:
    """Tests for prepare_ir_evaluator function."""

    @patch("dnd_dm_copilot.training.finetune.InformationRetrievalEvaluator")
    def test_prepare_ir_evaluator_success(
        self, mock_evaluator, sample_query_passage_pairs
    ):
        """Test successful IR evaluator preparation."""
        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance

        evaluator = prepare_ir_evaluator(sample_query_passage_pairs, name="test")

        assert evaluator is mock_evaluator_instance
        # Check that evaluator was called
        mock_evaluator.assert_called_once()
        # Get the positional arguments
        call_args = mock_evaluator.call_args[0]
        assert len(call_args) >= 3
        # Verify queries, corpus, and relevant_docs were passed
        queries_dict = call_args[0]
        corpus_dict = call_args[1]
        relevant_docs_dict = call_args[2]
        assert isinstance(queries_dict, dict)
        assert isinstance(corpus_dict, dict)
        assert isinstance(relevant_docs_dict, dict)
        assert len(queries_dict) > 0
        assert len(corpus_dict) > 0

    @patch("dnd_dm_copilot.training.finetune.InformationRetrievalEvaluator")
    def test_prepare_ir_evaluator_duplicate_queries(self, mock_evaluator):
        """Test IR evaluator with duplicate queries."""
        pairs = [
            {"query": "Same query", "passage": "Passage 1"},
            {"query": "Same query", "passage": "Passage 2"},
        ]

        mock_evaluator_instance = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance

        _evaluator = prepare_ir_evaluator(pairs)

        mock_evaluator.assert_called_once()
        call_args = mock_evaluator.call_args[0]
        queries_dict = call_args[0]
        corpus_dict = call_args[1]
        # Should have only 1 unique query
        assert len(queries_dict) == 1
        # Should have 2 passages
        assert len(corpus_dict) == 2
