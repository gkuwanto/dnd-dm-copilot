"""Shared pytest fixtures for all tests."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def sample_query_passage_pairs() -> List[Dict[str, str]]:
    """
    Fixture providing sample query-passage pairs for testing.

    Returns:
        List of valid query-passage dictionaries
    """
    return [
        {
            "query": "How does Divine Smite work?",
            "passage": (
                "Divine Smite is a Paladin class feature that allows you to "
                "expend a spell slot..."
            ),
        },
        {
            "query": "What is the range of Fireball?",
            "passage": (
                "Fireball has a range of 150 feet and creates a sphere of "
                "20-foot radius..."
            ),
        },
        {
            "query": "How does advantage work?",
            "passage": (
                "When you have advantage, you roll two d20s and use the "
                "higher result..."
            ),
        },
        {
            "query": "What are hit dice?",
            "passage": (
                "Hit dice are used to determine how many hit points you "
                "recover when resting..."
            ),
        },
        {
            "query": "How do saving throws work?",
            "passage": (
                "A saving throw represents an attempt to resist a spell, "
                "trap, poison, disease..."
            ),
        },
        {
            "query": "What is armor class?",
            "passage": (
                "Armor Class (AC) is a measure of how difficult it is to "
                "hit a creature..."
            ),
        },
        {
            "query": "How do spell slots work?",
            "passage": (
                "Spell slots represent the energy you have to cast spells "
                "of particular levels..."
            ),
        },
        {
            "query": "What is initiative?",
            "passage": "Initiative determines the order of turns during combat...",
        },
        {
            "query": "How does concentration work?",
            "passage": (
                "Some spells require concentration to maintain their effects..."
            ),
        },
        {
            "query": "What are ability scores?",
            "passage": "Ability scores define your character's basic capabilities...",
        },
        {
            "query": "How does inspiration work?",
            "passage": (
                "Inspiration is a reward for roleplay that lets you reroll a d20..."
            ),
        },
        {
            "query": "What is proficiency bonus?",
            "passage": (
                "Your proficiency bonus is added to rolls you are proficient in..."
            ),
        },
    ]


@pytest.fixture
def temp_json_file(sample_query_passage_pairs) -> Path:
    """
    Fixture providing a temporary JSON file with sample data.

    Returns:
        Path to the temporary JSON file
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_query_passage_pairs, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_dir() -> Path:
    """
    Fixture providing a temporary directory.

    Returns:
        Path to the temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config() -> MagicMock:
    """
    Fixture providing a mock configuration object.

    Returns:
        Mock Config object with common attributes
    """
    config = MagicMock()
    config.hf_token = "test-token"
    config.deepseek_api_key = "test-deepseek-key"
    config.batch_size = 32
    config.num_epochs = 2
    config.train_ratio = 0.8
    config.val_ratio = 0.1
    return config


@pytest.fixture
def mock_huggingface() -> None:
    """Mock HuggingFace operations to prevent actual API calls."""
    with patch("dnd_dm_copilot.utils.huggingface.upload_file") as mock_upload:
        mock_upload.return_value = None
        yield mock_upload


@pytest.fixture
def mock_deepseek_client() -> MagicMock:
    """
    Fixture providing a mock DeepSeek LLM client.

    Returns:
        Mock DeepSeek client
    """
    with patch("dnd_dm_copilot.model.llm_client.openai.OpenAI") as mock_client:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Mock LLM response"))]
        )
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_wandb() -> None:
    """Mock Weights & Biases operations to prevent actual API calls."""
    with (
        patch("wandb.init"),
        patch("wandb.log"),
        patch("wandb.finish"),
        patch("wandb.Artifact"),
        patch("wandb.run"),
    ):
        yield


@pytest.fixture
def mock_sentence_transformer() -> None:
    """Mock SentenceTransformer model operations."""
    with patch("sentence_transformers.SentenceTransformer") as mock_model:
        mock_instance = MagicMock()
        # Mock encode method to return dummy embeddings
        mock_instance.encode.return_value = [[0.1] * 384 for _ in range(5)]
        mock_instance.fit.return_value = None
        mock_model.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_dataset_load() -> None:
    """Mock datasets.load_dataset to prevent actual HuggingFace downloads."""
    with patch("datasets.load_dataset") as mock_load:
        # Return a mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(
            side_effect=lambda i: {
                "instruction": f"Query {i}",
                "output": f"Passage {i}",
            }
        )
        mock_load.return_value = mock_dataset
        yield mock_load
