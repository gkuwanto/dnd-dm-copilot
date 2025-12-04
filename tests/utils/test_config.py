"""Tests for configuration management utilities."""

import os
from unittest.mock import patch

import pytest

from dnd_dm_copilot.utils.config import (
    Config,
    load_config,
    validate_environment,
    ConfigError,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_initialization_defaults(self):
        """Test that Config initializes with default values."""
        config = Config()

        assert config.batch_size == 64
        assert config.num_epochs == 10
        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1

    def test_config_custom_values(self):
        """Test Config with custom values."""
        config = Config(batch_size=128, num_epochs=20, hf_token="test-token")

        assert config.batch_size == 128
        assert config.num_epochs == 20
        assert config.hf_token == "test-token"

    def test_config_invalid_train_ratio(self):
        """Test that invalid train_ratio raises error."""
        with pytest.raises(ConfigError, match="train_ratio must be between 0 and 1"):
            Config(train_ratio=1.5)

    def test_config_invalid_val_ratio(self):
        """Test that invalid val_ratio raises error."""
        with pytest.raises(ConfigError, match="val_ratio must be between 0 and 1"):
            Config(val_ratio=1.5)

    def test_config_train_val_ratio_too_large(self):
        """Test that train_ratio + val_ratio > 1 raises error."""
        with pytest.raises(ConfigError, match="train_ratio.*val_ratio must be <= 1"):
            Config(train_ratio=0.9, val_ratio=0.2)

    def test_config_invalid_batch_size(self):
        """Test that invalid batch_size raises error."""
        with pytest.raises(ConfigError, match="batch_size must be positive"):
            Config(batch_size=0)

    def test_config_invalid_num_epochs(self):
        """Test that invalid num_epochs raises error."""
        with pytest.raises(ConfigError, match="num_epochs must be positive"):
            Config(num_epochs=-1)


class TestLoadConfig:
    """Tests for load_config function."""

    @patch.dict(os.environ, {"HF_TOKEN": "test-token"})
    def test_load_config_from_env(self):
        """Test loading config from environment variables."""
        config = load_config()

        assert config.hf_token == "test-token"

    def test_load_config_no_token(self):
        """Test loading config with no HF_TOKEN in environment."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config()
            assert config.hf_token == ""

    def test_load_config_from_file(self, temp_dir):
        """Test loading config from JSON file."""
        import json

        config_file = temp_dir / "config.json"
        config_data = {"batch_size": 128, "num_epochs": 20}
        config_file.write_text(json.dumps(config_data))

        config = load_config(str(config_file))

        assert config.batch_size == 128
        assert config.num_epochs == 20

    def test_load_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(ConfigError, match="Failed to load config file"):
            load_config("/nonexistent/config.json")

    def test_load_config_invalid_json(self, temp_dir):
        """Test that invalid JSON config raises error."""
        config_file = temp_dir / "config.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(ConfigError, match="Failed to load config file"):
            load_config(str(config_file))


class TestValidateEnvironment:
    """Tests for validate_environment function."""

    @patch.dict(os.environ, {"HF_TOKEN": "test-token"})
    def test_validate_environment_success(self):
        """Test successful environment validation."""
        env_vars = validate_environment()

        assert "HF_TOKEN" in env_vars
        assert env_vars["HF_TOKEN"] == "test-token"

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_environment_missing_token(self):
        """Test that missing HF_TOKEN raises error."""
        with pytest.raises(ConfigError, match="Required environment variables not set"):
            validate_environment()

    @patch.dict(
        os.environ,
        {
            "HF_TOKEN": "test-token",
            "DEEPSEEK_API_KEY": "ds-key",
            "WANDB_API_KEY": "wb-key",
        },
    )
    def test_validate_environment_all_tokens(self):
        """Test validation with all optional tokens set."""
        env_vars = validate_environment()

        assert env_vars["HF_TOKEN"] == "test-token"
        assert env_vars["DEEPSEEK_API_KEY"] == "ds-key"
        assert env_vars["WANDB_API_KEY"] == "wb-key"
