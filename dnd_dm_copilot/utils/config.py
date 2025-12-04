"""Configuration management for the DnD DM Copilot project."""

import os
from dataclasses import dataclass
from typing import Dict, Optional


class ConfigError(Exception):
    """Custom exception for configuration errors."""

    pass


@dataclass
class Config:
    """Configuration for the DnD DM Copilot project."""

    # API Keys and tokens
    hf_token: str = ""
    deepseek_api_key: str = ""
    wandb_api_key: str = ""

    # Data Processing Configuration
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    random_state: int = 42

    # Training Configuration
    batch_size: int = 64
    num_epochs: int = 10
    evaluation_steps: int = 50
    learning_rate: float = 2e-5
    warmup_steps: int = 100

    # Model Configuration
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir: str = "models/sbert/"

    # File Paths
    data_dir: str = "data/"
    log_dir: str = "logs/"

    # Feature Flags
    disable_wandb: bool = False
    use_cuda: bool = True

    # Hardcoded values made configurable
    crd3_c_value: int = 2
    concurrent_requests: int = 10
    max_batches: Optional[int] = None
    fireball_batch_size: int = 1000
    reddit_file_count: int = 8

    # Quality Thresholds
    quality_threshold: float = 0.6
    dice_roll_threshold: float = 0.7
    filler_response_threshold: float = 0.6

    # Quality Scoring Weights
    meaningful_content_weight: float = 2.0
    dm_narration_weight: float = 1.5
    dialogue_variety_weight: float = 1.0
    length_weight: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0 <= self.train_ratio <= 1:
            raise ConfigError(
                f"train_ratio must be between 0 and 1, got {self.train_ratio}"
            )

        if not 0 <= self.val_ratio <= 1:
            raise ConfigError(
                f"val_ratio must be between 0 and 1, got {self.val_ratio}"
            )

        if self.train_ratio + self.val_ratio > 1:
            raise ConfigError(
                f"train_ratio + val_ratio must be <= 1, got {self.train_ratio + self.val_ratio}"
            )

        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_epochs <= 0:
            raise ConfigError(f"num_epochs must be positive, got {self.num_epochs}")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from environment variables and optional config file.

    Args:
        config_path: Optional path to a JSON config file

    Returns:
        Config instance with values from environment and config file

    Raises:
        ConfigError: If required environment variables are missing
    """
    # Load from environment variables
    config_dict: Dict[str, str] = {
        "hf_token": os.getenv("HF_TOKEN", ""),
        "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "wandb_api_key": os.getenv("WANDB_API_KEY", ""),
    }

    # Override with config file if provided
    if config_path:
        import json

        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
                config_dict.update(file_config)
        except (IOError, json.JSONDecodeError) as e:
            raise ConfigError(f"Failed to load config file '{config_path}': {e}")

    return Config(**config_dict)


def validate_environment() -> Dict[str, str]:
    """
    Validate that required environment variables are set.

    Returns:
        Dictionary of environment variables

    Raises:
        ConfigError: If required environment variables are missing
    """
    required_vars = ["HF_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ConfigError(
            f"Required environment variables not set: {', '.join(missing_vars)}. "
            "Please set them in your .env file or environment."
        )

    return {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", ""),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
    }
