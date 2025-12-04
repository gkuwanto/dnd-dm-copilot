"""File I/O utilities for handling JSON and file operations."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


class FileIOError(Exception):
    """Custom exception for file I/O errors."""

    pass


def save_json_pairs(pairs: List[Dict[str, Any]], filepath: str, **kwargs: Any) -> None:
    """
    Save a list of dictionaries to a JSON file.

    Args:
        pairs: List of dictionaries to save
        filepath: Path where the JSON file will be saved
        **kwargs: Additional arguments to pass to json.dump (e.g., indent, ensure_ascii)

    Raises:
        FileIOError: If the file cannot be written
        ValueError: If pairs is not a list
    """
    if not isinstance(pairs, list):
        raise ValueError(f"pairs must be a list, got {type(pairs).__name__}")

    # Ensure parent directory exists
    parent_dir = Path(filepath).parent
    if parent_dir != Path("."):
        ensure_directory_exists(str(parent_dir))

    # Set default kwargs
    default_kwargs: Dict[str, Any] = {"indent": 2, "ensure_ascii": False}
    default_kwargs.update(kwargs)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(pairs, f, **default_kwargs)
    except (IOError, OSError) as e:
        raise FileIOError(f"Failed to write JSON file '{filepath}': {e}")
    except TypeError as e:
        raise FileIOError(f"Failed to serialize data to JSON: {e}")


def load_json_pairs(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a list of dictionaries from a JSON file.

    Args:
        filepath: Path to the JSON file to load

    Returns:
        List of dictionaries loaded from the file

    Raises:
        FileIOError: If the file cannot be read or parsed
        ValueError: If the JSON does not contain a list
    """
    validate_file_exists(filepath, "JSON")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise FileIOError(f"Failed to parse JSON file '{filepath}': {e}")
    except (IOError, OSError) as e:
        raise FileIOError(f"Failed to read JSON file '{filepath}': {e}")

    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list, got {type(data).__name__}")

    return data


def validate_file_exists(filepath: str, file_type: str = "File") -> None:
    """
    Validate that a file exists and is readable.

    Args:
        filepath: Path to the file to validate
        file_type: Type of file for error message (e.g., "JSON", "CSV")

    Raises:
        FileIOError: If the file doesn't exist or is not readable
    """
    path = Path(filepath)

    if not path.exists():
        raise FileIOError(f"{file_type} file not found: '{filepath}'")

    if not path.is_file():
        raise FileIOError(f"Path exists but is not a file: '{filepath}'")

    if not os.access(filepath, os.R_OK):
        raise FileIOError(f"{file_type} file is not readable: '{filepath}'")


def ensure_directory_exists(dirpath: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        dirpath: Path to the directory

    Raises:
        FileIOError: If the directory cannot be created or exists as a file
    """
    path = Path(dirpath)

    if path.exists():
        if not path.is_dir():
            raise FileIOError(f"Path exists but is not a directory: '{dirpath}'")
        return

    try:
        path.mkdir(parents=True, exist_ok=True)
    except (IOError, OSError) as e:
        raise FileIOError(f"Failed to create directory '{dirpath}': {e}")
