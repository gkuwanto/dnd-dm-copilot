"""Tests for file I/O utilities."""

import pytest

from dnd_dm_copilot.utils.file_io import (
    save_json_pairs,
    load_json_pairs,
    validate_file_exists,
    ensure_directory_exists,
    FileIOError,
)


class TestSaveJsonPairs:
    """Tests for save_json_pairs function."""

    def test_save_json_pairs_success(self, temp_dir, sample_query_passage_pairs):
        """Test successful saving of JSON pairs."""
        output_file = temp_dir / "test_pairs.json"
        save_json_pairs(sample_query_passage_pairs, str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_save_json_pairs_creates_directory(
        self, temp_dir, sample_query_passage_pairs
    ):
        """Test that save_json_pairs creates parent directories."""
        output_file = temp_dir / "subdir" / "nested" / "pairs.json"
        save_json_pairs(sample_query_passage_pairs, str(output_file))

        assert output_file.exists()

    def test_save_json_pairs_invalid_input(self, temp_dir):
        """Test that save_json_pairs rejects non-list input."""
        output_file = temp_dir / "test.json"
        with pytest.raises(ValueError, match="pairs must be a list"):
            save_json_pairs({"query": "test"}, str(output_file))

    def test_save_json_pairs_with_indent(self, temp_dir, sample_query_passage_pairs):
        """Test saving with custom indent."""
        output_file = temp_dir / "test_pairs.json"
        save_json_pairs(sample_query_passage_pairs, str(output_file), indent=4)

        with open(output_file, "r") as f:
            content = f.read()
            # Check that indentation is applied
            assert "    " in content


class TestLoadJsonPairs:
    """Tests for load_json_pairs function."""

    def test_load_json_pairs_success(self, temp_json_file, sample_query_passage_pairs):
        """Test successful loading of JSON pairs."""
        loaded = load_json_pairs(str(temp_json_file))

        assert isinstance(loaded, list)
        assert len(loaded) == len(sample_query_passage_pairs)
        assert loaded[0]["query"] == sample_query_passage_pairs[0]["query"]

    def test_load_json_pairs_file_not_found(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileIOError, match="JSON file not found"):
            load_json_pairs("/nonexistent/path/file.json")

    def test_load_json_pairs_invalid_json(self, temp_dir):
        """Test that loading invalid JSON raises error."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{ invalid json }")

        with pytest.raises(FileIOError, match="Failed to parse JSON"):
            load_json_pairs(str(invalid_file))

    def test_load_json_pairs_not_list(self, temp_dir):
        """Test that non-list JSON raises error."""
        not_list_file = temp_dir / "not_list.json"
        not_list_file.write_text('{"query": "test"}')

        with pytest.raises(ValueError, match="JSON file must contain a list"):
            load_json_pairs(str(not_list_file))


class TestValidateFileExists:
    """Tests for validate_file_exists function."""

    def test_validate_file_exists_success(self, temp_json_file):
        """Test successful file validation."""
        # Should not raise
        validate_file_exists(str(temp_json_file))

    def test_validate_file_exists_not_found(self):
        """Test that non-existent file raises error."""
        with pytest.raises(FileIOError, match="File file not found"):
            validate_file_exists("/nonexistent/file.txt")

    def test_validate_file_exists_is_directory(self, temp_dir):
        """Test that directory raises error."""
        with pytest.raises(FileIOError, match="not a file"):
            validate_file_exists(str(temp_dir))

    def test_validate_file_exists_custom_file_type(self, temp_json_file):
        """Test that custom file type is used in error message."""
        # Should not raise
        validate_file_exists(str(temp_json_file), file_type="custom type")


class TestEnsureDirectoryExists:
    """Tests for ensure_directory_exists function."""

    def test_ensure_directory_exists_creates(self, temp_dir):
        """Test that function creates directory."""
        new_dir = temp_dir / "new" / "nested" / "dir"
        ensure_directory_exists(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_directory_exists_already_exists(self, temp_dir):
        """Test that function handles existing directory."""
        # Should not raise
        ensure_directory_exists(str(temp_dir))
        assert temp_dir.exists()

    def test_ensure_directory_exists_path_is_file(self, temp_json_file):
        """Test that error is raised if path is a file."""
        with pytest.raises(FileIOError, match="not a directory"):
            ensure_directory_exists(str(temp_json_file))
