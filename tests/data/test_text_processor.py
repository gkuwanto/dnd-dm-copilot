"""Tests for text/markdown processing module."""

from pathlib import Path
from unittest.mock import patch

from dnd_dm_copilot.data.text.process_texts import (
    TextDataProcessor,
    chunk_markdown,
    load_text_file,
)


class TestTextLoading:
    """Tests for text file loading."""

    def test_load_text_file(self, tmp_path: Path) -> None:
        """Test loading a simple text file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content\nLine 2", encoding="utf-8")

        result = load_text_file(str(test_file))

        assert "Test content" in result
        assert "Line 2" in result

    def test_load_text_file_handles_encoding(self, tmp_path: Path) -> None:
        """Test that UTF-8 encoding is handled correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Special chars: é, ñ, 中文", encoding="utf-8")

        result = load_text_file(str(test_file))

        assert "é" in result
        assert "ñ" in result
        assert "中文" in result


class TestMarkdownChunking:
    """Tests for markdown-specific chunking."""

    def test_chunk_markdown_by_headers(self) -> None:
        """Test that markdown is chunked by headers."""
        text = """# Chapter 1

Content for chapter 1.

## Section 1.1

More content here.

## Section 1.2

Even more content.

# Chapter 2

Different chapter."""

        chunks = chunk_markdown(text, chunk_size=100)

        assert len(chunks) > 0
        # Should preserve headers in chunks
        has_header = any(
            "Chapter" in chunk["text"] or "Section" in chunk["text"] for chunk in chunks
        )
        assert has_header

    def test_chunk_markdown_respects_size(self) -> None:
        """Test that markdown chunks respect size limits."""
        text = "# Header\n\n" + " ".join(["word"] * 500)
        chunks = chunk_markdown(text, chunk_size=100, min_chunk_size=20)

        for chunk in chunks:
            assert len(chunk["text"].split()) <= 100 or "Header" in chunk["text"]


class TestTextDataProcessor:
    """Tests for TextDataProcessor class."""

    def test_processor_inherits_from_base(self) -> None:
        """Test that TextDataProcessor inherits from BaseDataProcessor."""
        from dnd_dm_copilot.data.base_processor import BaseDataProcessor

        processor = TextDataProcessor(
            input_dir="data/raw/lore", output_file="data/processed/test.json"
        )
        assert isinstance(processor, BaseDataProcessor)

    def test_load_data_finds_text_files(self, tmp_path: Path) -> None:
        """Test that load_data finds text and markdown files."""
        text_dir = tmp_path / "texts"
        text_dir.mkdir()
        (text_dir / "notes1.txt").touch()
        (text_dir / "notes2.md").touch()
        (text_dir / "readme.pdf").touch()  # Should be ignored

        processor = TextDataProcessor(
            input_dir=str(text_dir), output_file="output.json"
        )

        text_files = processor.load_data()

        assert len(text_files) == 2
        assert all(f.endswith((".txt", ".md")) for f in text_files)

    def test_load_data_filters_extensions(self, tmp_path: Path) -> None:
        """Test that only specified extensions are loaded."""
        text_dir = tmp_path / "texts"
        text_dir.mkdir()
        (text_dir / "file.txt").touch()
        (text_dir / "file.md").touch()
        (text_dir / "file.rst").touch()

        processor = TextDataProcessor(
            input_dir=str(text_dir),
            output_file="output.json",
            extensions=[".txt"],  # Only .txt
        )

        text_files = processor.load_data()

        assert len(text_files) == 1
        assert text_files[0].endswith(".txt")

    def test_process_data_creates_passages(self, tmp_path: Path) -> None:
        """Test that process_data creates properly formatted passages."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content " * 200, encoding="utf-8")

        with patch(
            "dnd_dm_copilot.data.text.process_texts.load_text_file"
        ) as mock_load:
            mock_load.return_value = "Test content " * 200

            processor = TextDataProcessor(
                input_dir="data/raw/lore",
                output_file="output.json",
                chunk_size=100,
            )

            passages = processor.process_data([str(test_file)])

            assert len(passages) > 0
            for passage in passages:
                assert "query" in passage
                assert "passage" in passage
                assert "metadata" in passage
                assert passage["query"] == ""  # Empty for corpus-only
                assert len(passage["passage"]) > 0

    def test_process_data_handles_markdown(self, tmp_path: Path) -> None:
        """Test that markdown files are processed correctly."""
        test_file = tmp_path / "test.md"
        markdown_content = """# Chapter 1

Some content here.

## Section 1.1

More content."""
        test_file.write_text(markdown_content, encoding="utf-8")

        with patch(
            "dnd_dm_copilot.data.text.process_texts.load_text_file"
        ) as mock_load:
            mock_load.return_value = markdown_content

            processor = TextDataProcessor(
                input_dir="data/raw/lore", output_file="output.json"
            )

            passages = processor.process_data([str(test_file)])

            assert len(passages) > 0
            # Should have detected it's markdown
            assert any("Chapter" in p["passage"] for p in passages)

    def test_process_data_includes_metadata(self, tmp_path: Path) -> None:
        """Test that metadata includes source file information."""
        test_file = tmp_path / "notes.txt"
        test_file.write_text("Some text " * 100, encoding="utf-8")

        with patch(
            "dnd_dm_copilot.data.text.process_texts.load_text_file"
        ) as mock_load:
            mock_load.return_value = "Some text " * 100

            processor = TextDataProcessor(
                input_dir="data/raw/lore", output_file="output.json"
            )

            passages = processor.process_data([str(test_file)])

            for passage in passages:
                assert "source" in passage["metadata"]
                assert "notes.txt" in passage["metadata"]["source"]

    def test_full_pipeline_integration(self, tmp_path: Path) -> None:
        """Test the full pipeline from text files to JSON output."""
        text_dir = tmp_path / "texts"
        text_dir.mkdir()
        test_file = text_dir / "test.txt"
        test_file.write_text("Test content " * 200, encoding="utf-8")
        output_file = tmp_path / "output.json"

        with patch(
            "dnd_dm_copilot.data.text.process_texts.load_text_file"
        ) as mock_load:
            with patch(
                "dnd_dm_copilot.data.base_processor.save_json_pairs"
            ) as mock_save:
                mock_load.return_value = "Test content " * 200

                processor = TextDataProcessor(
                    input_dir=str(text_dir), output_file=str(output_file), upload=False
                )

                with patch("glob.glob", return_value=[str(test_file)]):
                    result = processor.process()

                assert len(result) > 0
                mock_save.assert_called_once()
