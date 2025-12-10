"""Tests for PDF processing module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from dnd_dm_copilot.data.pdf.process_pdfs import (
    PDFDataProcessor,
    chunk_text,
    extract_text_from_pdf,
)


class TestPDFExtractor:
    """Tests for PDFExtractor class."""

    def test_extract_text_from_simple_pdf(self, tmp_path: Path) -> None:
        """Test extracting text from a simple PDF."""
        # This will be implemented when we create PDFExtractor
        with patch("dnd_dm_copilot.data.pdf.process_pdfs.PdfReader") as mock_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = (
                "Chapter 1: Combat\nInitiative rules..."
            )
            mock_reader.return_value.pages = [mock_page]

            result = extract_text_from_pdf("dummy.pdf")

            assert "Chapter 1: Combat" in result
            assert "Initiative" in result

    def test_extract_text_handles_empty_pages(self) -> None:
        """Test that empty pages are handled gracefully."""
        with patch("dnd_dm_copilot.data.pdf.process_pdfs.PdfReader") as mock_reader:
            mock_page1 = MagicMock()
            mock_page1.extract_text.return_value = ""
            mock_page2 = MagicMock()
            mock_page2.extract_text.return_value = "Some content"
            mock_reader.return_value.pages = [mock_page1, mock_page2]

            result = extract_text_from_pdf("dummy.pdf")

            assert "Some content" in result
            # Empty page should be skipped, so no double separator


class TestChunking:
    """Tests for text chunking functionality."""

    def test_chunk_respects_size_limit(self) -> None:
        """Test that chunks don't exceed maximum size."""
        text = " ".join(["word"] * 1000)  # 1000 words
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)

        for chunk in chunks:
            assert len(chunk["text"].split()) <= 100

    def test_chunk_overlap(self) -> None:
        """Test that chunks have proper overlap."""
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10, min_chunk_size=20)

        assert len(chunks) > 1
        # Check that consecutive chunks share some words
        for i in range(len(chunks) - 1):
            chunk1_words = chunks[i]["text"].split()
            chunk2_words = chunks[i + 1]["text"].split()
            # Last words of chunk1 should appear in chunk2
            overlap = set(chunk1_words[-15:]) & set(chunk2_words[:15])
            assert len(overlap) > 0

    def test_chunk_minimum_size(self) -> None:
        """Test that very small chunks are filtered out."""
        text = "Short text. " * 5
        chunks = chunk_text(text, chunk_size=100, min_chunk_size=20)

        for chunk in chunks:
            assert len(chunk["text"].split()) >= 20 or chunk == chunks[-1]

    def test_section_aware_chunking(self) -> None:
        """Test that section headers are preserved in chunks."""
        text = """Chapter 1: Combat

INITIATIVE
At the beginning of combat, roll initiative.

ATTACK ROLLS
When you make an attack, roll a d20."""

        chunks = chunk_text(text, chunk_size=50, detect_headers=True)

        # Should detect headers and try to keep sections together
        assert len(chunks) >= 1
        # At least one chunk should contain a header
        has_header = any(
            "INITIATIVE" in chunk["text"] or "ATTACK ROLLS" in chunk["text"]
            for chunk in chunks
        )
        assert has_header


class TestPDFDataProcessor:
    """Tests for PDFDataProcessor class."""

    def test_processor_inherits_from_base(self) -> None:
        """Test that PDFDataProcessor inherits from BaseDataProcessor."""
        from dnd_dm_copilot.data.base_processor import BaseDataProcessor

        processor = PDFDataProcessor(
            input_dir="data/raw/pdfs", output_file="data/processed/test.json"
        )
        assert isinstance(processor, BaseDataProcessor)

    def test_load_data_finds_pdf_files(self, tmp_path: Path) -> None:
        """Test that load_data finds all PDF files in directory."""
        # Create test PDF files
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "test1.pdf").touch()
        (pdf_dir / "test2.pdf").touch()
        (pdf_dir / "test.txt").touch()  # Should be ignored

        processor = PDFDataProcessor(input_dir=str(pdf_dir), output_file="output.json")

        pdf_files = processor.load_data()

        assert len(pdf_files) == 2
        assert all(f.endswith(".pdf") for f in pdf_files)

    def test_process_data_creates_passages(self) -> None:
        """Test that process_data creates properly formatted passages."""
        with patch(
            "dnd_dm_copilot.data.pdf.process_pdfs.extract_text_from_pdf"
        ) as mock_extract:
            mock_extract.return_value = "Chapter 1: Combat\n\n" + " ".join(
                ["word"] * 600
            )

            processor = PDFDataProcessor(
                input_dir="data/raw/pdfs", output_file="output.json", chunk_size=100
            )

            passages = processor.process_data(["test.pdf"])

            assert len(passages) > 0
            for passage in passages:
                assert "query" in passage
                assert "passage" in passage
                assert "metadata" in passage
                assert passage["query"] == ""  # No queries for corpus-only data
                assert len(passage["passage"]) > 0

    def test_process_data_includes_metadata(self) -> None:
        """Test that metadata includes source file information."""
        with patch(
            "dnd_dm_copilot.data.pdf.process_pdfs.extract_text_from_pdf"
        ) as mock_extract:
            mock_extract.return_value = "Some text " * 100

            processor = PDFDataProcessor(
                input_dir="data/raw/pdfs", output_file="output.json"
            )

            passages = processor.process_data(["handbook.pdf"])

            for passage in passages:
                assert "source" in passage["metadata"]
                assert "handbook.pdf" in passage["metadata"]["source"]

    def test_full_pipeline_integration(self, tmp_path: Path) -> None:
        """Test the full pipeline from PDF to JSON output."""
        # Create test directory and PDF
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        output_file = tmp_path / "output.json"

        with patch(
            "dnd_dm_copilot.data.pdf.process_pdfs.extract_text_from_pdf"
        ) as mock_extract:
            with patch(
                "dnd_dm_copilot.data.base_processor.save_json_pairs"
            ) as mock_save:
                mock_extract.return_value = "Test content " * 200

                processor = PDFDataProcessor(
                    input_dir=str(pdf_dir), output_file=str(output_file), upload=False
                )

                # Mock glob to return a test file
                with patch("glob.glob", return_value=[str(pdf_dir / "test.pdf")]):
                    result = processor.process()

                assert len(result) > 0
                mock_save.assert_called_once()
