"""PDF processing module for extracting and chunking D&D handbook content."""

import glob
import re
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader

from dnd_dm_copilot.data.base_processor import BaseDataProcessor
from dnd_dm_copilot.utils import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """Extracts text from PDF files."""

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text with page separators
        """
        return extract_text_from_pdf(pdf_path)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pypdf.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text with page separators

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF reading fails
    """  # noqa: E501
    logger.info(f"Extracting text from {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        pages_text = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                pages_text.append(text)
                logger.debug(f"Extracted {len(text)} characters from page {i + 1}")

        full_text = "\n\n".join(pages_text)
        logger.info(
            f"Successfully extracted {len(full_text)} characters "
            f"from {len(pages_text)} pages"
        )
        return full_text

    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        raise


def detect_headers(text: str) -> List[tuple]:
    """
    Detect section headers in D&D handbook text.

    Looks for patterns like:
    - "Chapter X: Title"
    - "ALL CAPS HEADERS"
    - Lines ending with specific patterns

    Args:
        text: Text to analyze

    Returns:
        List of (position, header_text) tuples
    """
    headers = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Chapter headers
        if re.match(r"^Chapter \d+:", line_stripped, re.IGNORECASE):
            headers.append((i, line_stripped))

        # All caps headers (at least 3 words, all caps)
        elif re.match(r"^[A-Z][A-Z\s]{10,}$", line_stripped):
            words = line_stripped.split()
            if len(words) >= 2:  # At least 2 words
                headers.append((i, line_stripped))

    return headers


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
    detect_headers: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunk text into passages suitable for RAG.

    Args:
        text: Text to chunk
        chunk_size: Maximum number of words per chunk
        chunk_overlap: Number of words to overlap between chunks
        min_chunk_size: Minimum number of words per chunk
        detect_headers: Whether to detect and preserve section headers

    Returns:
        List of chunk dictionaries with 'text' and optional metadata
    """
    logger.debug(
        f"Chunking text: {len(text)} chars, "
        f"chunk_size={chunk_size}, overlap={chunk_overlap}"
    )

    # Split by sentences first for better boundary awareness
    # If no sentence boundaries, fall back to splitting all words
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: List[Dict[str, Any]] = []

    # If no sentence boundaries found, treat as single sentence
    if len(sentences) == 1 and len(sentences[0].split()) > chunk_size:
        # Split into word-based chunks
        all_words = text.split()
        i = 0
        while i < len(all_words):
            chunk_words = all_words[i : i + chunk_size]
            if len(chunk_words) >= min_chunk_size or i == 0 or not chunks:
                chunks.append({"text": " ".join(chunk_words)})
            # Move forward, accounting for overlap
            next_i = i + chunk_size - chunk_overlap
            if next_i <= i:  # Prevent infinite loop
                next_i = i + 1
            i = next_i
        logger.info(f"Created {len(chunks)} chunks from text (word-based)")
        return chunks

    # Sentence-based chunking
    current_chunk: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        # If adding this sentence exceeds chunk size, save current chunk
        if current_word_count + sentence_word_count > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(current_chunk) >= min_chunk_size or not chunks:
                chunks.append({"text": chunk_text})

            # Start new chunk with overlap
            overlap_word_count = min(chunk_overlap, len(current_chunk))
            current_chunk = current_chunk[-overlap_word_count:] + sentence_words
            current_word_count = len(current_chunk)
        else:
            current_chunk.extend(sentence_words)
            current_word_count += sentence_word_count

    # Add final chunk
    if current_chunk and (len(current_chunk) >= min_chunk_size or not chunks):
        chunks.append({"text": " ".join(current_chunk)})

    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks


class PDFDataProcessor(BaseDataProcessor):
    """
    Processes PDF files into chunked passages for RAG.

    This processor:
    1. Loads PDF files from a directory
    2. Extracts text from each PDF
    3. Chunks text into passages
    4. Outputs standardized JSON format
    """

    def __init__(
        self,
        input_dir: str,
        output_file: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        repo_id: str = "",
        upload: bool = True,
        config: Any = None,
    ):
        """
        Initialize PDF processor.

        Args:
            input_dir: Directory containing PDF files
            output_file: Path to output JSON file
            chunk_size: Maximum words per chunk
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum words per chunk
            repo_id: HuggingFace repo ID for upload
            upload: Whether to upload to HuggingFace
            config: Optional Config object
        """
        super().__init__(
            output_file=output_file, repo_id=repo_id, upload=upload, config=config
        )
        self.input_dir = input_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.extractor = PDFExtractor()

    def load_data(self) -> List[str]:
        """
        Load all PDF files from input directory.

        Returns:
            List of PDF file paths

        Raises:
            FileNotFoundError: If input directory doesn't exist
        """
        logger.info(f"Loading PDF files from {self.input_dir}")

        if not Path(self.input_dir).exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        pdf_files = glob.glob(f"{self.input_dir}/*.pdf")
        logger.info(f"Found {len(pdf_files)} PDF files")

        return pdf_files

    def process_data(self, pdf_files: List[str]) -> List[Dict[str, str]]:
        """
        Extract and chunk all PDFs into passages.

        Args:
            pdf_files: List of PDF file paths

        Returns:
            List of passage dictionaries with standardized format
        """
        logger.info(f"Processing {len(pdf_files)} PDF files")

        all_passages = []

        for pdf_path in pdf_files:
            try:
                # Extract text
                text = extract_text_from_pdf(pdf_path)

                # Chunk text
                chunks = chunk_text(
                    text,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    min_chunk_size=self.min_chunk_size,
                )

                # Convert to standardized format
                for chunk in chunks:
                    all_passages.append(
                        {
                            "query": "",  # Empty for corpus-only data
                            "passage": chunk["text"],
                            "metadata": {
                                "source": Path(pdf_path).name,
                                "source_path": pdf_path,
                            },
                        }
                    )

                logger.info(f"Processed {pdf_path}: {len(chunks)} chunks")

            except Exception as e:
                logger.warning(f"Failed to process {pdf_path}: {e}")
                continue

        logger.info(f"Total passages created: {len(all_passages)}")
        return all_passages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process D&D PDF handbooks")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/pdfs",
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/mechanics_corpus.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Maximum words per chunk"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=50, help="Words overlap between chunks"
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="Skip HuggingFace upload"
    )

    args = parser.parse_args()

    processor = PDFDataProcessor(
        input_dir=args.input_dir,
        output_file=args.output_file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        upload=not args.no_upload,
    )

    passages = processor.process()
    print(f"Successfully processed {len(passages)} passages to {args.output_file}")
