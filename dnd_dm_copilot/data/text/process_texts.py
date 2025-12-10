"""Text and Markdown processing module for lore content chunking."""

import glob
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dnd_dm_copilot.data.base_processor import BaseDataProcessor
from dnd_dm_copilot.data.pdf.process_pdfs import chunk_text
from dnd_dm_copilot.utils import get_logger

logger = get_logger(__name__)


def load_text_file(file_path: str) -> str:
    """
    Load text from a file with UTF-8 encoding.

    Args:
        file_path: Path to the text file

    Returns:
        File content as string

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file reading fails
    """
    logger.info(f"Loading text from {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"Loaded {len(content)} characters from {file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        raise


def detect_markdown_headers(text: str) -> List[tuple]:
    """
    Detect markdown headers in text.

    Args:
        text: Text to analyze

    Returns:
        List of (line_number, header_level, header_text) tuples
    """
    headers = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        # ATX-style headers (# Header)
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            level = len(match.group(1))
            text_content = match.group(2)
            headers.append((i, level, text_content))

    return headers


def chunk_markdown(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Chunk markdown text with header awareness.

    Tries to keep sections together based on markdown headers.
    Falls back to regular text chunking if no headers found.

    Args:
        text: Markdown text to chunk
        chunk_size: Maximum words per chunk
        chunk_overlap: Words overlap between chunks
        min_chunk_size: Minimum words per chunk

    Returns:
        List of chunk dictionaries
    """
    headers = detect_markdown_headers(text)

    # If no markdown headers, use regular chunking
    if not headers:
        return chunk_text(text, chunk_size, chunk_overlap, min_chunk_size)

    # Split by top-level headers first
    lines = text.split("\n")
    sections: List[Dict[str, Any]] = []
    current_section_start = 0

    # Group by headers
    for i, level, header_text in headers:
        if level <= 2:  # Split on h1 and h2
            if i > current_section_start:
                section_text = "\n".join(lines[current_section_start:i]).strip()
                if section_text:
                    sections.append(
                        {"text": section_text, "header": header_text, "level": level}
                    )
            current_section_start = i

    # Add final section
    if current_section_start < len(lines):
        section_text = "\n".join(lines[current_section_start:]).strip()
        if section_text:
            sections.append({"text": section_text})

    # Chunk each section
    all_chunks: List[Dict[str, Any]] = []
    for section in sections:
        section_chunks = chunk_text(
            section["text"], chunk_size, chunk_overlap, min_chunk_size
        )
        all_chunks.extend(section_chunks)

    logger.info(f"Created {len(all_chunks)} chunks from markdown")
    return all_chunks


class TextDataProcessor(BaseDataProcessor):
    """
    Processes text and markdown files into chunked passages for RAG.

    This processor:
    1. Loads .txt and .md files from a directory
    2. Chunks text with markdown-awareness
    3. Outputs standardized JSON format
    """

    def __init__(
        self,
        input_dir: str,
        output_file: str,
        extensions: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        repo_id: str = "",
        upload: bool = True,
        config: Any = None,
    ):
        """
        Initialize text processor.

        Args:
            input_dir: Directory containing text files
            output_file: Path to output JSON file
            extensions: File extensions to process (default: [".txt", ".md"])
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
        self.extensions = extensions if extensions else [".txt", ".md"]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def load_data(self) -> List[str]:
        """
        Load all text/markdown files from input directory.

        Returns:
            List of file paths

        Raises:
            FileNotFoundError: If input directory doesn't exist
        """
        logger.info(f"Loading text files from {self.input_dir}")

        if not Path(self.input_dir).exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        # Find all files with specified extensions
        all_files = []
        for ext in self.extensions:
            pattern = f"{self.input_dir}/*{ext}"
            files = glob.glob(pattern)
            all_files.extend(files)

        logger.info(f"Found {len(all_files)} text files")
        return all_files

    def process_data(self, text_files: List[str]) -> List[Dict[str, str]]:
        """
        Load and chunk all text files into passages.

        Args:
            text_files: List of text file paths

        Returns:
            List of passage dictionaries with standardized format
        """
        logger.info(f"Processing {len(text_files)} text files")

        all_passages = []

        for file_path in text_files:
            try:
                # Load text
                text = load_text_file(file_path)

                # Determine if markdown
                is_markdown = file_path.endswith(".md")

                # Chunk text
                if is_markdown:
                    chunks = chunk_markdown(
                        text,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        min_chunk_size=self.min_chunk_size,
                    )
                else:
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
                                "source": Path(file_path).name,
                                "source_path": file_path,
                                "is_markdown": is_markdown,
                            },
                        }
                    )

                logger.info(f"Processed {file_path}: {len(chunks)} chunks")

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue

        logger.info(f"Total passages created: {len(all_passages)}")
        return all_passages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process text/markdown files")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/lore",
        help="Directory containing text files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/lore_corpus.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".txt", ".md"],
        help="File extensions to process",
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

    processor = TextDataProcessor(
        input_dir=args.input_dir,
        output_file=args.output_file,
        extensions=args.extensions,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        upload=not args.no_upload,
    )

    passages = processor.process()
    print(f"Successfully processed {len(passages)} passages to {args.output_file}")
