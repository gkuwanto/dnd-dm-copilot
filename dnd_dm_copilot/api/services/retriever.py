"""FAISS-based retrieval service for RAG system."""

from pathlib import Path
from typing import Any, Dict, List

import faiss
from sentence_transformers import SentenceTransformer

from dnd_dm_copilot.utils import get_logger
from dnd_dm_copilot.utils.file_io import load_json_pairs, save_json_pairs

logger = get_logger(__name__)


class FAISSRetriever:
    """
    FAISS-based retriever for semantic search over passages.

    Uses sentence-transformers for encoding and FAISS for efficient
    similarity search with cosine similarity (IndexFlatIP).
    """

    def __init__(self, model_path: str):
        """
        Initialize retriever with a sentence-transformer model.

        Args:
            model_path: Path to sentence-transformers model
                       (local path or HuggingFace model ID)
        """
        logger.info(f"Loading sentence-transformer model: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.index: Any = None  # FAISS index
        self.passages: List[Dict[str, Any]] = []  # Passage storage
        logger.info("Model loaded successfully")

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the model.

        Returns:
            Embedding dimension size
        """
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError("Model does not have embedding dimension")
        return dim

    def build_index(self, passages: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from passages.

        Args:
            passages: List of passage dicts with 'text' and 'metadata'

        Raises:
            ValueError: If no passages provided
        """
        if not passages:
            raise ValueError("No passages provided to build index")

        logger.info(f"Building FAISS index from {len(passages)} passages")

        # Extract text for encoding
        texts = [p["text"] for p in passages]

        # Encode passages
        logger.info("Encoding passages...")
        embeddings = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index (Inner Product for normalized vectors = cosine)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Store passages
        self.passages = passages

        logger.info(
            f"Index built successfully with {self.index.ntotal} vectors "
            f"of dimension {dimension}"
        )

    def save_index(self, path: str) -> None:
        """
        Save FAISS index and passages to disk.

        Args:
            path: Directory path to save index

        Raises:
            ValueError: If index not built yet
        """
        if self.index is None:
            raise ValueError("Index not built yet, cannot save")

        logger.info(f"Saving index to {path}")

        # Create directory if needed
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = Path(path) / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save passages
        passages_file = Path(path) / "passages.json"
        save_json_pairs(self.passages, str(passages_file))

        logger.info(f"Index saved: {index_file}, passages saved: {passages_file}")

    def load_index(self, path: str) -> None:
        """
        Load FAISS index and passages from disk.

        Args:
            path: Directory path containing index

        Raises:
            FileNotFoundError: If index files not found
        """
        logger.info(f"Loading index from {path}")

        # Load FAISS index
        index_file = Path(path) / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        self.index = faiss.read_index(str(index_file))

        # Load passages
        passages_file = Path(path) / "passages.json"
        if not passages_file.exists():
            raise FileNotFoundError(f"Passages file not found: {passages_file}")
        self.passages = load_json_pairs(str(passages_file))

        logger.info(
            f"Index loaded: {self.index.ntotal} vectors, {len(self.passages)} passages"
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for top-k most similar passages to query.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of dicts with 'text', 'score', and 'metadata'

        Raises:
            ValueError: If index not built yet
        """
        if self.index is None:
            raise ValueError("Index not built or loaded yet")

        logger.debug(f"Searching for: '{query}' (top_k={top_k})")

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.passages):  # Valid index
                passage = self.passages[idx]
                results.append(
                    {
                        "text": passage["text"],
                        "score": float(scores[0][i]),
                        "metadata": passage.get("metadata", {}),
                    }
                )

        logger.debug(f"Found {len(results)} results")
        return results
