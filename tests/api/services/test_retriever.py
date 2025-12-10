"""Tests for FAISS retriever service."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dnd_dm_copilot.api.services.retriever import FAISSRetriever


class TestFAISSRetriever:
    """Tests for FAISSRetriever class."""

    def test_init_with_model_path(self) -> None:
        """Test initialization with model path."""
        with patch(
            "dnd_dm_copilot.api.services.retriever.SentenceTransformer"
        ) as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            retriever = FAISSRetriever(model_path="models/test")

            mock_st.assert_called_once_with("models/test")
            assert retriever.model == mock_model
            assert retriever.index is None
            assert retriever.passages == []

    def test_build_index_from_passages(self) -> None:
        """Test building FAISS index from passages."""
        with patch(
            "dnd_dm_copilot.api.services.retriever.SentenceTransformer"
        ) as mock_st:
            with patch("faiss.IndexFlatIP") as mock_index_class:
                # Setup mocks
                mock_model = MagicMock()
                mock_embeddings = np.random.rand(3, 384).astype("float32")
                mock_model.encode.return_value = mock_embeddings
                mock_st.return_value = mock_model

                mock_index = MagicMock()
                mock_index_class.return_value = mock_index

                retriever = FAISSRetriever(model_path="models/test")

                passages = [
                    {"text": "passage 1", "metadata": {}},
                    {"text": "passage 2", "metadata": {}},
                    {"text": "passage 3", "metadata": {}},
                ]

                with patch("faiss.normalize_L2") as mock_normalize:
                    retriever.build_index(passages)

                    # Verify encoding
                    mock_model.encode.assert_called_once()
                    call_args = mock_model.encode.call_args[0][0]
                    assert call_args == ["passage 1", "passage 2", "passage 3"]

                    # Verify normalization
                    mock_normalize.assert_called_once()

                    # Verify index creation
                    mock_index_class.assert_called_once_with(384)
                    mock_index.add.assert_called_once()

                    # Verify state
                    assert retriever.passages == passages
                    assert retriever.index == mock_index

    def test_save_index(self, tmp_path: Path) -> None:
        """Test saving FAISS index to disk."""
        with patch("dnd_dm_copilot.api.services.retriever.SentenceTransformer"):
            with patch("faiss.write_index") as mock_write:
                with patch(
                    "dnd_dm_copilot.api.services.retriever.save_json_pairs"
                ) as mock_save_json:
                    retriever = FAISSRetriever(model_path="models/test")
                    retriever.index = MagicMock()
                    retriever.passages = [{"text": "test", "metadata": {}}]

                    save_path = tmp_path / "index"
                    retriever.save_index(str(save_path))

                    # Verify index saved
                    mock_write.assert_called_once()
                    assert str(save_path / "index.faiss") in str(
                        mock_write.call_args[0]
                    )

                    # Verify passages saved
                    mock_save_json.assert_called_once()

    def test_load_index(self, tmp_path: Path) -> None:
        """Test loading FAISS index from disk."""
        with patch("dnd_dm_copilot.api.services.retriever.SentenceTransformer"):
            with patch("faiss.read_index") as mock_read:
                with patch(
                    "dnd_dm_copilot.api.services.retriever.load_json_pairs"
                ) as mock_load_json:
                    with patch(
                        "dnd_dm_copilot.api.services.retriever.Path"
                    ) as mock_path:
                        # Mock Path to return objects that exist
                        mock_path_obj = MagicMock()
                        mock_path_obj.__truediv__ = MagicMock(
                            side_effect=lambda x: MagicMock(
                                exists=MagicMock(return_value=True)
                            )
                        )
                        mock_path.return_value = mock_path_obj

                        mock_index = MagicMock()
                        mock_read.return_value = mock_index
                        mock_load_json.return_value = [{"text": "test", "metadata": {}}]

                        retriever = FAISSRetriever(model_path="models/test")

                        load_path = tmp_path / "index"
                        retriever.load_index(str(load_path))

                        # Verify index loaded
                        mock_read.assert_called_once()
                        assert retriever.index == mock_index

                        # Verify passages loaded
                        mock_load_json.assert_called_once()
                        assert len(retriever.passages) == 1

    def test_search_returns_top_k_results(self) -> None:
        """Test that search returns top-k results."""
        with patch(
            "dnd_dm_copilot.api.services.retriever.SentenceTransformer"
        ) as mock_st:
            mock_model = MagicMock()
            mock_query_embedding = np.random.rand(1, 384).astype("float32")
            mock_model.encode.return_value = mock_query_embedding
            mock_st.return_value = mock_model

            retriever = FAISSRetriever(model_path="models/test")
            retriever.passages = [
                {"text": f"passage {i}", "metadata": {"id": i}} for i in range(10)
            ]

            # Mock index search
            mock_index = MagicMock()
            # Return scores and indices for top 3
            mock_index.search.return_value = (
                np.array([[0.9, 0.8, 0.7]]),
                np.array([[2, 5, 1]]),
            )
            retriever.index = mock_index

            with patch("faiss.normalize_L2"):
                results = retriever.search("test query", top_k=3)

            assert len(results) == 3
            assert results[0]["text"] == "passage 2"
            assert results[0]["score"] == 0.9
            assert results[1]["text"] == "passage 5"
            assert results[1]["score"] == 0.8
            assert results[2]["text"] == "passage 1"
            assert results[2]["score"] == 0.7

    def test_search_includes_metadata(self) -> None:
        """Test that search results include metadata."""
        with patch(
            "dnd_dm_copilot.api.services.retriever.SentenceTransformer"
        ) as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(1, 384).astype("float32")
            mock_st.return_value = mock_model

            retriever = FAISSRetriever(model_path="models/test")
            retriever.passages = [
                {"text": "passage 1", "metadata": {"source": "doc1.pdf"}},
                {"text": "passage 2", "metadata": {"source": "doc2.pdf"}},
            ]

            mock_index = MagicMock()
            mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
            retriever.index = mock_index

            with patch("faiss.normalize_L2"):
                results = retriever.search("test", top_k=1)

            assert results[0]["metadata"]["source"] == "doc1.pdf"

    def test_search_requires_built_index(self) -> None:
        """Test that search raises error if index not built."""
        with patch("dnd_dm_copilot.api.services.retriever.SentenceTransformer"):
            retriever = FAISSRetriever(model_path="models/test")

            with pytest.raises(ValueError, match="Index not built"):
                retriever.search("test query")

    def test_build_index_handles_empty_passages(self) -> None:
        """Test that building index with empty passages raises error."""
        with patch("dnd_dm_copilot.api.services.retriever.SentenceTransformer"):
            retriever = FAISSRetriever(model_path="models/test")

            with pytest.raises(ValueError, match="No passages provided"):
                retriever.build_index([])

    def test_get_embedding_dimension(self) -> None:
        """Test getting embedding dimension from model."""
        with patch(
            "dnd_dm_copilot.api.services.retriever.SentenceTransformer"
        ) as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            retriever = FAISSRetriever(model_path="models/test")
            dim = retriever.get_embedding_dimension()

            assert dim == 384
            mock_model.get_sentence_embedding_dimension.assert_called_once()
