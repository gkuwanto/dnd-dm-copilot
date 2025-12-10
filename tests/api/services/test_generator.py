"""Tests for RAG generator service."""

from unittest.mock import MagicMock

from dnd_dm_copilot.api.services.generator import RAGGenerator


class TestRAGGenerator:
    """Tests for RAGGenerator class."""

    def test_init_with_retriever_and_llm(self) -> None:
        """Test initialization with retriever and LLM client."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        generator = RAGGenerator(retriever=mock_retriever, llm_client=mock_llm)

        assert generator.retriever == mock_retriever
        assert generator.llm_client == mock_llm

    def test_generate_retrieves_and_generates(self) -> None:
        """Test that generate performs retrieval and generation."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        # Mock retrieval results
        mock_retriever.search.return_value = [
            {"text": "Divine Smite deals radiant damage.", "score": 0.9},
            {"text": "It uses spell slots.", "score": 0.8},
        ]

        # Mock LLM generation
        mock_llm.generate.return_value = (
            "Divine Smite allows paladins to expend spell slots to deal radiant damage."
        )

        generator = RAGGenerator(retriever=mock_retriever, llm_client=mock_llm)

        query = "How does Divine Smite work?"
        result = generator.generate(query=query, top_k=2)

        # Verify retrieval was called
        mock_retriever.search.assert_called_once_with(query=query, top_k=2)

        # Verify generation was called with retrieved context
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args
        assert call_args[1]["query"] == query
        assert len(call_args[1]["context"]) == 2
        assert "Divine Smite deals radiant damage." in call_args[1]["context"]

        # Verify result structure
        assert result["query"] == query
        assert result["answer"] == mock_llm.generate.return_value
        assert len(result["passages"]) == 2
        assert result["passages"][0]["text"] == "Divine Smite deals radiant damage."
        assert result["passages"][0]["score"] == 0.9

    def test_generate_with_custom_top_k(self) -> None:
        """Test generation with custom top_k parameter."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        mock_retriever.search.return_value = [
            {"text": f"passage {i}", "score": 0.9 - i * 0.1} for i in range(5)
        ]
        mock_llm.generate.return_value = "answer"

        generator = RAGGenerator(retriever=mock_retriever, llm_client=mock_llm)

        generator.generate(query="test", top_k=5)

        mock_retriever.search.assert_called_once_with(query="test", top_k=5)
        # Should pass 5 passages to LLM
        assert len(mock_llm.generate.call_args[1]["context"]) == 5

    def test_generate_with_temperature(self) -> None:
        """Test that temperature is passed to LLM."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        mock_retriever.search.return_value = [{"text": "context", "score": 0.9}]
        mock_llm.generate.return_value = "answer"

        generator = RAGGenerator(retriever=mock_retriever, llm_client=mock_llm)

        generator.generate(query="test", temperature=0.7)

        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    def test_generate_with_max_tokens(self) -> None:
        """Test that max_tokens is passed to LLM."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        mock_retriever.search.return_value = [{"text": "context", "score": 0.9}]
        mock_llm.generate.return_value = "answer"

        generator = RAGGenerator(retriever=mock_retriever, llm_client=mock_llm)

        generator.generate(query="test", max_tokens=256)

        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["max_tokens"] == 256

    def test_generate_handles_empty_retrieval(self) -> None:
        """Test generation when no passages are retrieved."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        mock_retriever.search.return_value = []
        mock_llm.generate.return_value = "No context available."

        generator = RAGGenerator(retriever=mock_retriever, llm_client=mock_llm)

        result = generator.generate(query="test")

        # Should still call LLM with empty context
        mock_llm.generate.assert_called_once()
        assert mock_llm.generate.call_args[1]["context"] == []
        assert result["passages"] == []
        assert result["answer"] == "No context available."

    def test_generate_preserves_metadata(self) -> None:
        """Test that passage metadata is preserved in results."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        mock_retriever.search.return_value = [
            {
                "text": "passage 1",
                "score": 0.9,
                "metadata": {"source": "phb.pdf", "page": 42},
            },
        ]
        mock_llm.generate.return_value = "answer"

        generator = RAGGenerator(retriever=mock_retriever, llm_client=mock_llm)

        result = generator.generate(query="test")

        assert result["passages"][0]["metadata"]["source"] == "phb.pdf"
        assert result["passages"][0]["metadata"]["page"] == 42

    def test_generate_only_passes_text_to_llm(self) -> None:
        """Test that only passage text (not scores/metadata) is passed to LLM."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        mock_retriever.search.return_value = [
            {
                "text": "passage 1",
                "score": 0.9,
                "metadata": {"source": "doc.pdf"},
            },
            {
                "text": "passage 2",
                "score": 0.8,
                "metadata": {"source": "doc2.pdf"},
            },
        ]
        mock_llm.generate.return_value = "answer"

        generator = RAGGenerator(retriever=mock_retriever, llm_client=mock_llm)

        generator.generate(query="test", top_k=2)

        # Verify only text strings are passed
        context = mock_llm.generate.call_args[1]["context"]
        assert context == ["passage 1", "passage 2"]
        assert all(isinstance(c, str) for c in context)
