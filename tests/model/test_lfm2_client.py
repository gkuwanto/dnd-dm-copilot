"""Tests for LFM2 client using llama-cpp-python."""

from unittest.mock import MagicMock, patch

import pytest
from dnd_dm_copilot.model.lfm2_client import LFM2Client


class TestLFM2Client:
    """Tests for LFM2Client class."""

    def test_init_with_model_path(self) -> None:
        """Test initialization with model path."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_model = MagicMock()
            mock_llama.return_value = mock_model

            client = LFM2Client(
                model_path="models/test.gguf",
                n_gpu_layers=-1,
                n_ctx=4096,
            )

            mock_llama.assert_called_once_with(
                model_path="models/test.gguf",
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False,
                chat_format="chatml",
                flash_attn=True,
            )
            assert client.model == mock_model

    def test_generate_with_query_and_context(self) -> None:
        """Test generation with query and context passages."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_model = MagicMock()
            mock_llama.return_value = mock_model

            # Mock the create_chat_completion output
            mock_model.create_chat_completion.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Divine Smite allows you to deal extra radiant damage."
                            )
                        }
                    }
                ]
            }

            client = LFM2Client(model_path="models/test.gguf")

            query = "How does Divine Smite work?"
            context = [
                "Divine Smite is a Paladin feature.",
                "It deals radiant damage.",
            ]

            result = client.generate(query=query, context=context)

            # Verify create_chat_completion was called
            assert mock_model.create_chat_completion.call_count == 1
            call_kwargs = mock_model.create_chat_completion.call_args[1]

            # Verify messages contain query and context
            messages = call_kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert query in messages[1]["content"]
            assert context[0] in messages[1]["content"]
            assert context[1] in messages[1]["content"]

            # Verify result
            assert result == "Divine Smite allows you to deal extra radiant damage."

    def test_generate_with_temperature(self) -> None:
        """Test generation with custom temperature."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_model = MagicMock()
            mock_llama.return_value = mock_model
            mock_model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "answer"}}]
            }

            client = LFM2Client(model_path="models/test.gguf")
            client.generate(query="test", context=[], temperature=0.7)

            call_kwargs = mock_model.create_chat_completion.call_args[1]
            assert call_kwargs["temperature"] == 0.7

    def test_generate_with_max_tokens(self) -> None:
        """Test generation with custom max_tokens."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_model = MagicMock()
            mock_llama.return_value = mock_model
            mock_model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "answer"}}]
            }

            client = LFM2Client(model_path="models/test.gguf")
            client.generate(query="test", context=[], max_tokens=256)

            call_kwargs = mock_model.create_chat_completion.call_args[1]
            assert call_kwargs["max_tokens"] == 256

    def test_generate_formats_prompt_correctly(self) -> None:
        """Test that messages are formatted with context and query."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_model = MagicMock()
            mock_llama.return_value = mock_model
            mock_model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "answer"}}]
            }

            client = LFM2Client(model_path="models/test.gguf")

            query = "What is AC?"
            context = ["AC stands for Armor Class.", "Higher AC is better."]

            client.generate(query=query, context=context)

            call_kwargs = mock_model.create_chat_completion.call_args[1]
            messages = call_kwargs["messages"]

            # Verify system message contains assistant role info
            assert "Dungeon Master" in messages[0]["content"]

            # Verify user message contains context and query
            user_content = messages[1]["content"]
            assert "Context:" in user_content
            assert context[0] in user_content
            assert context[1] in user_content
            assert query in user_content

    def test_generate_handles_empty_context(self) -> None:
        """Test generation with empty context list."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_model = MagicMock()
            mock_llama.return_value = mock_model
            mock_model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "answer"}}]
            }

            client = LFM2Client(model_path="models/test.gguf")
            result = client.generate(query="test query", context=[])

            assert result == "answer"
            assert mock_model.create_chat_completion.call_count == 1

    def test_generate_strips_whitespace(self) -> None:
        """Test that generated text is stripped of whitespace."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_model = MagicMock()
            mock_llama.return_value = mock_model
            mock_model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "  answer with spaces  "}}]
            }

            client = LFM2Client(model_path="models/test.gguf")
            result = client.generate(query="test", context=[])

            assert result == "answer with spaces"

    def test_generate_with_stop_sequences(self) -> None:
        """Test generation with stop sequences."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_model = MagicMock()
            mock_llama.return_value = mock_model
            mock_model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "answer"}}]
            }

            client = LFM2Client(model_path="models/test.gguf")
            client.generate(query="test", context=[], stop=["</s>", "\n\nQuestion:"])

            call_kwargs = mock_model.create_chat_completion.call_args[1]
            assert call_kwargs["stop"] == ["</s>", "\n\nQuestion:"]

    def test_init_raises_error_for_missing_model(self) -> None:
        """Test that initialization raises error if model file not found."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_llama.side_effect = ValueError("Model file not found")

            with pytest.raises(ValueError, match="Model file not found"):
                LFM2Client(model_path="nonexistent.gguf")

    def test_generate_with_top_p_and_top_k(self) -> None:
        """Test generation with top_p and top_k sampling parameters."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_model = MagicMock()
            mock_llama.return_value = mock_model
            mock_model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "answer"}}]
            }

            client = LFM2Client(model_path="models/test.gguf")
            client.generate(query="test", context=[], top_p=0.9, top_k=40)

            call_kwargs = mock_model.create_chat_completion.call_args[1]
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["top_k"] == 40

    def test_create_messages_with_context(self) -> None:
        """Test _create_messages method with context."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_llama.return_value = MagicMock()

            client = LFM2Client(model_path="models/test.gguf")
            messages = client._create_messages(
                query="What is AC?",
                context=["AC stands for Armor Class."],
            )

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert "Context:" in messages[1]["content"]
            assert "Question:" in messages[1]["content"]

    def test_create_messages_without_context(self) -> None:
        """Test _create_messages method without context."""
        with patch("dnd_dm_copilot.model.lfm2_client.Llama") as mock_llama:
            mock_llama.return_value = MagicMock()

            client = LFM2Client(model_path="models/test.gguf")
            messages = client._create_messages(query="What is AC?", context=[])

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "What is AC?"
