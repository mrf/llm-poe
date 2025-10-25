"""Integration tests for text/chat models."""
import pytest
from unittest.mock import Mock
from llm_poe import PoeModel, PoeOptions
import json


class TestPoeModel:
    """Test PoeModel (text/chat) functionality."""

    def test_model_initialization(self):
        """Test model initialization with ID and name."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")
        assert model.model_id == "poe/gpt_4o"
        assert model.model_name == "GPT-4o"

    def test_model_string_representation(self):
        """Test __str__ method for clean display."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")
        assert str(model) == "Poe: gpt_4o"

    def test_model_can_stream(self):
        """Test that text models can stream."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")
        assert model.can_stream is True

    def test_non_streaming_response(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation, sample_chat_completion_response):
        """Test non-streaming chat completion."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_chat_completion_response
        )

        results = list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        assert len(results) == 1
        assert results[0] == "This is a test response from the model."
        assert mock_response.response_json == sample_chat_completion_response

    def test_streaming_response(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation, sample_streaming_chunks):
        """Test streaming chat completion."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        # Mock streaming response
        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            content=b"\n".join(chunk.encode() for chunk in sample_streaming_chunks)
        )

        results = list(model.execute(mock_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))

        # Should have received chunked content
        assert len(results) > 0
        assert "This" in results
        assert " is" in results
        assert " a" in results
        assert " test" in results

    def test_temperature_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_chat_completion_response):
        """Test that temperature option is passed correctly."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        prompt = Mock()
        prompt.prompt = "Test prompt"
        prompt.options = Mock()
        prompt.options.temperature = 0.5
        prompt.options.max_tokens = None

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_chat_completion_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["temperature"] == 0.5

    def test_max_tokens_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_chat_completion_response):
        """Test that max_tokens option is passed correctly."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        prompt = Mock()
        prompt.prompt = "Test prompt"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = 500

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_chat_completion_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 500

    def test_conversation_history(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_conversation, sample_chat_completion_response):
        """Test that conversation history is included in request."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_chat_completion_response
        )

        list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)

        # Should include conversation history
        assert len(payload["messages"]) == 5  # 2 prev Q/A pairs + current prompt
        assert payload["messages"][0] == {"role": "user", "content": "First question"}
        assert payload["messages"][1] == {"role": "assistant", "content": "First answer"}
        assert payload["messages"][2] == {"role": "user", "content": "Second question"}
        assert payload["messages"][3] == {"role": "assistant", "content": "Second answer"}
        assert payload["messages"][4] == {"role": "user", "content": "Test prompt"}

    def test_request_headers(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation, sample_chat_completion_response):
        """Test that correct headers are sent."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_chat_completion_response
        )

        list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check request headers
        request = httpx_mock.get_requests()[0]
        assert request.headers["Authorization"] == "Bearer test-api-key-12345"
        assert request.headers["Content-Type"] == "application/json"

    def test_request_payload_structure(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation, sample_chat_completion_response):
        """Test the structure of the request payload."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_chat_completion_response
        )

        list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload structure
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)

        assert "model" in payload
        assert payload["model"] == "GPT-4o"
        assert "messages" in payload
        assert "stream" in payload
        assert payload["stream"] is False

    def test_options_validation(self):
        """Test that PoeOptions validates correctly."""
        # Valid options
        options = PoeOptions(temperature=1.0, max_tokens=100)
        assert options.temperature == 1.0
        assert options.max_tokens == 100

        # Test temperature bounds
        with pytest.raises(Exception):  # Pydantic validation error
            PoeOptions(temperature=3.0)  # Too high

        with pytest.raises(Exception):  # Pydantic validation error
            PoeOptions(temperature=-0.5)  # Too low

        # Test max_tokens validation
        with pytest.raises(Exception):  # Pydantic validation error
            PoeOptions(max_tokens=0)  # Must be > 0

        with pytest.raises(Exception):  # Pydantic validation error
            PoeOptions(max_tokens=-10)  # Must be > 0


class TestPoeModelEdgeCases:
    """Test edge cases for PoeModel."""

    def test_empty_prompt(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_chat_completion_response):
        """Test handling of empty prompt."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        prompt = Mock()
        prompt.prompt = ""
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_chat_completion_response
        )

        results = list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Should still work with empty prompt
        assert len(results) == 1

    def test_none_options(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_chat_completion_response):
        """Test that None options are not included in payload."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        prompt = Mock()
        prompt.prompt = "Test"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_chat_completion_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)

        # None options should not be in payload
        assert "temperature" not in payload
        assert "max_tokens" not in payload

    def test_streaming_with_empty_chunks(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test streaming with some empty chunks."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{}}]}',  # Empty delta
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: [DONE]'
        ]

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            content=b"\n".join(chunk.encode() for chunk in chunks)
        )

        results = list(model.execute(mock_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))

        # Should only get non-empty content
        assert "Hello" in results
        assert " world" in results
        assert len(results) == 2

    def test_streaming_with_invalid_json(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test that invalid JSON chunks are skipped."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        chunks = [
            'data: {"choices":[{"delta":{"content":"Valid"}}]}',
            'data: {invalid json}',  # Invalid JSON
            'data: {"choices":[{"delta":{"content":" chunk"}}]}',
            'data: [DONE]'
        ]

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            content=b"\n".join(chunk.encode() for chunk in chunks)
        )

        results = list(model.execute(mock_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))

        # Should skip invalid JSON and continue
        assert "Valid" in results
        assert " chunk" in results
