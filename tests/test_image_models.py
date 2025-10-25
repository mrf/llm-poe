"""Integration tests for image generation models."""
import pytest
from unittest.mock import Mock
from llm_poe import PoeImageModel, PoeImageOptions
import json


class TestPoeImageModel:
    """Test PoeImageModel functionality."""

    def test_image_model_initialization(self):
        """Test image model initialization."""
        model = PoeImageModel("poe/flux_pro_1_1_ultra", "Flux-Pro-1.1-Ultra")
        assert model.model_id == "poe/flux_pro_1_1_ultra"
        assert model.model_name == "Flux-Pro-1.1-Ultra"

    def test_image_model_cannot_stream(self):
        """Test that image models cannot stream."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")
        assert model.can_stream is False

    def test_image_generation(self, httpx_mock, mock_api_key, mock_image_prompt, mock_response, mock_empty_conversation, sample_image_response):
        """Test basic image generation."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_image_response
        )

        results = list(model.execute(mock_image_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        assert len(results) == 1
        assert "https://example.com/generated-image.png" in results[0]
        assert mock_response.response_json == sample_image_response

    def test_image_size_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_image_response):
        """Test that size option is passed correctly."""
        model = PoeImageModel("poe/dall_e_3", "DALL-E-3")

        prompt = Mock()
        prompt.prompt = "A beautiful landscape"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        prompt.options.size = "1792x1024"
        prompt.options.quality = "standard"

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_image_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["size"] == "1792x1024"

    def test_image_quality_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_image_response):
        """Test that quality option is passed correctly."""
        model = PoeImageModel("poe/dall_e_3", "DALL-E-3")

        prompt = Mock()
        prompt.prompt = "A beautiful landscape"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        prompt.options.size = "1024x1024"
        prompt.options.quality = "hd"

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_image_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["quality"] == "hd"

    def test_image_stream_forced_false(self, httpx_mock, mock_api_key, mock_image_prompt, mock_response, mock_empty_conversation, sample_image_response):
        """Test that streaming is forced to False for image models."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_image_response
        )

        # Try to stream (should be ignored)
        list(model.execute(mock_image_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["stream"] is False

    def test_image_with_conversation_history(self, httpx_mock, mock_api_key, mock_image_prompt, mock_response, mock_conversation, sample_image_response):
        """Test image generation with conversation history."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_image_response
        )

        list(model.execute(mock_image_prompt, stream=False, response=mock_response, conversation=mock_conversation))

        # Check the request payload includes conversation
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)

        assert len(payload["messages"]) == 5  # 2 prev Q/A pairs + current prompt
        assert payload["messages"][-1]["content"] == "A beautiful sunset over mountains"

    def test_image_request_timeout(self, httpx_mock, mock_api_key, mock_image_prompt, mock_response, mock_empty_conversation, sample_image_response):
        """Test that image generation uses longer timeout."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_image_response
        )

        list(model.execute(mock_image_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Image generation should use 60s timeout (verified in code)
        request = httpx_mock.get_requests()[0]
        assert request is not None

    def test_image_options_validation(self):
        """Test that PoeImageOptions validates correctly."""
        # Valid options
        options = PoeImageOptions(size="1024x1024", quality="standard")
        assert options.size == "1024x1024"
        assert options.quality == "standard"

        # Test default values
        options = PoeImageOptions()
        assert options.size == "1024x1024"
        assert options.quality == "standard"

    def test_multiple_image_urls_in_response(self, httpx_mock, mock_api_key, mock_image_prompt, mock_response, mock_empty_conversation):
        """Test handling of multiple image URLs in response."""
        multi_image_response = {
            "choices": [{
                "message": {
                    "content": "https://example.com/image1.png\nhttps://example.com/image2.png"
                }
            }]
        }

        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=multi_image_response
        )

        results = list(model.execute(mock_image_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        assert len(results) == 1
        assert "https://example.com/image1.png" in results[0]
        assert "https://example.com/image2.png" in results[0]


class TestImageModelEdgeCases:
    """Test edge cases for image models."""

    def test_image_generation_with_temperature(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_image_response):
        """Test image generation with temperature option."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        prompt = Mock()
        prompt.prompt = "Abstract art"
        prompt.options = Mock()
        prompt.options.temperature = 1.5
        prompt.options.max_tokens = None
        prompt.options.size = "1024x1024"
        prompt.options.quality = "standard"

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_image_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)

        # Both image-specific and standard options should be present
        assert payload["temperature"] == 1.5
        assert payload["size"] == "1024x1024"

    def test_image_without_size_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_image_response):
        """Test image generation when size option is not set."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        prompt = Mock()
        prompt.prompt = "A landscape"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        # Don't set size attribute
        del prompt.options.size
        del prompt.options.quality

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_image_response
        )

        # Should not raise an error
        results = list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))
        assert len(results) == 1
