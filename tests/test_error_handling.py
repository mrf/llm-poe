"""Tests for error handling across all models."""
import pytest
from unittest.mock import Mock
import httpx
import llm
from llm_poe import PoeModel, PoeImageModel, PoeVideoModel, PoeAudioModel


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_missing_api_key_error(self, mock_api_key_missing, mock_prompt, mock_response, mock_empty_conversation):
        """Test that missing API key raises appropriate error."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        with pytest.raises(llm.ModelError) as exc_info:
            list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        assert "POE_API_KEY" in str(exc_info.value)

    def test_401_unauthorized_error(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of 401 Unauthorized error."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            status_code=401,
            json={"error": {"message": "Invalid API key"}}
        )

        with pytest.raises(httpx.HTTPStatusError):
            list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

    def test_404_model_not_found_error(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of 404 Model Not Found error."""
        model = PoeModel("poe/nonexistent_model", "Nonexistent-Model")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            status_code=404,
            json={"error": {"message": "Model not found"}}
        )

        with pytest.raises(httpx.HTTPStatusError):
            list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

    def test_429_rate_limit_error(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of 429 Rate Limit error."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            status_code=429,
            json={"error": {"message": "Rate limit exceeded"}}
        )

        with pytest.raises(httpx.HTTPStatusError):
            list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

    def test_500_server_error(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of 500 Internal Server Error."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            status_code=500,
            json={"error": {"message": "Internal server error"}}
        )

        with pytest.raises(httpx.HTTPStatusError):
            list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

    def test_503_service_unavailable_error(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of 503 Service Unavailable error."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            status_code=503,
            json={"error": {"message": "Service temporarily unavailable"}}
        )

        with pytest.raises(httpx.HTTPStatusError):
            list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))


class TestNetworkErrorHandling:
    """Test network-related error handling."""

    def test_connection_timeout(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of connection timeout."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_exception(
            httpx.TimeoutException("Connection timeout")
        )

        with pytest.raises(httpx.TimeoutException):
            list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

    def test_network_error(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of network connection error."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_exception(
            httpx.ConnectError("Failed to connect")
        )

        with pytest.raises(httpx.ConnectError):
            list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

    def test_read_timeout_streaming(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of read timeout during streaming."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_exception(
            httpx.ReadTimeout("Read timeout")
        )

        with pytest.raises(httpx.ReadTimeout):
            list(model.execute(mock_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))


class TestResponseFormatErrorHandling:
    """Test handling of malformed API responses."""

    def test_missing_choices_field(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of response missing 'choices' field."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json={"id": "test", "object": "chat.completion"}  # Missing choices
        )

        results = list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Should handle gracefully (no results)
        assert len(results) == 0

    def test_empty_choices_array(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of empty choices array."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json={"choices": []}  # Empty array
        )

        results = list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Should handle gracefully
        assert len(results) == 0

    def test_invalid_json_response(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of invalid JSON response."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            content=b"This is not JSON"
        )

        with pytest.raises(Exception):  # JSON decode error
            list(model.execute(mock_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))


class TestImageModelErrorHandling:
    """Test error handling specific to image models."""

    def test_image_generation_timeout(self, httpx_mock, mock_api_key, mock_image_prompt, mock_response, mock_empty_conversation):
        """Test image generation timeout handling."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        httpx_mock.add_exception(
            httpx.TimeoutException("Image generation timeout")
        )

        with pytest.raises(httpx.TimeoutException):
            list(model.execute(mock_image_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

    def test_image_400_bad_request(self, httpx_mock, mock_api_key, mock_image_prompt, mock_response, mock_empty_conversation):
        """Test handling of bad request for image generation."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            status_code=400,
            json={"error": {"message": "Invalid image size"}}
        )

        with pytest.raises(httpx.HTTPStatusError):
            list(model.execute(mock_image_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))


class TestVideoModelErrorHandling:
    """Test error handling specific to video models."""

    def test_video_generation_timeout(self, httpx_mock, mock_api_key, mock_video_prompt, mock_response, mock_empty_conversation):
        """Test video generation timeout handling."""
        model = PoeVideoModel("poe/sora", "Sora")

        httpx_mock.add_exception(
            httpx.TimeoutException("Video generation timeout")
        )

        with pytest.raises(httpx.TimeoutException):
            list(model.execute(mock_video_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

    def test_video_400_invalid_duration(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation):
        """Test handling of invalid duration parameter."""
        model = PoeVideoModel("poe/sora", "Sora")

        prompt = Mock()
        prompt.prompt = "Test video"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        prompt.options.duration = 999999  # Invalid duration
        prompt.options.aspect_ratio = "16:9"

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            status_code=400,
            json={"error": {"message": "Invalid duration"}}
        )

        with pytest.raises(httpx.HTTPStatusError):
            list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))


class TestAudioModelErrorHandling:
    """Test error handling specific to audio models."""

    def test_audio_generation_timeout(self, httpx_mock, mock_api_key, mock_audio_prompt, mock_response, mock_empty_conversation):
        """Test audio generation timeout handling."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        httpx_mock.add_exception(
            httpx.TimeoutException("Audio generation timeout")
        )

        with pytest.raises(httpx.TimeoutException):
            list(model.execute(mock_audio_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

    def test_audio_400_invalid_voice(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation):
        """Test handling of invalid voice parameter."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        prompt = Mock()
        prompt.prompt = "Test message"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        prompt.options.voice = "invalid_voice_name"
        prompt.options.speed = 1.0

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            status_code=400,
            json={"error": {"message": "Invalid voice"}}
        )

        with pytest.raises(httpx.HTTPStatusError):
            list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))


class TestStreamingErrorHandling:
    """Test error handling during streaming."""

    def test_streaming_connection_drop(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of connection drop during streaming."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_exception(
            httpx.RemoteProtocolError("Connection closed")
        )

        with pytest.raises(httpx.RemoteProtocolError):
            list(model.execute(mock_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))

    def test_streaming_incomplete_json(self, httpx_mock, mock_api_key, mock_prompt, mock_response, mock_empty_conversation):
        """Test handling of incomplete JSON during streaming."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        chunks = [
            'data: {"choices":[{"delta":{"content":"Valid"}}]}',
            'data: {"incomplete json',  # Incomplete JSON
            'data: [DONE]'
        ]

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            content=b"\n".join(chunk.encode() for chunk in chunks)
        )

        results = list(model.execute(mock_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))

        # Should skip invalid chunk and continue
        assert "Valid" in results
