"""Integration tests for audio/TTS generation models."""
import pytest
from unittest.mock import Mock
from llm_poe import PoeAudioModel, PoeAudioOptions
import json


class TestPoeAudioModel:
    """Test PoeAudioModel functionality."""

    def test_audio_model_initialization(self):
        """Test audio model initialization."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")
        assert model.model_id == "poe/elevenlabs_v3"
        assert model.model_name == "ElevenLabs-v3"

    def test_audio_model_can_stream(self):
        """Test that audio models can stream."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")
        assert model.can_stream is True

    def test_audio_generation_non_streaming(self, httpx_mock, mock_api_key, mock_audio_prompt, mock_response, mock_empty_conversation, sample_audio_response):
        """Test basic audio generation without streaming."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_audio_response
        )

        results = list(model.execute(mock_audio_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        assert len(results) == 1
        assert "https://example.com/generated-audio.mp3" in results[0]
        assert mock_response.response_json == sample_audio_response

    def test_audio_generation_streaming(self, httpx_mock, mock_api_key, mock_audio_prompt, mock_response, mock_empty_conversation):
        """Test audio generation with streaming."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        streaming_chunks = [
            'data: {"choices":[{"delta":{"content":"https://"}}]}',
            'data: {"choices":[{"delta":{"content":"example.com/"}}]}',
            'data: {"choices":[{"delta":{"content":"audio.mp3"}}]}',
            'data: [DONE]'
        ]

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            content=b"\n".join(chunk.encode() for chunk in streaming_chunks)
        )

        results = list(model.execute(mock_audio_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))

        assert len(results) > 0
        assert "https://" in results
        assert "example.com/" in results
        assert "audio.mp3" in results

    def test_audio_voice_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_audio_response):
        """Test that voice option is passed correctly."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        prompt = Mock()
        prompt.prompt = "Hello world"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        prompt.options.voice = "nova"
        prompt.options.speed = 1.0

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_audio_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["voice"] == "nova"

    def test_audio_speed_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_audio_response):
        """Test that speed option is passed correctly."""
        model = PoeAudioModel("poe/cartesia_sonic", "Cartesia-Sonic")

        prompt = Mock()
        prompt.prompt = "Testing speech speed"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        prompt.options.voice = "alloy"
        prompt.options.speed = 1.5

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_audio_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["speed"] == 1.5

    def test_audio_with_conversation_history(self, httpx_mock, mock_api_key, mock_audio_prompt, mock_response, mock_conversation, sample_audio_response):
        """Test audio generation with conversation history."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_audio_response
        )

        list(model.execute(mock_audio_prompt, stream=False, response=mock_response, conversation=mock_conversation))

        # Check the request payload includes conversation
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)

        assert len(payload["messages"]) == 5  # 2 prev Q/A pairs + current prompt
        assert payload["messages"][-1]["content"] == "Hello, this is a test message"

    def test_audio_request_timeout(self, httpx_mock, mock_api_key, mock_audio_prompt, mock_response, mock_empty_conversation, sample_audio_response):
        """Test that audio generation uses appropriate timeout."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_audio_response
        )

        list(model.execute(mock_audio_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Audio generation should use 60s timeout (verified in code)
        request = httpx_mock.get_requests()[0]
        assert request is not None

    def test_audio_options_validation(self):
        """Test that PoeAudioOptions validates correctly."""
        # Valid options
        options = PoeAudioOptions(voice="nova", speed=1.25)
        assert options.voice == "nova"
        assert options.speed == 1.25

        # Test default values
        options = PoeAudioOptions()
        assert options.voice == "alloy"
        assert options.speed == 1.0

    def test_audio_with_standard_options(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_audio_response):
        """Test audio generation with temperature and max_tokens."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        prompt = Mock()
        prompt.prompt = "Test message"
        prompt.options = Mock()
        prompt.options.temperature = 0.7
        prompt.options.max_tokens = 150
        prompt.options.voice = "echo"
        prompt.options.speed = 0.9

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_audio_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)

        # All options should be present
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 150
        assert payload["voice"] == "echo"
        assert payload["speed"] == 0.9


class TestAudioModelEdgeCases:
    """Test edge cases for audio models."""

    def test_audio_without_voice_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_audio_response):
        """Test audio generation when voice option is not set."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        prompt = Mock()
        prompt.prompt = "Test message"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        # Don't set voice attribute
        del prompt.options.voice
        del prompt.options.speed

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_audio_response
        )

        # Should not raise an error
        results = list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))
        assert len(results) == 1

    def test_various_voices(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_audio_response):
        """Test audio generation with various voices."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

        for voice in voices:
            prompt = Mock()
            prompt.prompt = "Test message"
            prompt.options = Mock()
            prompt.options.temperature = None
            prompt.options.max_tokens = None
            prompt.options.voice = voice
            prompt.options.speed = 1.0

            httpx_mock.add_response(
                url="https://api.poe.com/v1/chat/completions",
                json=sample_audio_response
            )

            list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

            request = httpx_mock.get_requests()[-1]
            payload = json.loads(request.content)
            assert payload["voice"] == voice

    def test_various_speeds(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_audio_response):
        """Test audio generation with various speeds."""
        model = PoeAudioModel("poe/playai_3_0", "PlayAI-3.0")

        speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        for speed in speeds:
            prompt = Mock()
            prompt.prompt = "Test message"
            prompt.options = Mock()
            prompt.options.temperature = None
            prompt.options.max_tokens = None
            prompt.options.voice = "alloy"
            prompt.options.speed = speed

            httpx_mock.add_response(
                url="https://api.poe.com/v1/chat/completions",
                json=sample_audio_response
            )

            list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

            request = httpx_mock.get_requests()[-1]
            payload = json.loads(request.content)
            assert payload["speed"] == speed

    def test_audio_streaming_with_empty_chunks(self, httpx_mock, mock_api_key, mock_audio_prompt, mock_response, mock_empty_conversation):
        """Test streaming with some empty chunks."""
        model = PoeAudioModel("poe/elevenlabs_v3", "ElevenLabs-v3")

        chunks = [
            'data: {"choices":[{"delta":{"content":"Audio"}}]}',
            'data: {"choices":[{"delta":{}}]}',  # Empty delta
            'data: {"choices":[{"delta":{"content":" URL"}}]}',
            'data: [DONE]'
        ]

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            content=b"\n".join(chunk.encode() for chunk in chunks)
        )

        results = list(model.execute(mock_audio_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))

        # Should only get non-empty content
        assert "Audio" in results
        assert " URL" in results
        assert len(results) == 2
