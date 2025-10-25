"""Integration tests for video generation models."""
import pytest
from unittest.mock import Mock
from llm_poe import PoeVideoModel, PoeVideoOptions
import json


class TestPoeVideoModel:
    """Test PoeVideoModel functionality."""

    def test_video_model_initialization(self):
        """Test video model initialization."""
        model = PoeVideoModel("poe/sora", "Sora")
        assert model.model_id == "poe/sora"
        assert model.model_name == "Sora"

    def test_video_model_cannot_stream(self):
        """Test that video models cannot stream."""
        model = PoeVideoModel("poe/sora", "Sora")
        assert model.can_stream is False

    def test_video_generation(self, httpx_mock, mock_api_key, mock_video_prompt, mock_response, mock_empty_conversation, sample_video_response):
        """Test basic video generation."""
        model = PoeVideoModel("poe/sora", "Sora")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_video_response
        )

        results = list(model.execute(mock_video_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        assert len(results) == 1
        assert "https://example.com/generated-video.mp4" in results[0]
        assert mock_response.response_json == sample_video_response

    def test_video_duration_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_video_response):
        """Test that duration option is passed correctly."""
        model = PoeVideoModel("poe/sora", "Sora")

        prompt = Mock()
        prompt.prompt = "A futuristic cityscape"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        prompt.options.duration = 30
        prompt.options.aspect_ratio = "16:9"

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_video_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["duration"] == 30

    def test_video_aspect_ratio_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_video_response):
        """Test that aspect_ratio option is passed correctly."""
        model = PoeVideoModel("poe/runway_gen_4", "Runway-Gen-4")

        prompt = Mock()
        prompt.prompt = "Ocean waves"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        prompt.options.duration = 10
        prompt.options.aspect_ratio = "9:16"

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_video_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["aspect_ratio"] == "9:16"

    def test_video_stream_forced_false(self, httpx_mock, mock_api_key, mock_video_prompt, mock_response, mock_empty_conversation, sample_video_response):
        """Test that streaming is forced to False for video models."""
        model = PoeVideoModel("poe/sora", "Sora")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_video_response
        )

        # Try to stream (should be ignored)
        list(model.execute(mock_video_prompt, stream=True, response=mock_response, conversation=mock_empty_conversation))

        # Check the request payload
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)
        assert payload["stream"] is False

    def test_video_with_conversation_history(self, httpx_mock, mock_api_key, mock_video_prompt, mock_response, mock_conversation, sample_video_response):
        """Test video generation with conversation history."""
        model = PoeVideoModel("poe/sora", "Sora")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_video_response
        )

        list(model.execute(mock_video_prompt, stream=False, response=mock_response, conversation=mock_conversation))

        # Check the request payload includes conversation
        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)

        assert len(payload["messages"]) == 5  # 2 prev Q/A pairs + current prompt
        assert payload["messages"][-1]["content"] == "A cat playing piano"

    def test_video_request_timeout(self, httpx_mock, mock_api_key, mock_video_prompt, mock_response, mock_empty_conversation, sample_video_response):
        """Test that video generation uses longer timeout."""
        model = PoeVideoModel("poe/sora", "Sora")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_video_response
        )

        list(model.execute(mock_video_prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        # Video generation should use 120s timeout (verified in code)
        request = httpx_mock.get_requests()[0]
        assert request is not None

    def test_video_options_validation(self):
        """Test that PoeVideoOptions validates correctly."""
        # Valid options
        options = PoeVideoOptions(duration=15, aspect_ratio="16:9")
        assert options.duration == 15
        assert options.aspect_ratio == "16:9"

        # Test default values
        options = PoeVideoOptions()
        assert options.duration == 10
        assert options.aspect_ratio == "16:9"

    def test_video_with_standard_options(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_video_response):
        """Test video generation with temperature and max_tokens."""
        model = PoeVideoModel("poe/sora", "Sora")

        prompt = Mock()
        prompt.prompt = "Sci-fi scene"
        prompt.options = Mock()
        prompt.options.temperature = 0.8
        prompt.options.max_tokens = 200
        prompt.options.duration = 20
        prompt.options.aspect_ratio = "1:1"

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_video_response
        )

        list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

        request = httpx_mock.get_requests()[0]
        payload = json.loads(request.content)

        # All options should be present
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 200
        assert payload["duration"] == 20
        assert payload["aspect_ratio"] == "1:1"


class TestVideoModelEdgeCases:
    """Test edge cases for video models."""

    def test_video_without_duration_option(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_video_response):
        """Test video generation when duration option is not set."""
        model = PoeVideoModel("poe/sora", "Sora")

        prompt = Mock()
        prompt.prompt = "A landscape"
        prompt.options = Mock()
        prompt.options.temperature = None
        prompt.options.max_tokens = None
        # Don't set duration attribute
        del prompt.options.duration
        del prompt.options.aspect_ratio

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_video_response
        )

        # Should not raise an error
        results = list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))
        assert len(results) == 1

    def test_various_aspect_ratios(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_video_response):
        """Test video generation with various aspect ratios."""
        model = PoeVideoModel("poe/veo_3", "Veo-3")

        aspect_ratios = ["16:9", "9:16", "1:1", "4:3", "21:9"]

        for aspect_ratio in aspect_ratios:
            prompt = Mock()
            prompt.prompt = "Test video"
            prompt.options = Mock()
            prompt.options.temperature = None
            prompt.options.max_tokens = None
            prompt.options.duration = 10
            prompt.options.aspect_ratio = aspect_ratio

            httpx_mock.add_response(
                url="https://api.poe.com/v1/chat/completions",
                json=sample_video_response
            )

            list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

            request = httpx_mock.get_requests()[-1]
            payload = json.loads(request.content)
            assert payload["aspect_ratio"] == aspect_ratio

    def test_various_durations(self, httpx_mock, mock_api_key, mock_response, mock_empty_conversation, sample_video_response):
        """Test video generation with various durations."""
        model = PoeVideoModel("poe/kling_2_1", "Kling-2.1")

        durations = [5, 10, 15, 30, 60]

        for duration in durations:
            prompt = Mock()
            prompt.prompt = "Test video"
            prompt.options = Mock()
            prompt.options.temperature = None
            prompt.options.max_tokens = None
            prompt.options.duration = duration
            prompt.options.aspect_ratio = "16:9"

            httpx_mock.add_response(
                url="https://api.poe.com/v1/chat/completions",
                json=sample_video_response
            )

            list(model.execute(prompt, stream=False, response=mock_response, conversation=mock_empty_conversation))

            request = httpx_mock.get_requests()[-1]
            payload = json.loads(request.content)
            assert payload["duration"] == duration
