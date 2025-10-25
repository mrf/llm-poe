"""Shared fixtures for llm-poe tests."""
import pytest
import time
from unittest.mock import Mock, MagicMock
import llm_poe


@pytest.fixture
def mock_api_key(monkeypatch):
    """Mock API key retrieval."""
    def mock_get_key(*args, **kwargs):
        return "test-api-key-12345"

    monkeypatch.setattr("llm.get_key", mock_get_key)
    return "test-api-key-12345"


@pytest.fixture
def mock_api_key_missing(monkeypatch):
    """Mock missing API key."""
    def mock_get_key(*args, **kwargs):
        return None

    monkeypatch.setattr("llm.get_key", mock_get_key)


@pytest.fixture
def sample_models_response():
    """Sample response from Poe /v1/models endpoint."""
    return {
        "object": "list",
        "data": [
            {"id": "GPT-4o", "object": "model", "created": 1677610602, "owned_by": "openai"},
            {"id": "Claude-Sonnet-4", "object": "model", "created": 1677610602, "owned_by": "anthropic"},
            {"id": "Gemini-2.5-Pro", "object": "model", "created": 1677610602, "owned_by": "google"},
            {"id": "Flux-Pro-1.1-Ultra", "object": "model", "created": 1677610602, "owned_by": "black-forest-labs"},
            {"id": "Sora", "object": "model", "created": 1677610602, "owned_by": "openai"},
            {"id": "ElevenLabs-v3", "object": "model", "created": 1677610602, "owned_by": "elevenlabs"},
        ]
    }


@pytest.fixture
def sample_chat_completion_response():
    """Sample chat completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "GPT-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the model."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


@pytest.fixture
def sample_streaming_chunks():
    """Sample streaming response chunks."""
    return [
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"GPT-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"This"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"GPT-4o","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"GPT-4o","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"GPT-4o","choices":[{"index":0,"delta":{"content":" test"},"finish_reason":null}]}',
        'data: [DONE]'
    ]


@pytest.fixture
def sample_image_response():
    """Sample image generation response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "Flux-Pro-1.1-Ultra",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "https://example.com/generated-image.png"
                },
                "finish_reason": "stop"
            }
        ]
    }


@pytest.fixture
def sample_video_response():
    """Sample video generation response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "Sora",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "https://example.com/generated-video.mp4"
                },
                "finish_reason": "stop"
            }
        ]
    }


@pytest.fixture
def sample_audio_response():
    """Sample audio generation response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "ElevenLabs-v3",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "https://example.com/generated-audio.mp3"
                },
                "finish_reason": "stop"
            }
        ]
    }


@pytest.fixture
def clear_model_cache():
    """Clear the global model cache before each test."""
    llm_poe._model_cache = None
    llm_poe._cache_timestamp = None
    yield
    # Clean up after test
    llm_poe._model_cache = None
    llm_poe._cache_timestamp = None


@pytest.fixture
def mock_prompt():
    """Mock llm.Prompt object."""
    prompt = Mock()
    prompt.prompt = "Test prompt"
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = 100
    return prompt


@pytest.fixture
def mock_image_prompt():
    """Mock prompt for image generation."""
    prompt = Mock()
    prompt.prompt = "A beautiful sunset over mountains"
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = None
    prompt.options.size = "1024x1024"
    prompt.options.quality = "standard"
    return prompt


@pytest.fixture
def mock_video_prompt():
    """Mock prompt for video generation."""
    prompt = Mock()
    prompt.prompt = "A cat playing piano"
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = None
    prompt.options.duration = 10
    prompt.options.aspect_ratio = "16:9"
    return prompt


@pytest.fixture
def mock_audio_prompt():
    """Mock prompt for audio generation."""
    prompt = Mock()
    prompt.prompt = "Hello, this is a test message"
    prompt.options = Mock()
    prompt.options.temperature = 1.0
    prompt.options.max_tokens = None
    prompt.options.voice = "alloy"
    prompt.options.speed = 1.0
    return prompt


@pytest.fixture
def mock_response():
    """Mock llm.Response object."""
    response = Mock()
    response.response_json = None
    return response


@pytest.fixture
def mock_conversation():
    """Mock conversation with history."""
    conversation = Mock()

    # Create mock previous responses
    prev_response_1 = Mock()
    prev_response_1.prompt = Mock()
    prev_response_1.prompt.prompt = "First question"
    prev_response_1.text = Mock(return_value="First answer")

    prev_response_2 = Mock()
    prev_response_2.prompt = Mock()
    prev_response_2.prompt.prompt = "Second question"
    prev_response_2.text = Mock(return_value="Second answer")

    conversation.responses = [prev_response_1, prev_response_2]
    return conversation


@pytest.fixture
def mock_empty_conversation():
    """Mock empty conversation (no history)."""
    return None
