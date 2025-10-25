"""Tests for model registration functionality."""
import pytest
from unittest.mock import Mock, patch
import llm_poe
from llm_poe import (
    fetch_available_models,
    register_models,
    PoeModel,
    PoeImageModel,
    PoeVideoModel,
    PoeAudioModel,
    FALLBACK_MODELS
)


class TestFetchAvailableModels:
    """Test the fetch_available_models function."""

    def test_fetch_models_success(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test successful model fetching from API."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        models = fetch_available_models()

        assert len(models) == 6
        assert models[0]["id"] == "GPT-4o"
        assert models[1]["id"] == "Claude-Sonnet-4"
        assert models[3]["id"] == "Flux-Pro-1.1-Ultra"

    def test_fetch_models_caching(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that model list is cached correctly."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # First call
        models1 = fetch_available_models()

        # Second call should use cache (no new HTTP request)
        models2 = fetch_available_models()

        assert models1 == models2
        # Should only have made one request due to caching
        assert len(httpx_mock.get_requests()) == 1

    def test_fetch_models_cache_expiration(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that cache expires after the configured duration."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # First call
        models1 = fetch_available_models()

        # Manually expire cache
        import time
        llm_poe._cache_timestamp = time.time() - 3601  # 1 hour + 1 second ago

        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # Second call should fetch again due to expired cache
        models2 = fetch_available_models()

        # Should have made two requests
        assert len(httpx_mock.get_requests()) == 2

    def test_fetch_models_no_api_key(self, mock_api_key_missing, clear_model_cache):
        """Test fallback when no API key is available."""
        models = fetch_available_models()

        assert models == FALLBACK_MODELS
        assert len(models) == 5

    def test_fetch_models_api_error(self, httpx_mock, mock_api_key, clear_model_cache):
        """Test fallback on API error."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            status_code=500
        )

        models = fetch_available_models()

        assert models == FALLBACK_MODELS

    def test_fetch_models_network_timeout(self, httpx_mock, mock_api_key, clear_model_cache):
        """Test fallback on network timeout."""
        import httpx

        httpx_mock.add_exception(
            httpx.TimeoutException("Connection timeout")
        )

        models = fetch_available_models()

        assert models == FALLBACK_MODELS


class TestRegisterModels:
    """Test the register_models hook."""

    def test_register_models_success(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test successful model registration."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        registered_models = []

        def mock_register(model):
            registered_models.append(model)

        register_models(mock_register)

        # Should register 6 models from sample response
        assert len(registered_models) == 6

        # Check model IDs are properly formatted
        assert registered_models[0].model_id == "poe/gpt_4o"
        assert registered_models[1].model_id == "poe/claude_sonnet_4"

    def test_register_models_type_detection(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that models are registered with correct types."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        registered_models = []

        def mock_register(model):
            registered_models.append(model)

        register_models(mock_register)

        # Find specific model types
        gpt_model = next(m for m in registered_models if "gpt_4o" in m.model_id)
        flux_model = next(m for m in registered_models if "flux" in m.model_id)
        sora_model = next(m for m in registered_models if "sora" in m.model_id)
        elevenlabs_model = next(m for m in registered_models if "elevenlabs" in m.model_id)

        assert isinstance(gpt_model, PoeModel)
        assert not isinstance(gpt_model, (PoeImageModel, PoeVideoModel, PoeAudioModel))

        assert isinstance(flux_model, PoeImageModel)
        assert isinstance(sora_model, PoeVideoModel)
        assert isinstance(elevenlabs_model, PoeAudioModel)

    def test_register_models_fallback_on_error(self, httpx_mock, mock_api_key, clear_model_cache):
        """Test that fallback models are registered on API error."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            status_code=500
        )

        registered_models = []

        def mock_register(model):
            registered_models.append(model)

        register_models(mock_register)

        # Should register fallback models
        assert len(registered_models) == len(FALLBACK_MODELS)

    def test_model_id_sanitization(self, httpx_mock, mock_api_key, clear_model_cache):
        """Test that model IDs are properly sanitized."""
        models_with_special_chars = {
            "object": "list",
            "data": [
                {"id": "Model-With.Dots", "object": "model"},
                {"id": "Model With Spaces", "object": "model"},
                {"id": "Model_With_Underscores", "object": "model"},
            ]
        }

        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=models_with_special_chars
        )

        registered_models = []

        def mock_register(model):
            registered_models.append(model)

        register_models(mock_register)

        # Check sanitized IDs
        assert registered_models[0].model_id == "poe/model_with_dots"
        assert registered_models[1].model_id == "poe/model_with_spaces"
        assert registered_models[2].model_id == "poe/model_with_underscores"

    def test_model_name_preservation(self, httpx_mock, mock_api_key, clear_model_cache):
        """Test that original model names are preserved for API calls."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json={
                "object": "list",
                "data": [{"id": "GPT-4o", "object": "model"}]
            }
        )

        registered_models = []

        def mock_register(model):
            registered_models.append(model)

        register_models(mock_register)

        # Original name should be preserved
        assert registered_models[0].model_name == "GPT-4o"
        # But ID should be sanitized
        assert registered_models[0].model_id == "poe/gpt_4o"
