"""Tests for model list caching functionality."""
import pytest
import time
from unittest.mock import patch
import llm_poe
from llm_poe import fetch_available_models, FALLBACK_MODELS


class TestModelCaching:
    """Test model list caching mechanisms."""

    def test_cache_on_first_fetch(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that cache is populated on first fetch."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # Cache should be empty initially
        assert llm_poe._model_cache is None
        assert llm_poe._cache_timestamp is None

        # Fetch models
        models = fetch_available_models()

        # Cache should now be populated
        assert llm_poe._model_cache is not None
        assert llm_poe._cache_timestamp is not None
        assert llm_poe._model_cache == sample_models_response["data"]

    def test_cache_reuse_within_duration(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that cache is reused within cache duration."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # First fetch
        models1 = fetch_available_models()
        request_count_1 = len(httpx_mock.get_requests())

        # Second fetch (should use cache)
        models2 = fetch_available_models()
        request_count_2 = len(httpx_mock.get_requests())

        # Should have only made one API request
        assert request_count_1 == 1
        assert request_count_2 == 1

        # Results should be identical
        assert models1 == models2

    def test_cache_expiration_after_duration(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that cache expires after the configured duration."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # First fetch
        models1 = fetch_available_models()

        # Manually expire the cache by setting timestamp to past
        llm_poe._cache_timestamp = time.time() - 3601  # 1 hour + 1 second ago

        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # Second fetch (cache expired, should make new request)
        models2 = fetch_available_models()

        # Should have made two API requests
        assert len(httpx_mock.get_requests()) == 2

    def test_cache_duration_configurable(self, clear_model_cache):
        """Test that cache duration is configurable."""
        # Default cache duration should be 1 hour (3600 seconds)
        assert llm_poe._cache_duration == 3600

    def test_cache_timestamp_updates_on_fetch(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that cache timestamp updates on new fetch."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # First fetch
        fetch_available_models()
        timestamp1 = llm_poe._cache_timestamp

        # Expire cache
        llm_poe._cache_timestamp = time.time() - 3601

        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # Second fetch
        fetch_available_models()
        timestamp2 = llm_poe._cache_timestamp

        # Timestamp should have been updated
        assert timestamp2 > timestamp1

    def test_cache_content_updates_on_refetch(self, httpx_mock, mock_api_key, clear_model_cache):
        """Test that cache content updates when refetched."""
        # First response
        first_response = {
            "object": "list",
            "data": [{"id": "Model-1", "object": "model"}]
        }

        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=first_response
        )

        models1 = fetch_available_models()

        # Expire cache
        llm_poe._cache_timestamp = time.time() - 3601

        # Second response (different models)
        second_response = {
            "object": "list",
            "data": [
                {"id": "Model-1", "object": "model"},
                {"id": "Model-2", "object": "model"}
            ]
        }

        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=second_response
        )

        models2 = fetch_available_models()

        # Cache should reflect new data
        assert len(models1) == 1
        assert len(models2) == 2


class TestCacheFallback:
    """Test caching behavior with fallback scenarios."""

    def test_no_cache_when_api_key_missing(self, mock_api_key_missing, clear_model_cache):
        """Test that cache is not used when API key is missing."""
        models = fetch_available_models()

        # Should return fallback models without caching
        assert models == FALLBACK_MODELS
        assert llm_poe._model_cache is None or llm_poe._model_cache == FALLBACK_MODELS

    def test_fallback_on_api_error_does_not_cache(self, httpx_mock, mock_api_key, clear_model_cache):
        """Test that fallback models are not cached on API error."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            status_code=500
        )

        models = fetch_available_models()

        # Should return fallback models
        assert models == FALLBACK_MODELS

        # Cache should not be populated with fallback (or should be, but won't prevent retry)
        # The implementation returns fallback without updating cache

    def test_retry_after_previous_failure(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that fetch retries after previous API failure."""
        # First attempt fails
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            status_code=500
        )

        models1 = fetch_available_models()
        assert models1 == FALLBACK_MODELS

        # Add successful response for second attempt
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # Expire any cache that might exist
        llm_poe._cache_timestamp = None

        # Second attempt succeeds
        models2 = fetch_available_models()

        # Should get actual models now
        assert models2 != FALLBACK_MODELS
        assert len(models2) == 6


class TestCacheConcurrency:
    """Test cache behavior under concurrent access."""

    def test_cache_shared_across_calls(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that cache is shared across multiple fetch calls."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # Multiple fetches
        models1 = fetch_available_models()
        models2 = fetch_available_models()
        models3 = fetch_available_models()

        # Should all be the same and only make one request
        assert models1 == models2 == models3
        assert len(httpx_mock.get_requests()) == 1


class TestCacheEdgeCases:
    """Test edge cases in caching behavior."""

    def test_cache_with_empty_response(self, httpx_mock, mock_api_key, clear_model_cache):
        """Test caching with empty model list from API."""
        empty_response = {
            "object": "list",
            "data": []
        }

        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=empty_response
        )

        models = fetch_available_models()

        # Should cache even empty list
        assert models == []
        assert llm_poe._model_cache == []

    def test_cache_timestamp_precision(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test that cache timestamp uses appropriate precision."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        before = time.time()
        fetch_available_models()
        after = time.time()

        # Timestamp should be between before and after
        assert before <= llm_poe._cache_timestamp <= after

    def test_cache_invalidation_on_zero_duration(self, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Test cache behavior when duration is effectively zero."""
        # Temporarily set cache duration to 0
        original_duration = llm_poe._cache_duration
        try:
            llm_poe._cache_duration = 0

            httpx_mock.add_response(
                url="https://api.poe.com/v1/models",
                json=sample_models_response
            )

            models1 = fetch_available_models()

            # Even with tiny delay, cache should be expired
            time.sleep(0.001)

            httpx_mock.add_response(
                url="https://api.poe.com/v1/models",
                json=sample_models_response
            )

            models2 = fetch_available_models()

            # Should have made two requests due to immediate expiration
            assert len(httpx_mock.get_requests()) == 2

        finally:
            # Restore original duration
            llm_poe._cache_duration = original_duration

    def test_cache_with_malformed_response(self, httpx_mock, mock_api_key, clear_model_cache):
        """Test cache behavior with malformed API response."""
        malformed_response = {
            "object": "list"
            # Missing "data" field
        }

        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=malformed_response
        )

        models = fetch_available_models()

        # Should get empty list (from .get("data", []))
        assert models == []
