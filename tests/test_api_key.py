"""Tests for API key management."""
import pytest
from unittest.mock import patch
import llm
from llm_poe import get_api_key, get_api_key_optional


class TestGetApiKey:
    """Test the get_api_key function."""

    def test_get_api_key_success(self, mock_api_key):
        """Test successful API key retrieval."""
        api_key = get_api_key()
        assert api_key == "test-api-key-12345"

    def test_get_api_key_missing(self, mock_api_key_missing):
        """Test error when API key is missing."""
        with pytest.raises(llm.ModelError) as exc_info:
            get_api_key()

        assert "POE_API_KEY" in str(exc_info.value)

    def test_get_api_key_from_llm_store(self, monkeypatch):
        """Test that API key is retrieved from LLM's key store."""
        called_with = []

        def mock_get_key(*args, **kwargs):
            called_with.append((args, kwargs))
            return "my-poe-key"

        monkeypatch.setattr("llm.get_key", mock_get_key)

        api_key = get_api_key()

        assert api_key == "my-poe-key"
        # Verify it called llm.get_key with correct parameters
        assert len(called_with) == 1
        assert called_with[0][0] == ("", "poe", "POE_API_KEY")


class TestGetApiKeyOptional:
    """Test the get_api_key_optional function."""

    def test_get_api_key_optional_success(self, mock_api_key):
        """Test successful optional API key retrieval."""
        api_key = get_api_key_optional()
        assert api_key == "test-api-key-12345"

    def test_get_api_key_optional_missing(self, mock_api_key_missing):
        """Test that missing key returns None instead of raising."""
        api_key = get_api_key_optional()
        assert api_key is None

    def test_get_api_key_optional_on_exception(self, monkeypatch):
        """Test that exceptions are caught and None is returned."""
        def mock_get_key(*args, **kwargs):
            raise Exception("Some error")

        monkeypatch.setattr("llm.get_key", mock_get_key)

        api_key = get_api_key_optional()
        assert api_key is None
