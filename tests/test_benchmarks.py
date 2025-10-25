"""Performance benchmarks for llm-poe plugin.

These benchmarks measure key performance metrics including:
- Model list fetching time
- Cache performance (hit vs miss)
- API response time simulation
- Model registration time
"""
import pytest
import time
from unittest.mock import Mock
import llm_poe
from llm_poe import (
    fetch_available_models,
    register_models,
    get_model_type,
    PoeModel,
    PoeImageModel,
    PoeVideoModel,
    PoeAudioModel,
)


class TestModelFetchingPerformance:
    """Benchmark model fetching operations."""

    def test_benchmark_fetch_models_cold(self, benchmark, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Benchmark cold fetch of model list from API."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # Clear cache to ensure cold fetch
        llm_poe._model_cache = None
        llm_poe._cache_timestamp = None

        result = benchmark(fetch_available_models)
        assert len(result) == 6

    def test_benchmark_fetch_models_cached(self, benchmark, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Benchmark cached model list retrieval (cache hit)."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        # Warm up the cache
        fetch_available_models()

        # Now benchmark cached retrieval
        result = benchmark(fetch_available_models)
        assert len(result) == 6

    def test_benchmark_fetch_models_fallback(self, benchmark, mock_api_key_missing, clear_model_cache):
        """Benchmark fallback model list retrieval (no API key)."""
        result = benchmark(fetch_available_models)
        assert len(result) == 5  # FALLBACK_MODELS count


class TestCachePerformance:
    """Benchmark cache-related operations."""

    @pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
    def test_benchmark_cache_miss(self, benchmark, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Benchmark cache miss scenario with fresh API call."""
        def fetch_with_cache_clear():
            # Clear cache before each iteration
            llm_poe._model_cache = None
            llm_poe._cache_timestamp = None

            httpx_mock.add_response(
                url="https://api.poe.com/v1/models",
                json=sample_models_response
            )
            return fetch_available_models()

        result = benchmark(fetch_with_cache_clear)
        assert len(result) == 6

    def test_benchmark_cache_hit_performance(self, benchmark, mock_api_key, sample_models_response, clear_model_cache):
        """Benchmark pure cache hit performance (no network)."""
        # Pre-populate cache for fastest possible retrieval
        llm_poe._model_cache = sample_models_response["data"]
        llm_poe._cache_timestamp = time.time()

        result = benchmark(fetch_available_models)
        assert len(result) == 6

    def test_benchmark_cache_expiration_check(self, benchmark, mock_api_key, sample_models_response, clear_model_cache):
        """Benchmark cache expiration validation logic."""
        # Set cache with recent timestamp (should not expire)
        llm_poe._model_cache = sample_models_response["data"]
        llm_poe._cache_timestamp = time.time() - 100  # 100 seconds ago (still valid)

        def check_cache_validity():
            return fetch_available_models()

        result = benchmark(check_cache_validity)
        assert len(result) == 6


class TestModelTypeDetectionPerformance:
    """Benchmark model type detection operations."""

    def test_benchmark_get_model_type_text(self, benchmark):
        """Benchmark model type detection for text models."""
        result = benchmark(get_model_type, "GPT-4o")
        assert result == "text"

    def test_benchmark_get_model_type_image(self, benchmark):
        """Benchmark model type detection for image models."""
        result = benchmark(get_model_type, "Flux-Pro-1.1-Ultra")
        assert result == "image"

    def test_benchmark_get_model_type_video(self, benchmark):
        """Benchmark model type detection for video models."""
        result = benchmark(get_model_type, "Sora")
        assert result == "video"

    def test_benchmark_get_model_type_audio(self, benchmark):
        """Benchmark model type detection for audio models."""
        result = benchmark(get_model_type, "ElevenLabs-v3")
        assert result == "audio"

    def test_benchmark_multiple_model_types(self, benchmark):
        """Benchmark type detection across multiple model names."""
        model_names = [
            "GPT-4o", "Claude-Sonnet-4", "Gemini-2.5-Pro",
            "Flux-Pro", "DALL-E-3", "Stable-Diffusion",
            "Sora", "Runway-Gen-3", "Veo-2",
            "ElevenLabs-v3", "PlayAI", "Cartesia"
        ]

        def detect_all_types():
            return [get_model_type(name) for name in model_names]

        results = benchmark(detect_all_types)
        assert len(results) == 12


class TestModelRegistrationPerformance:
    """Benchmark model registration operations."""

    @pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
    def test_benchmark_register_models_success(self, benchmark, httpx_mock, mock_api_key, sample_models_response, clear_model_cache):
        """Benchmark successful model registration with API fetch."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/models",
            json=sample_models_response
        )

        def register_all_models():
            registered = []

            def mock_register(model):
                registered.append(model)

            # Clear cache to ensure fresh fetch
            llm_poe._model_cache = None
            llm_poe._cache_timestamp = None

            # Add new mock response for each iteration
            httpx_mock.add_response(
                url="https://api.poe.com/v1/models",
                json=sample_models_response
            )

            register_models(mock_register)
            return registered

        result = benchmark(register_all_models)
        assert len(result) == 6

    def test_benchmark_register_models_cached(self, benchmark, mock_api_key, sample_models_response, clear_model_cache):
        """Benchmark model registration with warm cache."""
        # Pre-populate cache
        llm_poe._model_cache = sample_models_response["data"]
        llm_poe._cache_timestamp = time.time()

        def register_all_models():
            registered = []

            def mock_register(model):
                registered.append(model)

            register_models(mock_register)
            return registered

        result = benchmark(register_all_models)
        assert len(result) == 6

    def test_benchmark_register_models_fallback(self, benchmark, mock_api_key_missing, clear_model_cache):
        """Benchmark model registration using fallback models."""
        def register_fallback_models():
            registered = []

            def mock_register(model):
                registered.append(model)

            register_models(mock_register)
            return registered

        result = benchmark(register_fallback_models)
        assert len(result) == 5


class TestModelInstantiationPerformance:
    """Benchmark model class instantiation."""

    def test_benchmark_text_model_creation(self, benchmark):
        """Benchmark PoeModel instantiation."""
        def create_text_model():
            return PoeModel("poe/gpt_4o", "GPT-4o")

        model = benchmark(create_text_model)
        assert model.model_id == "poe/gpt_4o"

    def test_benchmark_image_model_creation(self, benchmark):
        """Benchmark PoeImageModel instantiation."""
        def create_image_model():
            return PoeImageModel("poe/flux_pro", "Flux-Pro")

        model = benchmark(create_image_model)
        assert model.model_id == "poe/flux_pro"

    def test_benchmark_video_model_creation(self, benchmark):
        """Benchmark PoeVideoModel instantiation."""
        def create_video_model():
            return PoeVideoModel("poe/sora", "Sora")

        model = benchmark(create_video_model)
        assert model.model_id == "poe/sora"

    def test_benchmark_audio_model_creation(self, benchmark):
        """Benchmark PoeAudioModel instantiation."""
        def create_audio_model():
            return PoeAudioModel("poe/elevenlabs", "ElevenLabs-v3")

        model = benchmark(create_audio_model)
        assert model.model_id == "poe/elevenlabs"

    def test_benchmark_batch_model_creation(self, benchmark):
        """Benchmark creating multiple model instances."""
        def create_multiple_models():
            models = []
            for i in range(10):
                models.append(PoeModel(f"poe/model_{i}", f"Model-{i}"))
                models.append(PoeImageModel(f"poe/image_{i}", f"Image-{i}"))
                models.append(PoeVideoModel(f"poe/video_{i}", f"Video-{i}"))
                models.append(PoeAudioModel(f"poe/audio_{i}", f"Audio-{i}"))
            return models

        models = benchmark(create_multiple_models)
        assert len(models) == 40


class TestAPIResponseSimulation:
    """Benchmark simulated API response handling."""

    @pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
    def test_benchmark_model_execute_nonstreaming(self, benchmark, httpx_mock, mock_api_key, sample_chat_completion_response):
        """Benchmark non-streaming model execution."""
        model = PoeModel("poe/gpt_4o", "GPT-4o")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_chat_completion_response
        )

        prompt = Mock()
        prompt.prompt = "Test prompt"
        prompt.options = Mock()
        prompt.options.temperature = 1.0
        prompt.options.max_tokens = 100

        response = Mock()
        response.response_json = None

        def execute_model():
            # Add response for each iteration
            httpx_mock.add_response(
                url="https://api.poe.com/v1/chat/completions",
                json=sample_chat_completion_response
            )
            return list(model.execute(prompt, False, response, None))

        result = benchmark(execute_model)
        assert len(result) == 1

    @pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
    def test_benchmark_image_model_execute(self, benchmark, httpx_mock, mock_api_key, sample_image_response):
        """Benchmark image model execution."""
        model = PoeImageModel("poe/flux_pro", "Flux-Pro")

        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json=sample_image_response
        )

        prompt = Mock()
        prompt.prompt = "A beautiful sunset"
        prompt.options = Mock()
        prompt.options.temperature = 1.0
        prompt.options.max_tokens = None
        prompt.options.size = "1024x1024"
        prompt.options.quality = "standard"

        response = Mock()
        response.response_json = None

        def execute_model():
            httpx_mock.add_response(
                url="https://api.poe.com/v1/chat/completions",
                json=sample_image_response
            )
            return list(model.execute(prompt, False, response, None))

        result = benchmark(execute_model)
        assert len(result) == 1


class TestModelIDSanitization:
    """Benchmark model ID sanitization operations."""

    def test_benchmark_model_id_sanitization(self, benchmark):
        """Benchmark model ID creation and sanitization."""
        test_names = [
            "GPT-4o", "Claude-Sonnet-4", "Gemini-2.5-Pro",
            "Model.With.Dots", "Model With Spaces", "Model_With_Underscores",
            "DALL-E-3", "Stable-Diffusion-XL", "Flux.Pro.1.1.Ultra"
        ]

        def sanitize_ids():
            sanitized = []
            for name in test_names:
                model_id = f"poe/{name.lower().replace('-', '_').replace('.', '_').replace(' ', '_')}"
                sanitized.append(model_id)
            return sanitized

        results = benchmark(sanitize_ids)
        assert len(results) == len(test_names)
