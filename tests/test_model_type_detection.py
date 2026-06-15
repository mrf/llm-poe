"""Tests for model type detection."""
import pytest
from llm_poe import get_model_type, detect_model_type_dynamically


class TestGetModelType:
    """Test the get_model_type function."""

    # Text/Chat Models
    def test_detect_text_model_gpt(self):
        """Test detection of GPT models as text."""
        assert get_model_type("GPT-4o") == "text"
        assert get_model_type("GPT-4-Turbo") == "text"

    def test_detect_text_model_claude(self):
        """Test detection of Claude models as text."""
        assert get_model_type("Claude-Sonnet-4") == "text"
        assert get_model_type("Claude-Opus-3") == "text"

    def test_detect_text_model_gemini(self):
        """Test detection of Gemini models as text."""
        assert get_model_type("Gemini-2.5-Pro") == "text"
        assert get_model_type("Gemini-Flash") == "text"

    def test_detect_text_model_llama(self):
        """Test detection of Llama models as text."""
        assert get_model_type("Llama-3.1-405B") == "text"
        assert get_model_type("Llama-70B") == "text"

    # Image Generation Models
    def test_detect_image_model_flux(self):
        """Test detection of Flux models as image."""
        assert get_model_type("Flux-Pro-1.1-Ultra") == "image"
        assert get_model_type("FLUX-Dev") == "image"

    def test_detect_image_model_imagen(self):
        """Test detection of Imagen models as image."""
        assert get_model_type("Imagen-4") == "image"
        assert get_model_type("Google-Imagen-3") == "image"

    def test_detect_image_model_dall_e(self):
        """Test detection of DALL-E models as image."""
        assert get_model_type("DALL-E-3") == "image"
        assert get_model_type("dall-e-2") == "image"

    def test_detect_image_model_stable_diffusion(self):
        """Test detection of Stable Diffusion models as image."""
        assert get_model_type("Stable-Diffusion-XL") == "image"
        assert get_model_type("stable-diffusion-3") == "image"

    def test_detect_image_model_midjourney(self):
        """Test detection of Midjourney models as image."""
        assert get_model_type("Midjourney-v6") == "image"
        assert get_model_type("midjourney-alpha") == "image"

    def test_detect_image_model_ideogram(self):
        """Test detection of Ideogram models as image."""
        assert get_model_type("Ideogram-2.0") == "image"
        assert get_model_type("ideogram-turbo") == "image"

    def test_detect_image_model_by_keyword(self):
        """Test detection using image-related keywords."""
        assert get_model_type("Image-Generator-Pro") == "image"
        assert get_model_type("Art-Creator-3000") == "image"
        assert get_model_type("Picture-Draw-AI") == "image"

    # Video Generation Models
    def test_detect_video_model_sora(self):
        """Test detection of Sora as video."""
        assert get_model_type("Sora") == "video"
        assert get_model_type("Sora-Turbo") == "video"

    def test_detect_video_model_runway(self):
        """Test detection of Runway models as video."""
        assert get_model_type("Runway-Gen-4-Turbo") == "video"
        assert get_model_type("runway-gen-3") == "video"

    def test_detect_video_model_veo(self):
        """Test detection of Veo models as video."""
        assert get_model_type("Veo-3") == "video"
        assert get_model_type("Google-Veo-2") == "video"

    def test_detect_video_model_kling(self):
        """Test detection of Kling models as video."""
        assert get_model_type("Kling-2.1") == "video"
        assert get_model_type("kling-pro") == "video"

    def test_detect_video_model_pika(self):
        """Test detection of Pika models as video."""
        assert get_model_type("Pika-1.5") == "video"
        assert get_model_type("pika-labs") == "video"

    def test_detect_video_model_by_keyword(self):
        """Test detection using video-related keywords."""
        assert get_model_type("Video-Generator-AI") == "video"
        assert get_model_type("Motion-Creator") == "video"
        assert get_model_type("Film-Maker-Pro") == "video"

    # Audio/TTS Models
    def test_detect_audio_model_elevenlabs(self):
        """Test detection of ElevenLabs models as audio."""
        assert get_model_type("ElevenLabs-v3") == "audio"
        assert get_model_type("elevenlabs-turbo") == "audio"

    def test_detect_audio_model_cartesia(self):
        """Test detection of Cartesia models as audio."""
        assert get_model_type("Cartesia-Sonic") == "audio"
        assert get_model_type("cartesia-voice") == "audio"

    def test_detect_audio_model_playai(self):
        """Test detection of PlayAI models as audio."""
        assert get_model_type("PlayAI-3.0") == "audio"
        assert get_model_type("playai-tts") == "audio"

    def test_detect_audio_model_by_keyword(self):
        """Test detection using audio-related keywords."""
        assert get_model_type("TTS-Engine-Pro") == "audio"
        assert get_model_type("Voice-Generator") == "audio"
        assert get_model_type("Speech-Synthesis-AI") == "audio"
        assert get_model_type("Audio-Creator") == "audio"
        assert get_model_type("Music-Generator") == "audio"

    # Edge Cases
    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        assert get_model_type("FLUX-PRO") == "image"
        assert get_model_type("flux-pro") == "image"
        assert get_model_type("Flux-Pro") == "image"

    def test_partial_keyword_matching(self):
        """Test that keywords are matched anywhere in the name."""
        assert get_model_type("Super-Flux-Mega") == "image"
        assert get_model_type("Ultra-Video-Gen") == "video"
        assert get_model_type("Best-TTS-Ever") == "audio"

    def test_ambiguous_models_default_to_text(self):
        """Test that unknown/ambiguous models default to text."""
        assert get_model_type("Unknown-Model-XYZ") == "text"
        assert get_model_type("Random-AI-123") == "text"
        assert get_model_type("Mystery-Bot") == "text"

    def test_priority_of_detection(self):
        """Test that first matching category wins (audio > video > image > text)."""
        # If a model name has multiple indicators, audio should take priority
        assert get_model_type("Audio-Image-Generator") == "audio"
        # Video should take priority over image
        assert get_model_type("Video-Image-Creator") == "video"


class TestGetModelTypeFromArchitecture:
    """Architecture output_modalities are the source of truth for type."""

    def test_image_output_classifies_as_image(self):
        """A model that outputs images is an image model regardless of name."""
        assert get_model_type(
            {"id": "nano-banana-pro",
             "architecture": {"input_modalities": ["text", "image"],
                              "output_modalities": ["image"]}}
        ) == "image"

    def test_text_input_image_output_still_image(self):
        """nano-banana-2 (text in, image out) is an image model."""
        assert get_model_type(
            {"id": "nano-banana-2",
             "architecture": {"input_modalities": ["text"],
                              "output_modalities": ["image"]}}
        ) == "image"

    def test_video_output_classifies_as_video(self):
        assert get_model_type(
            {"id": "some-model", "architecture": {"output_modalities": ["video"]}}
        ) == "video"

    def test_audio_output_classifies_as_audio(self):
        assert get_model_type(
            {"id": "some-model", "architecture": {"output_modalities": ["audio"]}}
        ) == "audio"

    def test_text_output_classifies_as_text(self):
        """A vision model that outputs text is a text model, not image."""
        assert get_model_type(
            {"id": "claude-opus-4.8",
             "architecture": {"input_modalities": ["text", "image"],
                              "output_modalities": ["text"]}}
        ) == "text"

    def test_architecture_overrides_misleading_name(self):
        """seedream (matches 'dream' video keyword) is image by architecture."""
        assert get_model_type(
            {"id": "seedream-5.0-lite",
             "architecture": {"input_modalities": ["text"],
                              "output_modalities": ["image"]}}
        ) == "image"

    def test_missing_architecture_falls_back_to_name(self):
        """No architecture signal → name heuristic."""
        assert get_model_type({"id": "Flux-Pro"}) == "image"
        assert get_model_type({"id": "GPT-4o"}) == "text"

    def test_empty_output_modalities_falls_back_to_name(self):
        """Empty output_modalities → name heuristic."""
        assert get_model_type(
            {"id": "Flux-Pro", "architecture": {"output_modalities": []}}
        ) == "image"


class TestDetectModelTypeDynamically:
    """Test the detect_model_type_dynamically function."""

    def test_dynamic_detection_text_model(self, httpx_mock, mock_api_key):
        """Test dynamic detection for text model."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json={
                "choices": [{
                    "message": {"content": "This is a text response"}
                }]
            }
        )

        model_type = detect_model_type_dynamically("GPT-4o", "test-api-key-12345")
        assert model_type == "text"

    def test_dynamic_detection_image_model(self, httpx_mock, mock_api_key):
        """Test dynamic detection for image model."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json={
                "choices": [{
                    "message": {"content": "https://example.com/image.png"}
                }]
            }
        )

        model_type = detect_model_type_dynamically("Flux-Pro", "test-api-key-12345")
        assert model_type == "image"

    def test_dynamic_detection_video_model(self, httpx_mock, mock_api_key):
        """Test dynamic detection for video model."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json={
                "choices": [{
                    "message": {"content": "https://example.com/video.mp4"}
                }]
            }
        )

        model_type = detect_model_type_dynamically("Sora", "test-api-key-12345")
        assert model_type == "video"

    def test_dynamic_detection_audio_model(self, httpx_mock, mock_api_key):
        """Test dynamic detection for audio model."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            json={
                "choices": [{
                    "message": {"content": "https://example.com/audio.mp3"}
                }]
            }
        )

        model_type = detect_model_type_dynamically("ElevenLabs", "test-api-key-12345")
        assert model_type == "audio"

    def test_dynamic_detection_fallback_on_error(self, httpx_mock, mock_api_key):
        """Test fallback to name-based detection on API error."""
        httpx_mock.add_response(
            url="https://api.poe.com/v1/chat/completions",
            status_code=500
        )

        # Should fall back to name-based detection
        model_type = detect_model_type_dynamically("Flux-Pro", "test-api-key-12345")
        assert model_type == "image"

    def test_dynamic_detection_fallback_on_timeout(self, httpx_mock, mock_api_key):
        """Test fallback to name-based detection on timeout."""
        import httpx
        httpx_mock.add_exception(
            httpx.TimeoutException("Connection timeout")
        )

        # Should fall back to name-based detection
        model_type = detect_model_type_dynamically("Sora", "test-api-key-12345")
        assert model_type == "video"
