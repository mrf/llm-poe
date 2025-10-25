# llm-poe Test Suite

Comprehensive test suite for the llm-poe plugin with 94% code coverage.

## Quick Start

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=llm_poe --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/test_text_models.py

# Run specific test class
pytest tests/test_text_models.py::TestPoeModel

# Run specific test
pytest tests/test_text_models.py::TestPoeModel::test_streaming_response
```

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and test configuration
├── test_api_key.py                  # API key management tests (6 tests)
├── test_model_registration.py      # Model registration tests (11 tests)
├── test_model_type_detection.py    # Model type detection tests (31 tests)
├── test_text_models.py              # Text/chat model tests (15 tests)
├── test_image_models.py             # Image generation tests (12 tests)
├── test_video_models.py             # Video generation tests (13 tests)
├── test_audio_models.py             # Audio/TTS tests (14 tests)
├── test_error_handling.py           # Error handling tests (20 tests)
└── test_caching.py                  # Caching mechanism tests (14 tests)
```

## Test Coverage

**Total: 136 tests, 94% code coverage**

### Unit Tests (48 tests)
- **API Key Management** (6 tests)
  - Key retrieval from llm store
  - Missing key error handling
  - Optional key retrieval

- **Model Registration** (11 tests)
  - Dynamic model fetching
  - Cache initialization and reuse
  - Fallback on API errors
  - Model ID sanitization
  - Model name preservation

- **Model Type Detection** (31 tests)
  - Text/chat models (GPT, Claude, Gemini, Llama)
  - Image models (Flux, Imagen, DALL-E, Stable Diffusion, Midjourney)
  - Video models (Sora, Runway, Veo, Kling, Pika)
  - Audio models (ElevenLabs, Cartesia, PlayAI)
  - Dynamic detection with API test requests
  - Edge cases and priority rules

### Integration Tests (68 tests)
- **Text/Chat Models** (15 tests)
  - Streaming and non-streaming responses
  - Conversation history handling
  - Temperature and max_tokens options
  - Request/response validation

- **Image Generation Models** (12 tests)
  - Image generation with prompts
  - Size and quality options
  - Forced non-streaming
  - Extended timeouts

- **Video Generation Models** (13 tests)
  - Video generation with prompts
  - Duration and aspect ratio options
  - Various aspect ratios (16:9, 9:16, 1:1, etc.)
  - Extended timeouts

- **Audio/TTS Models** (14 tests)
  - Audio generation (streaming and non-streaming)
  - Voice selection options
  - Speed control
  - Various voice options

- **Error Handling** (14 tests)
  - HTTP errors (401, 404, 429, 500, 503)
  - Network timeouts and connection errors
  - Malformed API responses
  - Missing/invalid parameters

### Performance Tests (20 tests)
- **Caching** (14 tests)
  - Cache initialization and population
  - Cache reuse within duration
  - Cache expiration after timeout
  - Fallback when cache unavailable
  - Concurrent access handling

- **Edge Cases** (6 tests)
  - Empty responses
  - Malformed data
  - Missing fields
  - Invalid JSON

## Fixtures

The `conftest.py` file provides shared fixtures:

- **API Key Mocking**: `mock_api_key`, `mock_api_key_missing`
- **API Responses**: `sample_models_response`, `sample_chat_completion_response`, `sample_streaming_chunks`, etc.
- **Cache Management**: `clear_model_cache`
- **Mock Objects**: `mock_prompt`, `mock_image_prompt`, `mock_video_prompt`, `mock_audio_prompt`
- **Conversation Mocking**: `mock_conversation`, `mock_empty_conversation`

## Running Specific Test Categories

```bash
# Unit tests only
pytest tests/test_api_key.py tests/test_model_registration.py tests/test_model_type_detection.py

# Integration tests only
pytest tests/test_text_models.py tests/test_image_models.py tests/test_video_models.py tests/test_audio_models.py

# Error handling tests
pytest tests/test_error_handling.py

# Performance tests
pytest tests/test_caching.py
```

## Coverage Reports

After running tests with coverage, view detailed HTML reports:

```bash
# Generate HTML coverage report
pytest --cov=llm_poe --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## CI/CD

Tests run automatically on:
- Every push to main branch
- Every pull request
- Multiple Python versions (3.8, 3.9, 3.10, 3.11, 3.12)
- Multiple platforms (Ubuntu, macOS, Windows)

See `.github/workflows/test.yml` for CI configuration.

## Writing New Tests

When adding new features, follow these patterns:

1. **Create descriptive test classes and methods**
   ```python
   class TestNewFeature:
       def test_basic_functionality(self):
           """Test basic functionality of new feature."""
           pass
   ```

2. **Use existing fixtures from conftest.py**
   ```python
   def test_with_api_key(self, mock_api_key, httpx_mock):
       # Your test here
       pass
   ```

3. **Mock HTTP requests with httpx_mock**
   ```python
   httpx_mock.add_response(
       url="https://api.poe.com/v1/endpoint",
       json={"data": "response"}
   )
   ```

4. **Test both success and failure cases**
   - Happy path
   - Error conditions
   - Edge cases
   - Invalid input

5. **Aim for high coverage**
   - Target 90%+ coverage for new code
   - Test all code paths
   - Include edge cases

## Troubleshooting

**Import errors**: Make sure package is installed in editable mode:
```bash
pip install -e ".[test]"
```

**Coverage warnings**: The "module-not-measured" warning can be ignored if tests pass and coverage is reported.

**Test isolation**: Each test should be independent. Use `clear_model_cache` fixture to reset state.

**Mock not working**: Ensure httpx_mock is used correctly and URL matches exactly.
