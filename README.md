# llm-poe

[![Tests](https://github.com/mrf/llm-poe/actions/workflows/test.yml/badge.svg)](https://github.com/mrf/llm-poe/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/mrf/llm-poe/branch/main/graph/badge.svg)](https://codecov.io/gh/mrf/llm-poe)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for Poe API models. This plugin dynamically fetches all available models from Poe's API, ensuring you always have access to the latest models.

**✅ Full Multi-Modal Support:** Works with text, image generation, video generation, and audio/TTS models with optimized handling for each content type.

**✅ 94% Test Coverage:** Comprehensive test suite with 136 automated tests ensuring reliability.

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-poe
```

Or install from source for development:

```bash
llm install -e .
```

**Important:** Use `llm install -e .` (not `pip install -e .`) for local development to ensure proper plugin discovery.

## Configuration

You need to set your Poe API key. Get your API key from https://poe.com/api_key

Set the key using:

```bash
llm keys set poe
```

Or set the `POE_API_KEY` environment variable:

```bash
export POE_API_KEY="your-api-key-here"
```

## Usage

List available models:

```bash
llm models list | grep poe/
```

The plugin automatically fetches all available models from Poe's API. The model list is cached for 1 hour to improve performance. Over **240+ models** are currently available including the latest AI models from OpenAI, Anthropic, Google, Meta, and more.

Run prompts with different model types:

```bash
# Text generation
llm -m poe/gpt_4o "Hello, how are you?"

# Image generation
llm -m poe/flux_pro_1_1_ultra "Generate a cat image" --no-stream

# Video generation  
llm -m poe/sora "Create a short video of a sunset" --no-stream

# Audio/TTS
llm -m poe/elevenlabs_v3 "Say hello world" --no-stream
```

## Available Models

The plugin dynamically fetches all models available through Poe's API, including:

**Latest Text/Code Models:**
- GPT-4.1, O3-Pro, O4-Mini, O3-Mini
- Claude-Sonnet-4, Claude-Opus-4.1
- Gemini-2.5-Pro, Gemini-2.5-Flash
- DeepSeek-R1, DeepSeek-V3
- Grok-4, Grok-3
- Llama-4-Scout, Llama-4-Maverick
- Qwen-3-235B, Mistral-Small-3.2

**Image Generation Models:**
- Imagen-4-Ultra, Flux-Pro-1.1
- DALL-E-3, Ideogram-v3
- Phoenix-1.0, Recraft-v3
- Flux-Kontext-Max

**Video Generation Models:**
- Veo-3, Sora, Runway-Gen-4-Turbo
- Kling-2.1, Hailuo-02
- Dream-Machine, Pika

**Audio/TTS Models:**
- ElevenLabs, Cartesia
- PlayAI-TTS, Orpheus-TTS
- Lyria, Hailuo-Speech

**Search-Enhanced Models:**
- GPT-4o-Search, Claude-Sonnet-4-Search
- Perplexity-Sonar-Pro, Web-Search

And **200+ more models**! Use `llm models list | grep poe/` to see the complete current list.

## Options

**Text Models:**
```bash
llm -m poe/claude_sonnet_4 "Tell me a story" -o temperature 0.7 -o max_tokens 500
```

**Image Models:**
```bash
llm -m poe/flux_pro_1_1_ultra "Generate landscape" -o size 1024x1024 -o quality standard --no-stream
```

**Video Models:**
```bash
llm -m poe/sora "Create a nature scene" -o duration 10 -o aspect_ratio 16:9 --no-stream
```

**Audio Models:**
```bash
llm -m poe/elevenlabs_v3 "Read this text" -o voice alloy -o speed 1.0 --no-stream
```

## Development

To set up for development:

```bash
git clone https://github.com/mrf/llm-poe
cd llm-poe
llm install -e .
```

**Important:** Use `llm install -e .` (not `pip install -e .`) for local development to ensure proper plugin discovery.

## Testing

This plugin has a comprehensive test suite with 94% code coverage.

### Running Tests

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=llm_poe --cov-report=term-missing --cov-report=html

# View HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Test Coverage

- **136 automated tests** covering all functionality
- **94% code coverage** across the entire plugin
- **Multi-platform testing** (Ubuntu, macOS, Windows)
- **Python 3.8-3.12 compatibility** verified

Test categories:
- Unit tests for model registration, API keys, and type detection
- Integration tests for all model types (text, image, video, audio)
- Error handling and edge case tests
- Performance and caching tests

See [tests/README.md](tests/README.md) for detailed testing documentation.

## License

Apache-2.0

---
*This plugin was created with assistance from Claude.*
