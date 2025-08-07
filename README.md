# llm-poe

Plugin for [LLM](https://llm.datasette.io/) adding support for Poe API models. This plugin dynamically fetches all available models from Poe's API, ensuring you always have access to the latest models.

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

Run a prompt:

```bash
llm -m poe/gpt_4o "Hello, how are you?"
```

Use streaming:

```bash
llm -m poe/claude_sonnet_4 "Write a haiku about programming" --stream
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

You can set temperature and max_tokens:

```bash
llm -m poe/claude_sonnet_4 "Tell me a story" -o temperature 0.7 -o max_tokens 500
```

## Development

To set up for development:

```bash
git clone https://github.com/mrf/llm-poe
cd llm-poe
pip install -e .
```

## License

Apache-2.0

---
*This plugin was created with assistance from Claude.*