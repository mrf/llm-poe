import llm
import httpx
from typing import Optional, Dict, Any, AsyncIterator, List
from pydantic import Field
import json
import time
import os


POE_API_BASE = "https://api.poe.com/v1"

# Cache for model list to avoid frequent API calls
_model_cache = None
_cache_timestamp = None
_cache_duration = 3600  # Cache for 1 hour


def get_api_key() -> str:
    """Get Poe API key from environment or LLM key store."""
    api_key = llm.get_key("", "poe", "POE_API_KEY")
    if not api_key:
        raise llm.ModelError("POE_API_KEY environment variable is not set")
    return api_key


def get_api_key_optional() -> str:
    """Get Poe API key from environment or LLM key store, return None if not found."""
    try:
        return llm.get_key("", "poe", "POE_API_KEY")
    except:
        return None


def fetch_available_models() -> List[Dict[str, Any]]:
    """Fetch available models from Poe API with caching."""
    global _model_cache, _cache_timestamp
    
    # Check if cache is still valid
    current_time = time.time()
    if (_model_cache is not None and 
        _cache_timestamp is not None and 
        current_time - _cache_timestamp < _cache_duration):
        return _model_cache
    
    # Try to get API key, but don't fail if it's not available during registration
    api_key = get_api_key_optional()
    if not api_key:
        # Return fallback models if no API key available
        fallback_models = [
            {"id": "GPT-4o", "object": "model"},
            {"id": "Claude-Sonnet-4", "object": "model"}, 
            {"id": "Gemini-2.5-Pro", "object": "model"},
            {"id": "Llama-3.1-405B", "object": "model"},
            {"id": "Grok-4", "object": "model"}
        ]
        return fallback_models
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        with httpx.Client() as client:
            response = client.get(
                f"{POE_API_BASE}/models",
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract model data
            models = data.get("data", [])
            
            # Update cache
            _model_cache = models
            _cache_timestamp = current_time
            
            return models
    
    except Exception as e:
        # If API call fails, return a fallback list of common models
        fallback_models = [
            {"id": "GPT-4o", "object": "model"},
            {"id": "Claude-Sonnet-4", "object": "model"}, 
            {"id": "Gemini-2.5-Pro", "object": "model"},
            {"id": "Llama-3.1-405B", "object": "model"},
            {"id": "Grok-4", "object": "model"}
        ]
        print(f"Warning: Failed to fetch models from Poe API ({e}), using fallback list")
        return fallback_models


class PoeOptions(llm.Options):
    temperature: Optional[float] = Field(
        default=1.0, 
        description="Sampling temperature (0-2)", 
        ge=0, 
        le=2
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate",
        gt=0
    )


class PoeModel(llm.Model):
    can_stream = True

    class Options(PoeOptions):
        pass

    def __init__(self, model_id: str, model_name: str):
        self.model_id = model_id
        self.model_name = model_name

    def __str__(self):
        # Display cleaner name in model listings
        return f"Poe: {self.model_id.replace('poe/', '')}"

    def execute(self, prompt, stream, response, conversation):
        api_key = get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if conversation:
            for prev_response in conversation.responses:
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})
        
        messages.append({"role": "user", "content": prompt.prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream
        }

        if prompt.options.temperature is not None:
            payload["temperature"] = prompt.options.temperature
        if prompt.options.max_tokens is not None:
            payload["max_tokens"] = prompt.options.max_tokens

        with httpx.Client() as client:
            if stream:
                with client.stream(
                    "POST", 
                    f"{POE_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                ) as stream_response:
                    stream_response.raise_for_status()
                    for line in stream_response.iter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
            else:
                api_response = client.post(
                    f"{POE_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                api_response.raise_for_status()
                data = api_response.json()
                
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    yield content

                response.response_json = data


@llm.hookimpl
def register_models(register):
    """Register all available Poe models dynamically."""
    try:
        models = fetch_available_models()
        for model_data in models:
            model_name = model_data.get("id", "")
            if model_name:
                # Create a clean model_id for LLM
                model_id = f"poe/{model_name.lower().replace('-', '_').replace('.', '_').replace(' ', '_')}"
                register(PoeModel(model_id, model_name))
    except Exception as e:
        # Register fallback models if dynamic loading fails
        fallback_models = ["GPT-4o", "Claude-Sonnet-4", "Gemini-2.5-Pro", "Llama-3.1-405B", "Grok-4"]
        for model_name in fallback_models:
            model_id = f"poe/{model_name.lower().replace('-', '_')}"
            register(PoeModel(model_id, model_name))