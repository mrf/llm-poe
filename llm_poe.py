import llm
import httpx
from typing import Optional, Dict, Any, AsyncIterator, List
from pydantic import Field
import json
import time
import os


POE_API_BASE = "https://api.poe.com/v1"

# Fallback models to use when API is unavailable
FALLBACK_MODELS = [
    {"id": "GPT-4o", "object": "model"},
    {"id": "Claude-Sonnet-4", "object": "model"}, 
    {"id": "Gemini-2.5-Pro", "object": "model"},
    {"id": "Llama-3.1-405B", "object": "model"},
    {"id": "Grok-4", "object": "model"}
]

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
        return FALLBACK_MODELS
    
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
        print(f"Warning: Failed to fetch models from Poe API ({e}), using fallback list")
        return FALLBACK_MODELS


def get_model_type(model_name: str) -> str:
    """Determine model type from name patterns and known model families."""
    name_lower = model_name.lower()
    
    # Image generation models
    image_indicators = [
        'imagen', 'dall', 'flux', 'ideogram', 'recraft', 'phoenix', 'midjourney',
        'stable', 'diffusion', 'draw', 'generate', 'create', 'art', 'picture', 'image'
    ]
    
    # Video generation models  
    video_indicators = [
        'veo', 'sora', 'runway', 'kling', 'hailuo', 'dream', 'pika', 'video',
        'motion', 'animate', 'film', 'movie'
    ]
    
    # Audio/TTS models
    audio_indicators = [
        'elevenlabs', 'cartesia', 'playai', 'orpheus', 'lyria', 'tts', 'speech',
        'voice', 'audio', 'speak', 'sound', 'music'
    ]
    
    # Check for type indicators (check more specific ones first)
    # Check audio first as it has more specific names
    if any(indicator in name_lower for indicator in audio_indicators):
        return 'audio'
    elif any(indicator in name_lower for indicator in video_indicators):
        return 'video'
    elif any(indicator in name_lower for indicator in image_indicators):
        return 'image'
    
    # Default to text for traditional LLMs
    return 'text'


def detect_model_type_dynamically(model_name: str, api_key: str) -> str:
    """
    Attempt to detect model type by making a test request.
    Falls back to name-based detection if test fails.
    """
    try:
        # Make a simple test request to see response format
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        test_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }
        
        with httpx.Client() as client:
            response = client.post(
                f"{POE_API_BASE}/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                # Analyze response to determine type
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    # If response contains URLs, likely media generation
                    if any(ext in content.lower() for ext in ['.jpg', '.png', '.gif', '.mp4', '.wav', '.mp3', 'http']):
                        # Could be image, video, or audio - refine detection
                        if any(ext in content.lower() for ext in ['.jpg', '.png', '.gif']):
                            return 'image'
                        elif any(ext in content.lower() for ext in ['.mp4', '.mov', '.avi']):
                            return 'video'
                        elif any(ext in content.lower() for ext in ['.wav', '.mp3', '.m4a']):
                            return 'audio'
                        return 'image'  # Default media assumption
                    return 'text'
    except Exception:
        pass
    
    # Fall back to name-based detection
    return get_model_type(model_name)


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


class PoeImageOptions(PoeOptions):
    size: Optional[str] = Field(default="1024x1024", description="Image size")
    quality: Optional[str] = Field(default="standard", description="Image quality")


class PoeVideoOptions(PoeOptions):
    duration: Optional[int] = Field(default=10, description="Video duration in seconds")
    aspect_ratio: Optional[str] = Field(default="16:9", description="Video aspect ratio")


class PoeAudioOptions(PoeOptions):
    voice: Optional[str] = Field(default="alloy", description="Voice selection")
    speed: Optional[float] = Field(default=1.0, description="Speech speed")


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


class PoeImageModel(PoeModel):
    can_stream = False  # Image generation typically doesn't stream

    class Options(PoeImageOptions):
        pass

    def execute(self, prompt, stream, response, conversation):
        api_key = get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Build messages array like standard chat models
        messages = []
        if conversation:
            for prev_response in conversation.responses:
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})
        
        # For image models, we still use the standard chat format
        # The model should understand image generation prompts
        messages.append({"role": "user", "content": prompt.prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False  # Force non-streaming for image models
        }

        # Add image-specific options as system context or model parameters
        if hasattr(prompt.options, 'size') and prompt.options.size:
            # Add size preference to the prompt context
            payload["size"] = prompt.options.size
        if hasattr(prompt.options, 'quality') and prompt.options.quality:
            payload["quality"] = prompt.options.quality

        # Add standard options
        if hasattr(prompt.options, 'temperature') and prompt.options.temperature is not None:
            payload["temperature"] = prompt.options.temperature
        if hasattr(prompt.options, 'max_tokens') and prompt.options.max_tokens is not None:
            payload["max_tokens"] = prompt.options.max_tokens

        with httpx.Client() as client:
            api_response = client.post(
                f"{POE_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0  # Longer timeout for image generation
            )
            api_response.raise_for_status()
            data = api_response.json()
            
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                yield content
            response.response_json = data


class PoeVideoModel(PoeModel):
    can_stream = False  # Video generation typically doesn't stream

    class Options(PoeVideoOptions):
        pass

    def execute(self, prompt, stream, response, conversation):
        api_key = get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Build messages array like standard chat models
        messages = []
        if conversation:
            for prev_response in conversation.responses:
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})
        
        messages.append({"role": "user", "content": prompt.prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False  # Force non-streaming for video models
        }

        # Add video-specific options
        if hasattr(prompt.options, 'duration') and prompt.options.duration:
            payload["duration"] = prompt.options.duration
        if hasattr(prompt.options, 'aspect_ratio') and prompt.options.aspect_ratio:
            payload["aspect_ratio"] = prompt.options.aspect_ratio

        # Add standard options
        if hasattr(prompt.options, 'temperature') and prompt.options.temperature is not None:
            payload["temperature"] = prompt.options.temperature
        if hasattr(prompt.options, 'max_tokens') and prompt.options.max_tokens is not None:
            payload["max_tokens"] = prompt.options.max_tokens

        with httpx.Client() as client:
            api_response = client.post(
                f"{POE_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0  # Even longer timeout for video generation
            )
            api_response.raise_for_status()
            data = api_response.json()
            
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                yield content
            response.response_json = data


class PoeAudioModel(PoeModel):
    can_stream = True  # Audio/TTS might support streaming

    class Options(PoeAudioOptions):
        pass

    def execute(self, prompt, stream, response, conversation):
        api_key = get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Build messages array like standard chat models
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

        # Add audio-specific options
        if hasattr(prompt.options, 'voice') and prompt.options.voice:
            payload["voice"] = prompt.options.voice
        if hasattr(prompt.options, 'speed') and prompt.options.speed:
            payload["speed"] = prompt.options.speed

        # Add standard options
        if hasattr(prompt.options, 'temperature') and prompt.options.temperature is not None:
            payload["temperature"] = prompt.options.temperature
        if hasattr(prompt.options, 'max_tokens') and prompt.options.max_tokens is not None:
            payload["max_tokens"] = prompt.options.max_tokens

        with httpx.Client() as client:
            if stream:
                with client.stream(
                    "POST", 
                    f"{POE_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60.0
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
                    timeout=60.0
                )
                api_response.raise_for_status()
                data = api_response.json()
                
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    yield content
                response.response_json = data


@llm.hookimpl
def register_models(register):
    """Register all available Poe models dynamically with appropriate model classes."""
    try:
        models = fetch_available_models()
        for model_data in models:
            model_name = model_data.get("id", "")
            if model_name:
                # Create a clean model_id for LLM
                model_id = f"poe/{model_name.lower().replace('-', '_').replace('.', '_').replace(' ', '_')}"
                
                # Determine model type and use appropriate class
                model_type = get_model_type(model_name)
                
                if model_type == 'image':
                    register(PoeImageModel(model_id, model_name))
                elif model_type == 'video':
                    register(PoeVideoModel(model_id, model_name))
                elif model_type == 'audio':
                    register(PoeAudioModel(model_id, model_name))
                else:
                    register(PoeModel(model_id, model_name))
    except Exception as e:
        # Register fallback models if dynamic loading fails
        for model_data in FALLBACK_MODELS:
            model_name = model_data["id"]
            model_id = f"poe/{model_name.lower().replace('-', '_')}"
            
            # Apply type detection to fallback models too
            model_type = get_model_type(model_name)
            
            if model_type == 'image':
                register(PoeImageModel(model_id, model_name))
            elif model_type == 'video':
                register(PoeVideoModel(model_id, model_name))
            elif model_type == 'audio':
                register(PoeAudioModel(model_id, model_name))
            else:
                register(PoeModel(model_id, model_name))