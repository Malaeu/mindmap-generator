"""Factory for creating LLM provider instances."""

from typing import Any, Optional
from decouple import config

from rich.console import Console


console = Console()


def create_llm_provider(provider_name: str) -> Any:
    """Create an LLM provider instance.
    
    Args:
        provider_name: Name of the provider (OPENAI, ANTHROPIC, DEEPSEEK, GEMINI)
        
    Returns:
        Configured LLM client
        
    Raises:
        ValueError: If provider is not supported or API key is missing
    """
    provider_name = provider_name.upper()
    
    if provider_name == "OPENAI":
        return create_openai_provider()
    elif provider_name in ["ANTHROPIC", "CLAUDE"]:
        return create_anthropic_provider()
    elif provider_name == "DEEPSEEK":
        return create_deepseek_provider()
    elif provider_name == "GEMINI":
        return create_gemini_provider()
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


def create_openai_provider() -> Any:
    """Create OpenAI provider."""
    api_key = config("OPENAI_API_KEY", default="")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        console.print("[green]✓ OpenAI provider initialized[/green]")
        return client
    except ImportError:
        raise ValueError("OpenAI package not installed")


def create_anthropic_provider() -> Any:
    """Create Anthropic/Claude provider."""
    api_key = config("ANTHROPIC_API_KEY", default="")
    
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")
    
    try:
        from anthropic import AsyncAnthropic
        
        client = AsyncAnthropic(api_key=api_key)
        console.print("[green]✓ Anthropic provider initialized[/green]")
        return client
    except ImportError:
        raise ValueError("Anthropic package not installed")


def create_deepseek_provider() -> Any:
    """Create DeepSeek provider."""
    api_key = config("DEEPSEEK_API_KEY", default="")
    
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment")
    
    try:
        from openai import AsyncOpenAI
        
        # DeepSeek uses OpenAI-compatible API
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        console.print("[green]✓ DeepSeek provider initialized[/green]")
        return client
    except ImportError:
        raise ValueError("OpenAI package not installed (needed for DeepSeek)")


def create_gemini_provider() -> Any:
    """Create Google Gemini provider."""
    api_key = config("GEMINI_API_KEY", default="")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    try:
        from google import genai
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        client = genai.Client(api_key=api_key)
        console.print("[green]✓ Gemini provider initialized[/green]")
        return client
    except ImportError:
        raise ValueError("Google GenAI package not installed")


def get_provider_config(provider_name: str) -> dict:
    """Get configuration for a specific provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Dictionary with provider configuration
    """
    configs = {
        "OPENAI": {
            "model": "gpt-4o-mini-2024-07-18",
            "max_tokens": 8192,
            "temperature": 0.7,
            "supports_json_mode": True,
            "supports_streaming": True
        },
        "ANTHROPIC": {
            "model": "claude-3-5-haiku-latest", 
            "max_tokens": 200000,
            "temperature": 0.7,
            "supports_json_mode": False,
            "supports_streaming": True
        },
        "DEEPSEEK": {
            "model": "deepseek-chat",
            "max_tokens": 8192,
            "temperature": 0.7,
            "supports_json_mode": True,
            "supports_streaming": True
        },
        "GEMINI": {
            "model": "gemini-2.0-flash-lite",
            "max_tokens": 8192,
            "temperature": 0.7,
            "supports_json_mode": True,
            "supports_streaming": True
        }
    }
    
    return configs.get(provider_name.upper(), configs["OPENAI"])