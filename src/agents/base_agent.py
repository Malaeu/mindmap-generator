"""Base agent class for all specialized agents."""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from rich.console import Console

from src.core.types import ExtractionResult
from src.utils.tracker import TokenUsageTracker


console = Console()


class BaseAgent(ABC):
    """Abstract base class for all mindmap generation agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        token_tracker: Optional[TokenUsageTracker] = None,
        max_retries: int = 3,
        cache_enabled: bool = True
    ):
        """Initialize base agent.
        
        Args:
            name: Agent identifier
            description: Agent purpose description
            llm_provider: LLM client (OpenAI, Anthropic, etc.)
            token_tracker: Optional token usage tracker
            max_retries: Maximum retry attempts for LLM calls
            cache_enabled: Whether to cache results
        """
        self.name = name
        self.description = description
        self.llm_provider = llm_provider
        self.token_tracker = token_tracker or TokenUsageTracker()
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
        self._call_count = 0
        
        console.print(f"[green]✅ Initialized agent: {name}[/green]")
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> ExtractionResult:
        """Process input data and return result.
        
        Args:
            input_data: Input to process
            **kwargs: Additional parameters
            
        Returns:
            ExtractionResult with processed data
        """
        pass
    
    async def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        task_name: Optional[str] = None
    ) -> str:
        """Call LLM with retry logic and token tracking.
        
        Args:
            prompt: Prompt to send to LLM
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            task_name: Task identifier for tracking
            
        Returns:
            LLM response text
        """
        task_name = task_name or f"{self.name}_call_{self._call_count}"
        self._call_count += 1
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Make the actual LLM call (provider-specific)
                response = await self._make_llm_call(prompt, max_tokens, temperature)
                
                # Track tokens if we have usage data
                if hasattr(response, 'usage'):
                    self.token_tracker.update(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        task=task_name
                    )
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract text from response
                if hasattr(response, 'choices'):
                    text = response.choices[0].message.content
                elif hasattr(response, 'content'):
                    text = response.content[0].text if isinstance(response.content, list) else response.content
                else:
                    text = str(response)
                
                console.print(f"[dim]└─ {self.name}: LLM call completed in {duration_ms:.0f}ms[/dim]")
                return text
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    console.print(f"[yellow]⚠️  {self.name}: Retry {attempt + 1}/{self.max_retries} after {wait_time}s[/yellow]")
                    await asyncio.sleep(wait_time)
                else:
                    console.print(f"[red]❌ {self.name}: Failed after {self.max_retries} attempts: {e}[/red]")
                    raise
    
    @abstractmethod
    async def _make_llm_call(self, prompt: str, max_tokens: int, temperature: float) -> Any:
        """Provider-specific LLM call implementation.
        
        Must be implemented by subclasses for specific providers.
        """
        pass
    
    def _get_cache_key(self, data: Any, **kwargs) -> str:
        """Generate cache key from input data."""
        cache_data = {
            "data": str(data)[:1000],  # Limit size for hashing
            "kwargs": kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    async def _get_cached_or_process(
        self,
        cache_key: str,
        process_func,
        *args,
        **kwargs
    ) -> Any:
        """Get cached result or process and cache."""
        if self.cache_enabled and cache_key in self._cache:
            console.print(f"[dim]└─ {self.name}: Using cached result[/dim]")
            return self._cache[cache_key]
        
        result = await process_func(*args, **kwargs)
        
        if self.cache_enabled:
            self._cache[cache_key] = result
        
        return result
    
    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response, handling common issues."""
        # Clean up response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            if len(lines) > 2:
                response = "\n".join(lines[1:-1])
        
        # Try to parse JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'[\[{].*[\]}]', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            # If all else fails, return as list with single item
            console.print(f"[yellow]⚠️  {self.name}: Could not parse JSON, returning as text[/yellow]")
            return [{"text": response}]
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            "name": self.name,
            "call_count": self._call_count,
            "cache_size": len(self._cache),
            "tokens_used": self.token_tracker.get_summary()
        }