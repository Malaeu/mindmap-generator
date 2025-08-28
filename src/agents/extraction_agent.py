"""Content extraction agents for topics, subtopics, and details."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from rich.console import Console
from fuzzywuzzy import fuzz

from src.agents.base_agent import BaseAgent
from src.core.types import Detail, ExtractionResult, Subtopic, Topic


console = Console()


class TopicExtractor(BaseAgent):
    """Agent for extracting main topics from document chunks."""
    
    def __init__(self, *args, **kwargs):
        """Initialize topic extractor."""
        super().__init__(
            name="TopicExtractor",
            description="Extracts main topics and themes from documents",
            *args,
            **kwargs
        )
        self.min_topics = 3
        self.max_topics = 12
        self.similarity_threshold = 0.85
    
    async def process(self, input_data: Dict, **kwargs) -> ExtractionResult:
        """Extract main topics from document.
        
        Args:
            input_data: Dict with 'content' and 'document_type'
            
        Returns:
            ExtractionResult with list of topics
        """
        content = input_data.get('content', '')
        doc_type = input_data.get('document_type', 'UNKNOWN')
        
        console.print(f"[blue]🎯 Extracting topics for {doc_type} document...[/blue]")
        
        try:
            topics = await self._extract_topics(content, doc_type)
            
            # Deduplicate topics
            unique_topics = self._deduplicate_topics(topics)
            
            return ExtractionResult(
                success=True,
                data={
                    "topics": unique_topics,
                    "total_extracted": len(topics),
                    "after_deduplication": len(unique_topics)
                }
            )
        except Exception as e:
            console.print(f"[red]❌ Topic extraction failed: {e}[/red]")
            return ExtractionResult(
                success=False,
                error=str(e)
            )
    
    async def _extract_topics(self, content: str, doc_type: str) -> List[Topic]:
        """Extract topics from content."""
        prompt = self._get_topic_prompt(content, doc_type)
        
        response = await self._call_llm(
            prompt=prompt,
            max_tokens=800,
            temperature=0.7,
            task_name="extracting_main_topics"
        )
        
        # Parse response
        topics_data = self._parse_json_response(response)
        
        # Convert to Topic objects
        topics = []
        for item in topics_data:
            if isinstance(item, dict):
                topic = Topic(name=item.get('name', item.get('text', str(item))))
            else:
                topic = Topic(name=str(item))
            topics.append(topic)
        
        console.print(f"[green]✓ Extracted {len(topics)} topics[/green]")
        return topics
    
    def _deduplicate_topics(self, topics: List[Topic]) -> List[Topic]:
        """Remove duplicate or highly similar topics."""
        unique = []
        
        for topic in topics:
            is_duplicate = False
            for existing in unique:
                similarity = fuzz.ratio(
                    topic.name.lower(),
                    existing.name.lower()
                )
                if similarity > self.similarity_threshold * 100:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(topic)
        
        if len(unique) < len(topics):
            console.print(f"[yellow]→ Removed {len(topics) - len(unique)} duplicate topics[/yellow]")
        
        return unique
    
    def _get_topic_prompt(self, content: str, doc_type: str) -> str:
        """Get specialized prompt based on document type."""
        base_prompt = """Extract the main topics from this document.

Requirements:
1. Identify {min_topics}-{max_topics} distinct, high-level themes
2. Each topic should be self-contained and meaningful
3. Topics should cover different aspects of the document
4. Use clear, concise names (2-5 words)

Document content:
{content}

Return a JSON array of topic names.
Example: ["System Architecture", "Performance Optimization", "Security Features"]
""".format(
            min_topics=self.min_topics,
            max_topics=self.max_topics,
            content=content[:6000]  # Limit content size
        )
        
        return base_prompt
    
    async def _make_llm_call(self, prompt: str, max_tokens: int, temperature: float) -> Any:
        """Provider-specific implementation (to be overridden)."""
        # Mock for testing
        await asyncio.sleep(0.1)
        mock_topics = '["System Design", "Implementation Details", "Performance Metrics"]'
        return type("MockResponse", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": mock_topics})()
            })()],
            "usage": type("Usage", (), {
                "prompt_tokens": 100,
                "completion_tokens": 20
            })()
        })()


class SubtopicExtractor(BaseAgent):
    """Agent for extracting subtopics under main topics."""
    
    def __init__(self, *args, **kwargs):
        """Initialize subtopic extractor."""
        super().__init__(
            name="SubtopicExtractor",
            description="Extracts subtopics for each main topic",
            *args,
            **kwargs
        )
        self.min_subtopics = 2
        self.max_subtopics = 6
    
    async def process(self, input_data: Dict, **kwargs) -> ExtractionResult:
        """Extract subtopics for a topic.
        
        Args:
            input_data: Dict with 'topic', 'content', and 'document_type'
            
        Returns:
            ExtractionResult with list of subtopics
        """
        topic = input_data.get('topic', '')
        content = input_data.get('content', '')
        
        console.print(f"[blue]  └─ Extracting subtopics for '{topic}'...[/blue]")
        
        try:
            subtopics = await self._extract_subtopics(topic, content)
            
            return ExtractionResult(
                success=True,
                data={
                    "subtopics": subtopics,
                    "parent_topic": topic
                }
            )
        except Exception as e:
            console.print(f"[red]    ❌ Subtopic extraction failed: {e}[/red]")
            return ExtractionResult(
                success=False,
                error=str(e)
            )
    
    async def _extract_subtopics(self, topic: str, content: str) -> List[Subtopic]:
        """Extract subtopics for a given topic."""
        prompt = f"""For the topic "{topic}", identify {self.min_subtopics}-{self.max_subtopics} specific subtopics.

Each subtopic should:
- Be a distinct aspect or component of "{topic}"
- Be specific and actionable
- Represent important information from the document

Document excerpt:
{content[:4000]}

Return a JSON array of subtopic names.
Example: ["API Design", "Database Schema", "Caching Strategy"]"""
        
        response = await self._call_llm(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            task_name=f"extracting_subtopics_{topic}"
        )
        
        # Parse response
        subtopics_data = self._parse_json_response(response)
        
        # Convert to Subtopic objects
        subtopics = []
        for item in subtopics_data:
            if isinstance(item, dict):
                name = item.get('name', item.get('text', str(item)))
            else:
                name = str(item)
            
            subtopic = Subtopic(
                name=name,
                parent_topic=topic
            )
            subtopics.append(subtopic)
        
        console.print(f"[green]    ✓ Extracted {len(subtopics)} subtopics[/green]")
        return subtopics
    
    async def _make_llm_call(self, prompt: str, max_tokens: int, temperature: float) -> Any:
        """Provider-specific implementation (to be overridden)."""
        # Mock for testing
        await asyncio.sleep(0.1)
        mock_subtopics = '["Component A", "Component B", "Component C"]'
        return type("MockResponse", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": mock_subtopics})()
            })()],
            "usage": type("Usage", (), {
                "prompt_tokens": 80,
                "completion_tokens": 15
            })()
        })()


class DetailExtractor(BaseAgent):
    """Agent for extracting specific details under subtopics."""
    
    def __init__(self, *args, **kwargs):
        """Initialize detail extractor."""
        super().__init__(
            name="DetailExtractor",
            description="Extracts specific details and evidence for subtopics",
            *args,
            **kwargs
        )
        self.min_details = 2
        self.max_details = 5
    
    async def process(self, input_data: Dict, **kwargs) -> ExtractionResult:
        """Extract details for a subtopic.
        
        Args:
            input_data: Dict with 'subtopic', 'content', and optional 'topic'
            
        Returns:
            ExtractionResult with list of details
        """
        subtopic = input_data.get('subtopic', '')
        content = input_data.get('content', '')
        
        console.print(f"[blue]      └─ Extracting details for '{subtopic}'...[/blue]")
        
        try:
            details = await self._extract_details(subtopic, content)
            
            return ExtractionResult(
                success=True,
                data={
                    "details": details,
                    "parent_subtopic": subtopic
                }
            )
        except Exception as e:
            console.print(f"[red]        ❌ Detail extraction failed: {e}[/red]")
            return ExtractionResult(
                success=False,
                error=str(e)
            )
    
    async def _extract_details(self, subtopic: str, content: str) -> List[Detail]:
        """Extract specific details for a subtopic."""
        prompt = f"""For the subtopic "{subtopic}", extract {self.min_details}-{self.max_details} specific details.

Each detail should:
- Provide concrete, factual information
- Include examples, numbers, or evidence when available
- Be 2-3 sentences long
- Add unique value not covered by other details

Document excerpt:
{content[:3000]}

Return a JSON array where each object has:
- "text": The detail description (2-3 sentences)
- "importance": "high", "medium", or "low"

Example:
[
  {{"text": "The API uses REST principles with JSON payloads. Response times average 200ms under normal load.", "importance": "high"}},
  {{"text": "Authentication is handled via JWT tokens with 1-hour expiration.", "importance": "medium"}}
]"""
        
        response = await self._call_llm(
            prompt=prompt,
            max_tokens=600,
            temperature=0.7,
            task_name=f"extracting_details_{subtopic}"
        )
        
        # Parse response
        details_data = self._parse_json_response(response)
        
        # Convert to Detail objects
        details = []
        for item in details_data:
            if isinstance(item, dict):
                detail = Detail(
                    text=item.get('text', ''),
                    importance=item.get('importance', 'medium'),
                    parent_subtopic=subtopic
                )
                details.append(detail)
        
        console.print(f"[green]        ✓ Extracted {len(details)} details[/green]")
        return details
    
    async def _make_llm_call(self, prompt: str, max_tokens: int, temperature: float) -> Any:
        """Provider-specific implementation (to be overridden)."""
        # Mock for testing
        await asyncio.sleep(0.1)
        mock_details = json.dumps([
            {"text": "Detail 1 about the subtopic with specific information.", "importance": "high"},
            {"text": "Detail 2 providing additional context and examples.", "importance": "medium"}
        ])
        return type("MockResponse", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": mock_details})()
            })()],
            "usage": type("Usage", (), {
                "prompt_tokens": 120,
                "completion_tokens": 40
            })()
        })()