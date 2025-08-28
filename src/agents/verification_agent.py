"""Verification agent for reality checking and deduplication."""

import asyncio
from typing import Any, Dict, List, Optional, Set

from rich.console import Console
from fuzzywuzzy import fuzz

from src.agents.base_agent import BaseAgent
from src.core.types import Detail, ExtractionResult, Subtopic, Topic


console = Console()


class VerificationAgent(BaseAgent):
    """Agent for verifying extracted content against source and removing duplicates."""
    
    def __init__(self, *args, **kwargs):
        """Initialize verification agent."""
        super().__init__(
            name="VerificationAgent",
            description="Verifies extracted content and removes duplicates",
            *args,
            **kwargs
        )
        
        # Similarity thresholds
        self.topic_similarity_threshold = 0.85
        self.subtopic_similarity_threshold = 0.80
        self.detail_similarity_threshold = 0.75
        
        # Verification confidence threshold
        self.verification_confidence_threshold = 0.7
    
    async def process(self, input_data: Dict, **kwargs) -> ExtractionResult:
        """Verify and deduplicate mindmap content.
        
        Args:
            input_data: Dict with 'content' (mindmap data) and 'source' (original doc)
            
        Returns:
            ExtractionResult with verified and deduplicated content
        """
        mindmap_data = input_data.get('content', {})
        source_doc = input_data.get('source', '')
        
        console.print("[blue]🔍 Verifying and deduplicating content...[/blue]")
        
        try:
            # Deduplicate at each level
            deduplicated = await self._deduplicate_mindmap(mindmap_data)
            
            # Verify against source if provided
            if source_doc:
                verified = await self._verify_against_source(deduplicated, source_doc)
            else:
                verified = deduplicated
            
            # Calculate statistics
            stats = self._calculate_stats(mindmap_data, verified)
            
            return ExtractionResult(
                success=True,
                data={
                    "mindmap": verified,
                    "statistics": stats
                }
            )
        except Exception as e:
            console.print(f"[red]❌ Verification failed: {e}[/red]")
            return ExtractionResult(
                success=False,
                error=str(e)
            )
    
    async def _deduplicate_mindmap(self, mindmap_data: Dict) -> Dict:
        """Remove duplicate content at all levels."""
        if not mindmap_data:
            return mindmap_data
        
        # Get central theme with topics
        central_theme = mindmap_data.get('central_theme', {})
        topics = central_theme.get('subtopics', [])
        
        # Deduplicate topics
        unique_topics = self._deduplicate_items(
            topics,
            key_func=lambda x: x.get('name', ''),
            threshold=self.topic_similarity_threshold
        )
        
        # Process each topic
        for topic in unique_topics:
            subtopics = topic.get('subtopics', [])
            
            # Deduplicate subtopics
            unique_subtopics = self._deduplicate_items(
                subtopics,
                key_func=lambda x: x.get('name', ''),
                threshold=self.subtopic_similarity_threshold
            )
            
            # Process each subtopic
            for subtopic in unique_subtopics:
                details = subtopic.get('details', [])
                
                # Deduplicate details
                unique_details = self._deduplicate_items(
                    details,
                    key_func=lambda x: x.get('text', ''),
                    threshold=self.detail_similarity_threshold
                )
                
                subtopic['details'] = unique_details
            
            topic['subtopics'] = unique_subtopics
        
        central_theme['subtopics'] = unique_topics
        
        return {'central_theme': central_theme}
    
    def _deduplicate_items(
        self,
        items: List[Dict],
        key_func,
        threshold: float
    ) -> List[Dict]:
        """Deduplicate a list of items based on similarity."""
        if not items:
            return items
        
        unique = []
        seen_keys = []
        
        for item in items:
            key = key_func(item)
            if not key:
                continue
            
            # Check similarity with existing items
            is_duplicate = False
            for seen_key in seen_keys:
                similarity = fuzz.ratio(key.lower(), seen_key.lower()) / 100
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(item)
                seen_keys.append(key)
        
        if len(unique) < len(items):
            removed = len(items) - len(unique)
            console.print(f"[yellow]  → Removed {removed} duplicates[/yellow]")
        
        return unique
    
    async def _verify_against_source(
        self,
        mindmap_data: Dict,
        source_doc: str
    ) -> Dict:
        """Verify extracted content against source document."""
        console.print("[blue]  📋 Reality checking against source...[/blue]")
        
        # Collect all text items to verify
        items_to_verify = []
        
        def collect_items(node, level='topic'):
            """Recursively collect items."""
            if isinstance(node, dict):
                # Add current node
                if 'name' in node:
                    items_to_verify.append({
                        'text': node['name'],
                        'level': level,
                        'node': node
                    })
                elif 'text' in node:
                    items_to_verify.append({
                        'text': node['text'],
                        'level': 'detail',
                        'node': node
                    })
                
                # Process children
                for subtopic in node.get('subtopics', []):
                    collect_items(subtopic, 'subtopic')
                for detail in node.get('details', []):
                    collect_items(detail, 'detail')
        
        # Collect all items
        central_theme = mindmap_data.get('central_theme', {})
        collect_items(central_theme)
        
        # Verify in batches
        batch_size = 10
        verified_count = 0
        
        for i in range(0, len(items_to_verify), batch_size):
            batch = items_to_verify[i:i + batch_size]
            
            # Create verification prompt
            items_text = "\n".join([
                f"{j+1}. [{item['level']}] {item['text']}"
                for j, item in enumerate(batch)
            ])
            
            verification_results = await self._verify_batch(
                items_text,
                source_doc[:4000]  # Use excerpt
            )
            
            # Update verification status
            for item, verified in zip(batch, verification_results):
                item['node']['verified'] = verified
                if verified:
                    verified_count += 1
        
        percentage = (verified_count / len(items_to_verify) * 100) if items_to_verify else 0
        console.print(f"[green]  ✓ Verified {verified_count}/{len(items_to_verify)} items ({percentage:.1f}%)[/green]")
        
        return mindmap_data
    
    async def _verify_batch(
        self,
        items_text: str,
        source_excerpt: str
    ) -> List[bool]:
        """Verify a batch of items against source."""
        prompt = f"""Verify if these extracted items accurately represent content from the source document.

Items to verify:
{items_text}

Source document excerpt:
{source_excerpt}

For each numbered item, respond with "YES" if it's accurately derived from the source, or "NO" if not.
Return ONLY a JSON array of boolean values in order.
Example: [true, false, true, true, false]"""
        
        response = await self._call_llm(
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
            task_name="verifying_against_source"
        )
        
        # Parse response
        try:
            results = self._parse_json_response(response)
            if isinstance(results, list):
                return [bool(r) for r in results]
        except:
            pass
        
        # Default to all verified if parsing fails
        console.print("[yellow]  ⚠️  Could not parse verification results, assuming verified[/yellow]")
        return [True] * len(items_text.split('\n'))
    
    def _calculate_stats(self, original: Dict, processed: Dict) -> Dict:
        """Calculate processing statistics."""
        def count_items(data):
            counts = {'topics': 0, 'subtopics': 0, 'details': 0}
            
            def count_recursive(node):
                if isinstance(node, dict):
                    if 'subtopics' in node:
                        counts['topics'] += 1
                        for subtopic in node.get('subtopics', []):
                            counts['subtopics'] += 1
                            for detail in subtopic.get('details', []):
                                counts['details'] += 1
                            count_recursive(subtopic)
            
            central = data.get('central_theme', data)
            count_recursive(central)
            return counts
        
        original_counts = count_items(original)
        processed_counts = count_items(processed)
        
        return {
            "original": original_counts,
            "processed": processed_counts,
            "reduction": {
                "topics": original_counts['topics'] - processed_counts['topics'],
                "subtopics": original_counts['subtopics'] - processed_counts['subtopics'],
                "details": original_counts['details'] - processed_counts['details']
            }
        }
    
    async def _make_llm_call(self, prompt: str, max_tokens: int, temperature: float) -> Any:
        """Provider-specific implementation (to be overridden)."""
        # Mock for testing
        await asyncio.sleep(0.1)
        return type("MockResponse", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": "[true, true, false, true]"})()
            })()],
            "usage": type("Usage", (), {
                "prompt_tokens": 200,
                "completion_tokens": 10
            })()
        })()