"""Document analysis agent for type detection and chunking."""

import asyncio
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from src.agents.base_agent import BaseAgent
from src.core.types import ChunkMetadata, DocumentType, ExtractionResult


console = Console()


class DocumentAnalyzer(BaseAgent):
    """Agent responsible for document analysis, type detection, and chunking."""
    
    def __init__(self, *args, **kwargs):
        """Initialize document analyzer agent."""
        super().__init__(
            name="DocumentAnalyzer",
            description="Analyzes documents for type detection and intelligent chunking",
            *args,
            **kwargs
        )
        
        # Chunking configuration
        self.chunk_size = 4000
        self.chunk_overlap = 500
        self.max_chunks = 50
    
    async def process(self, input_data: str, **kwargs) -> ExtractionResult:
        """Process document for type detection and chunking.
        
        Args:
            input_data: Full document text
            
        Returns:
            ExtractionResult with document type and chunks
        """
        console.print(f"[blue]📄 Analyzing document ({len(input_data)} chars)...[/blue]")
        
        try:
            # Detect document type
            doc_type = await self.detect_document_type(input_data)
            
            # Create intelligent chunks
            chunks = self.create_chunks(input_data)
            
            return ExtractionResult(
                success=True,
                data={
                    "document_type": doc_type,
                    "chunks": chunks,
                    "total_chunks": len(chunks),
                    "document_length": len(input_data)
                }
            )
        except Exception as e:
            console.print(f"[red]❌ Document analysis failed: {e}[/red]")
            return ExtractionResult(
                success=False,
                error=str(e)
            )
    
    async def detect_document_type(self, content: str) -> DocumentType:
        """Detect the type of document for specialized processing.
        
        Args:
            content: Document text
            
        Returns:
            Detected DocumentType
        """
        # Use first 2000 chars for type detection
        sample = content[:2000]
        
        prompt = """Analyze this document excerpt and determine its primary type.
        
Categories:
1. TECHNICAL - Software documentation, API specs, technical guides
2. SCIENTIFIC - Research papers, studies, experimental reports  
3. LEGAL - Contracts, laws, regulations, legal documents
4. BUSINESS - Business plans, market analysis, strategy documents
5. ACADEMIC - Textbooks, lectures, educational materials
6. NARRATIVE - Stories, historical accounts, descriptive texts

Document excerpt:
{sample}

Respond with ONLY the category name (e.g., "TECHNICAL").""".format(sample=sample)
        
        cache_key = self._get_cache_key(sample, task="detect_type")
        
        result = await self._get_cached_or_process(
            cache_key,
            self._detect_type_impl,
            prompt
        )
        
        return result
    
    async def _detect_type_impl(self, prompt: str) -> DocumentType:
        """Implementation of document type detection."""
        response = await self._call_llm(
            prompt=prompt,
            max_tokens=50,
            temperature=0.3,
            task_name="detecting_document_type"
        )
        
        # Parse response
        type_str = response.strip().upper()
        
        # Map to enum
        type_map = {
            "TECHNICAL": DocumentType.TECHNICAL,
            "SCIENTIFIC": DocumentType.SCIENTIFIC,
            "LEGAL": DocumentType.LEGAL,
            "BUSINESS": DocumentType.BUSINESS,
            "ACADEMIC": DocumentType.ACADEMIC,
            "NARRATIVE": DocumentType.NARRATIVE,
        }
        
        doc_type = type_map.get(type_str, DocumentType.UNKNOWN)
        
        console.print(f"[green]✓ Detected document type: {doc_type.name}[/green]")
        return doc_type
    
    def create_chunks(self, content: str) -> List[Dict]:
        """Create overlapping chunks from document.
        
        Args:
            content: Full document text
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Adaptive chunk size based on document length
        doc_length = len(content)
        if doc_length < 10000:
            chunk_size = 2000
            overlap = 200
        elif doc_length < 50000:
            chunk_size = 4000
            overlap = 500
        else:
            chunk_size = 6000
            overlap = 800
        
        # Create chunks with overlap
        position = 0
        chunk_index = 0
        
        while position < doc_length and chunk_index < self.max_chunks:
            # Calculate chunk boundaries
            start = max(0, position - overlap if chunk_index > 0 else 0)
            end = min(doc_length, position + chunk_size)
            
            # Extract chunk text
            chunk_text = content[start:end]
            
            # Create chunk metadata
            chunk = {
                "text": chunk_text,
                "metadata": ChunkMetadata(
                    index=chunk_index,
                    start_position=start,
                    end_position=end,
                    overlap_size=overlap if chunk_index > 0 else 0,
                    hash=hashlib.md5(chunk_text.encode()).hexdigest(),
                    processed=False
                ).model_dump()
            }
            
            chunks.append(chunk)
            
            # Move position forward
            position = end
            chunk_index += 1
            
            # Break if we've reached the end
            if end >= doc_length:
                break
        
        console.print(f"[green]✓ Created {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})[/green]")
        return chunks
    
    async def _make_llm_call(self, prompt: str, max_tokens: int, temperature: float) -> Any:
        """Provider-specific LLM call implementation.
        
        This should be overridden by provider-specific subclasses.
        For now, returns a mock response for testing.
        """
        # This will be replaced with actual provider calls
        await asyncio.sleep(0.1)  # Simulate API call
        return type("MockResponse", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": "TECHNICAL"})()
            })()],
            "usage": type("Usage", (), {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 10
            })()
        })()