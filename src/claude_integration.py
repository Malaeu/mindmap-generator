#!/usr/bin/env python
"""Integration module for Claude Code /mindmap command."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import MindmapOrchestrator
from src.providers.factory import create_llm_provider, get_provider_config


async def generate_mindmap_from_file(
    file_path: str,
    provider: str = "OPENAI",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Generate mindmap from a file - entry point for Claude Code.
    
    Args:
        file_path: Path to the document file
        provider: LLM provider to use (OPENAI, ANTHROPIC, DEEPSEEK, GEMINI)
        output_dir: Optional output directory
        
    Returns:
        Dictionary with results and statistics
    """
    file_path = Path(file_path)
    
    # Validate file
    if not file_path.exists():
        return {
            'success': False,
            'error': f'File not found: {file_path}'
        }
    
    if not file_path.is_file():
        return {
            'success': False,
            'error': f'Not a file: {file_path}'
        }
    
    try:
        # Read document
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create provider
        llm_provider = create_llm_provider(provider)
        
        # Initialize orchestrator
        orchestrator = MindmapOrchestrator(
            llm_provider=llm_provider,
            provider_name=provider,
            cache_enabled=True
        )
        
        # Generate mindmap
        result = await orchestrator.generate_mindmap(
            document_content=content,
            output_dir=Path(output_dir) if output_dir else Path("mindmap_outputs"),
            filename_prefix=file_path.stem
        )
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


async def generate_mindmap_from_text(
    text: str,
    provider: str = "OPENAI",
    output_filename: str = "mindmap",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Generate mindmap from text content - for Claude Code inline usage.
    
    Args:
        text: Document text content
        provider: LLM provider to use
        output_filename: Base name for output files
        output_dir: Optional output directory
        
    Returns:
        Dictionary with results and statistics
    """
    try:
        # Create provider
        llm_provider = create_llm_provider(provider)
        
        # Initialize orchestrator
        orchestrator = MindmapOrchestrator(
            llm_provider=llm_provider,
            provider_name=provider,
            cache_enabled=True
        )
        
        # Generate mindmap
        result = await orchestrator.generate_mindmap(
            document_content=text,
            output_dir=Path(output_dir) if output_dir else Path("mindmap_outputs"),
            filename_prefix=output_filename
        )
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def mindmap_command(
    file_or_text: str,
    provider: str = "OPENAI",
    output_dir: Optional[str] = None,
    is_file: bool = True
) -> Dict[str, Any]:
    """Synchronous wrapper for Claude Code integration.
    
    This function can be called directly from Claude Code as:
    /mindmap path/to/file.md --provider OPENAI
    
    Args:
        file_or_text: File path or text content
        provider: LLM provider
        output_dir: Output directory
        is_file: Whether input is a file path (True) or text (False)
        
    Returns:
        Result dictionary
    """
    if is_file:
        return asyncio.run(generate_mindmap_from_file(
            file_or_text,
            provider,
            output_dir
        ))
    else:
        return asyncio.run(generate_mindmap_from_text(
            file_or_text,
            provider,
            output_filename="claude_mindmap",
            output_dir=output_dir
        ))


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate mindmap from file")
    parser.add_argument("file", help="Input file path")
    parser.add_argument("--provider", default="OPENAI", help="LLM provider")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()
    
    result = mindmap_command(
        args.file,
        provider=args.provider,
        output_dir=args.output,
        is_file=True
    )
    
    if result['success']:
        print("\n✅ Mindmap generated successfully!")
        print(f"Processing time: {result['statistics']['processing_time_seconds']}s")
        print(f"Total cost: ${result['statistics']['token_usage']['total_cost']:.4f}")
    else:
        print(f"\n❌ Error: {result['error']}")