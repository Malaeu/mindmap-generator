#!/usr/bin/env python
"""Main entry point for the agentized mindmap generator."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from decouple import config

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import MindmapOrchestrator
from src.providers.factory import create_llm_provider


console = Console()


def print_banner():
    """Print a stylish banner."""
    banner = Text.from_markup(
        """[bold cyan]
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🧠  [white]MINDMAP GENERATOR - AGENTIC EDITION[/white]  🧠       ║
║                                                               ║
║       Transform documents into intelligent mindmaps          ║
║         using specialized AI agents and LLMs                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        [/bold cyan]"""
    )
    console.print(banner)


async def process_document(
    file_path: Path,
    provider: str,
    output_dir: Optional[Path] = None
) -> bool:
    """Process a single document file.
    
    Args:
        file_path: Path to the document
        provider: LLM provider to use
        output_dir: Output directory for results
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read document
        console.print(f"\n📖 Reading document: [yellow]{file_path}[/yellow]")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        console.print(f"  Size: [green]{len(content):,} characters[/green]")
        
        # Create LLM provider
        console.print(f"\n🤖 Initializing {provider} provider...")
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
            output_dir=output_dir or Path("mindmap_outputs"),
            filename_prefix=file_path.stem
        )
        
        if result['success']:
            console.print("\n[bold green]✅ Success![/bold green]")
            
            # Display statistics
            stats = result['statistics']
            console.print(Panel.fit(
                f"""[bold]Generation Statistics:[/bold]
                
Document Type: [yellow]{stats['document_type']}[/yellow]
Processing Time: [yellow]{stats['processing_time_seconds']}s[/yellow]
Chunks Processed: [yellow]{stats['chunks_processed']}[/yellow]

[bold]Node Counts:[/bold]
Topics: [green]{stats['nodes']['topics']}[/green]
Subtopics: [green]{stats['nodes']['subtopics']}[/green]
Details: [green]{stats['nodes']['details']}[/green]

[bold]Token Usage:[/bold]
Total Tokens: [yellow]{stats['token_usage']['total_tokens']:,}[/yellow]
Total Cost: [yellow]${stats['token_usage']['total_cost']:.4f}[/yellow]
""",
                title="📊 Results",
                border_style="green"
            ))
            
            return True
        else:
            console.print(f"\n[bold red]❌ Failed: {result.get('error', 'Unknown error')}[/bold red]")
            return False
            
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        return False


async def interactive_mode():
    """Run in interactive mode with user prompts."""
    print_banner()
    
    # Get available providers
    available_providers = []
    if config("OPENAI_API_KEY", default=""):
        available_providers.append("OPENAI")
    if config("ANTHROPIC_API_KEY", default=""):
        available_providers.append("ANTHROPIC")
    if config("DEEPSEEK_API_KEY", default=""):
        available_providers.append("DEEPSEEK")
    if config("GEMINI_API_KEY", default=""):
        available_providers.append("GEMINI")
    
    if not available_providers:
        console.print("[bold red]❌ No API keys configured![/bold red]")
        console.print("\nPlease set one or more of the following in your .env file:")
        console.print("  - OPENAI_API_KEY")
        console.print("  - ANTHROPIC_API_KEY")
        console.print("  - DEEPSEEK_API_KEY")
        console.print("  - GEMINI_API_KEY")
        return
    
    console.print(f"\n✅ Available providers: {', '.join(available_providers)}")
    
    # Select provider
    if len(available_providers) == 1:
        provider = available_providers[0]
        console.print(f"Using provider: [yellow]{provider}[/yellow]")
    else:
        provider = Prompt.ask(
            "\n🤖 Select provider",
            choices=available_providers,
            default=available_providers[0]
        )
    
    while True:
        # Get document path
        file_path = Prompt.ask("\n📄 Enter document path (or 'quit' to exit)")
        
        if file_path.lower() in ['quit', 'exit', 'q']:
            console.print("\n👋 Goodbye!")
            break
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            console.print(f"[red]❌ File not found: {file_path}[/red]")
            continue
        
        if not file_path.is_file():
            console.print(f"[red]❌ Not a file: {file_path}[/red]")
            continue
        
        # Process document
        await process_document(file_path, provider)
        
        # Ask if user wants to continue
        if not Confirm.ask("\n🔄 Process another document?"):
            console.print("\n👋 Goodbye!")
            break


async def batch_mode(
    input_pattern: str,
    provider: str,
    output_dir: Optional[str] = None
):
    """Process multiple documents matching a pattern.
    
    Args:
        input_pattern: Glob pattern for input files
        provider: LLM provider to use
        output_dir: Output directory for results
    """
    print_banner()
    
    # Find matching files
    files = list(Path().glob(input_pattern))
    
    if not files:
        console.print(f"[red]❌ No files found matching: {input_pattern}[/red]")
        return
    
    console.print(f"\n📁 Found {len(files)} files to process")
    
    output_path = Path(output_dir) if output_dir else Path("mindmap_outputs")
    
    # Process each file
    success_count = 0
    for i, file_path in enumerate(files, 1):
        console.print(f"\n[bold]Processing file {i}/{len(files)}[/bold]")
        
        if await process_document(file_path, provider, output_path):
            success_count += 1
    
    # Summary
    console.print("\n" + "=" * 80)
    console.print(f"[bold]Batch Processing Complete[/bold]")
    console.print(f"  Success: [green]{success_count}/{len(files)}[/green]")
    console.print(f"  Failed: [red]{len(files) - success_count}/{len(files)}[/red]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate intelligent mindmaps from documents using AI agents"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input document path or glob pattern"
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["OPENAI", "ANTHROPIC", "DEEPSEEK", "GEMINI"],
        default=config("API_PROVIDER", default="OPENAI"),
        help="LLM provider to use"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: mindmap_outputs)"
    )
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Process multiple files matching pattern"
    )
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.input:
        if args.batch:
            # Batch mode
            asyncio.run(batch_mode(
                args.input,
                args.provider,
                args.output
            ))
        else:
            # Single file mode
            file_path = Path(args.input)
            if not file_path.exists():
                console.print(f"[red]❌ File not found: {file_path}[/red]")
                sys.exit(1)
            
            asyncio.run(process_document(
                file_path,
                args.provider,
                Path(args.output) if args.output else None
            ))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()