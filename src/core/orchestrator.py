"""Orchestrator for coordinating mindmap generation agents."""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.agents.document_agent import DocumentAnalyzer
from src.agents.extraction_agent import TopicExtractor, SubtopicExtractor, DetailExtractor
from src.agents.verification_agent import VerificationAgent
from src.agents.visualization_agent import VisualizationAgent
from src.core.types import DocumentType, MindmapData, Topic
from src.utils.tracker import TokenUsageTracker


console = Console()


class MindmapOrchestrator:
    """Orchestrates the mindmap generation process using specialized agents."""
    
    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        provider_name: str = "OPENAI",
        max_concurrent_tasks: int = 10,
        cache_enabled: bool = True
    ):
        """Initialize the orchestrator.
        
        Args:
            llm_provider: LLM client instance
            provider_name: Name of the LLM provider
            max_concurrent_tasks: Maximum concurrent agent tasks
            cache_enabled: Whether to enable caching
        """
        self.llm_provider = llm_provider
        self.provider_name = provider_name
        self.max_concurrent_tasks = max_concurrent_tasks
        self.cache_enabled = cache_enabled
        
        # Initialize token tracker
        self.token_tracker = TokenUsageTracker(provider=provider_name)
        
        # Initialize agents
        self._initialize_agents()
        
        # Processing state
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.processing_stats = {}
        
        console.print(f"[bold green]🎭 Mindmap Orchestrator initialized with {provider_name}[/bold green]")
    
    def _initialize_agents(self):
        """Initialize all specialized agents."""
        agent_args = {
            "llm_provider": self.llm_provider,
            "token_tracker": self.token_tracker,
            "cache_enabled": self.cache_enabled
        }
        
        self.document_analyzer = DocumentAnalyzer(**agent_args)
        self.topic_extractor = TopicExtractor(**agent_args)
        self.subtopic_extractor = SubtopicExtractor(**agent_args)
        self.detail_extractor = DetailExtractor(**agent_args)
        self.verification_agent = VerificationAgent(**agent_args)
        self.visualization_agent = VisualizationAgent(**agent_args)
    
    async def generate_mindmap(
        self,
        document_content: str,
        output_dir: Optional[Path] = None,
        filename_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a complete mindmap from document content.
        
        Args:
            document_content: The document text to process
            output_dir: Directory to save outputs (default: mindmap_outputs/)
            filename_prefix: Prefix for output files
            
        Returns:
            Dict containing mindmap data, visualizations, and statistics
        """
        start_time = time.time()
        
        console.print("\n[bold cyan]🚀 Starting Mindmap Generation[/bold cyan]")
        console.print("=" * 80)
        
        try:
            # Phase 1: Document Analysis
            console.print("\n[bold]Phase 1: Document Analysis[/bold]")
            doc_analysis = await self.document_analyzer.process(document_content)
            
            if not doc_analysis.success:
                raise Exception(f"Document analysis failed: {doc_analysis.error}")
            
            doc_type = doc_analysis.data['document_type']
            chunks = doc_analysis.data['chunks']
            
            console.print(f"  Document type: [yellow]{doc_type.name}[/yellow]")
            console.print(f"  Chunks created: [yellow]{len(chunks)}[/yellow]")
            
            # Phase 2: Topic Extraction
            console.print("\n[bold]Phase 2: Topic Extraction[/bold]")
            topics = await self._extract_topics_from_chunks(chunks, doc_type)
            console.print(f"  Topics found: [yellow]{len(topics)}[/yellow]")
            
            # Phase 3: Subtopic Extraction (parallel)
            console.print("\n[bold]Phase 3: Subtopic Extraction[/bold]")
            topics_with_subtopics = await self._extract_subtopics_parallel(
                topics,
                chunks,
                doc_type
            )
            
            subtopic_count = sum(
                len(t.subtopics) for t in topics_with_subtopics
            )
            console.print(f"  Subtopics extracted: [yellow]{subtopic_count}[/yellow]")
            
            # Phase 4: Detail Extraction (parallel)
            console.print("\n[bold]Phase 4: Detail Extraction[/bold]")
            complete_topics = await self._extract_details_parallel(
                topics_with_subtopics,
                chunks
            )
            
            detail_count = sum(
                sum(len(s.details) for s in t.subtopics)
                for t in complete_topics
            )
            console.print(f"  Details extracted: [yellow]{detail_count}[/yellow]")
            
            # Phase 5: Verification and Deduplication
            console.print("\n[bold]Phase 5: Verification & Deduplication[/bold]")
            
            # Create mindmap data structure
            mindmap_data = {
                'central_theme': {
                    'name': 'Document',
                    'subtopics': [t.model_dump() for t in complete_topics]
                }
            }
            
            verification_result = await self.verification_agent.process({
                'content': mindmap_data,
                'source': document_content
            })
            
            if verification_result.success:
                verified_mindmap = verification_result.data['mindmap']
                stats = verification_result.data['statistics']
                
                console.print(f"  Topics after dedup: [green]{stats['processed']['topics']}[/green]")
                console.print(f"  Subtopics after dedup: [green]{stats['processed']['subtopics']}[/green]")
                console.print(f"  Details after dedup: [green]{stats['processed']['details']}[/green]")
            else:
                verified_mindmap = mindmap_data
            
            # Phase 6: Visualization Generation
            console.print("\n[bold]Phase 6: Visualization Generation[/bold]")
            
            visualization_result = await self.visualization_agent.process({
                'mindmap': verified_mindmap
            })
            
            if not visualization_result.success:
                raise Exception(f"Visualization failed: {visualization_result.error}")
            
            visualizations = visualization_result.data
            
            # Save outputs if directory specified
            if output_dir:
                await self._save_outputs(
                    visualizations,
                    output_dir,
                    filename_prefix or "mindmap"
                )
            
            # Calculate final statistics
            duration = time.time() - start_time
            
            result = {
                'success': True,
                'mindmap_data': verified_mindmap,
                'visualizations': visualizations,
                'statistics': {
                    'document_type': doc_type.name,
                    'processing_time_seconds': round(duration, 2),
                    'chunks_processed': len(chunks),
                    'nodes': visualizations['node_count'],
                    'token_usage': self.token_tracker.get_summary()
                }
            }
            
            # Print final summary
            console.print("\n" + "=" * 80)
            console.print("[bold green]✅ Mindmap Generation Complete![/bold green]")
            console.print(f"  Time: [yellow]{duration:.2f}s[/yellow]")
            console.print(f"  Total cost: [yellow]${self.token_tracker.total_cost:.4f}[/yellow]")
            
            # Print detailed token report
            self.token_tracker.print_usage_report()
            
            return result
            
        except Exception as e:
            console.print(f"\n[bold red]❌ Generation failed: {e}[/bold red]")
            return {
                'success': False,
                'error': str(e),
                'statistics': {
                    'processing_time_seconds': time.time() - start_time,
                    'token_usage': self.token_tracker.get_summary()
                }
            }
    
    async def _extract_topics_from_chunks(
        self,
        chunks: List[Dict],
        doc_type: DocumentType
    ) -> List[Topic]:
        """Extract topics from document chunks."""
        # Combine first few chunks for topic extraction
        combined_content = "\n\n".join([
            chunk['text'] for chunk in chunks[:3]
        ])
        
        result = await self.topic_extractor.process({
            'content': combined_content,
            'document_type': doc_type.name
        })
        
        if result.success:
            return result.data['topics']
        else:
            raise Exception(f"Topic extraction failed: {result.error}")
    
    async def _extract_subtopics_parallel(
        self,
        topics: List[Topic],
        chunks: List[Dict],
        doc_type: DocumentType
    ) -> List[Topic]:
        """Extract subtopics for all topics in parallel."""
        # Combine chunks for context
        combined_content = "\n\n".join([
            chunk['text'] for chunk in chunks[:5]
        ])
        
        async def process_topic(topic: Topic) -> Topic:
            """Process a single topic."""
            async with self.semaphore:
                result = await self.subtopic_extractor.process({
                    'topic': topic.name,
                    'content': combined_content,
                    'document_type': doc_type.name
                })
                
                if result.success:
                    topic.subtopics = result.data['subtopics']
                
                return topic
        
        # Process all topics in parallel
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            
            task = progress.add_task(
                "  Extracting subtopics...",
                total=len(topics)
            )
            
            processed_topics = []
            for topic in topics:
                processed = await process_topic(topic)
                processed_topics.append(processed)
                progress.update(task, advance=1)
        
        return processed_topics
    
    async def _extract_details_parallel(
        self,
        topics: List[Topic],
        chunks: List[Dict]
    ) -> List[Topic]:
        """Extract details for all subtopics in parallel."""
        # Get relevant content for detail extraction
        detail_content = "\n\n".join([
            chunk['text'] for chunk in chunks[:7]
        ])
        
        # Count total subtopics
        total_subtopics = sum(len(t.subtopics) for t in topics)
        
        async def process_subtopic(topic: Topic, subtopic):
            """Process a single subtopic."""
            async with self.semaphore:
                result = await self.detail_extractor.process({
                    'subtopic': subtopic.name,
                    'content': detail_content,
                    'topic': topic.name
                })
                
                if result.success:
                    subtopic.details = result.data['details']
                
                return subtopic
        
        # Process all subtopics in parallel
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            
            task = progress.add_task(
                "  Extracting details...",
                total=total_subtopics
            )
            
            tasks = []
            for topic in topics:
                for subtopic in topic.subtopics:
                    tasks.append(process_subtopic(topic, subtopic))
            
            # Process in batches
            for coro in tasks:
                await coro
                progress.update(task, advance=1)
        
        return topics
    
    async def _save_outputs(
        self,
        visualizations: Dict,
        output_dir: Path,
        filename_prefix: str
    ):
        """Save visualization outputs to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Mermaid syntax
        mermaid_file = output_dir / f"{filename_prefix}_{self.provider_name.lower()}.mmd"
        with open(mermaid_file, 'w', encoding='utf-8') as f:
            f.write(visualizations['mermaid'])
        console.print(f"  Saved: [green]{mermaid_file}[/green]")
        
        # Save HTML
        html_file = output_dir / f"{filename_prefix}_{self.provider_name.lower()}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(visualizations['html'])
        console.print(f"  Saved: [green]{html_file}[/green]")
        
        # Save Markdown outline
        md_file = output_dir / f"{filename_prefix}_{self.provider_name.lower()}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(visualizations['markdown'])
        console.print(f"  Saved: [green]{md_file}[/green]")