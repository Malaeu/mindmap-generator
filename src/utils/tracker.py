"""Token usage tracking for cost optimization."""

from typing import Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.text import Text


console = Console()


class TokenUsageTracker:
    """Track token usage and costs across different LLM providers."""
    
    # Provider pricing in USD per token
    PRICING = {
        "OPENAI": {
            "input": 0.15 / 1_000_000,  # GPT-4o-mini
            "output": 0.60 / 1_000_000,
        },
        "ANTHROPIC": {
            "input": 0.80 / 1_000_000,  # Claude 3.5 Haiku
            "output": 4.00 / 1_000_000,
        },
        "DEEPSEEK_CHAT": {
            "input": 0.27 / 1_000_000,
            "output": 1.10 / 1_000_000,
        },
        "DEEPSEEK_REASONER": {
            "input": 0.14 / 1_000_000,
            "output": 2.19 / 1_000_000,
        },
        "GEMINI": {
            "input": 0.075 / 1_000_000,  # Gemini 2.0 Flash Lite
            "output": 0.30 / 1_000_000,
        },
    }
    
    def __init__(self, provider: str = "OPENAI"):
        """Initialize tracker with specified provider."""
        self.provider = provider.upper()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_counts: Dict[str, int] = {}
        self.token_counts_by_task: Dict[str, Dict[str, int]] = {}
        self.cost_by_task: Dict[str, float] = {}
        
        # Task categories for reporting
        self.task_categories = {
            'topics': ['extracting_main_topics', 'consolidating_topics', 'detecting_document_type'],
            'subtopics': ['extracting_subtopics', 'consolidate_subtopics'],
            'details': ['extracting_details', 'consolidate_details'],
            'similarity': ['checking_content_similarity'],
            'verification': ['verifying_against_source'],
            'emoji': ['selecting_emoji'],
            'other': []
        }
        
        # Initialize category counters
        self.call_counts_by_category = {cat: 0 for cat in self.task_categories}
        self.token_counts_by_category = {
            cat: {'input': 0, 'output': 0} for cat in self.task_categories
        }
        self.cost_by_category = {cat: 0.0 for cat in self.task_categories}
    
    def update(self, input_tokens: int, output_tokens: int, task: str):
        """Update token usage for a task."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Calculate cost
        pricing_key = self._get_pricing_key()
        if pricing_key in self.PRICING:
            pricing = self.PRICING[pricing_key]
            task_cost = (
                input_tokens * pricing["input"] +
                output_tokens * pricing["output"]
            )
        else:
            task_cost = 0.0
            console.print(f"[yellow]Warning: No pricing for provider {pricing_key}[/yellow]")
        
        self.total_cost += task_cost
        
        # Update task-specific metrics
        if task not in self.token_counts_by_task:
            self.token_counts_by_task[task] = {'input': 0, 'output': 0}
            self.cost_by_task[task] = 0.0
        
        self.token_counts_by_task[task]['input'] += input_tokens
        self.token_counts_by_task[task]['output'] += output_tokens
        self.cost_by_task[task] += task_cost
        self.call_counts[task] = self.call_counts.get(task, 0) + 1
        
        # Update category metrics
        category = self._get_task_category(task)
        self.call_counts_by_category[category] += 1
        self.token_counts_by_category[category]['input'] += input_tokens
        self.token_counts_by_category[category]['output'] += output_tokens
        self.cost_by_category[category] += task_cost
    
    def _get_pricing_key(self) -> str:
        """Get the pricing key for current provider."""
        if self.provider == "CLAUDE":
            return "ANTHROPIC"
        elif self.provider == "DEEPSEEK":
            # Would need model info to distinguish chat vs reasoner
            return "DEEPSEEK_CHAT"
        return self.provider
    
    def _get_task_category(self, task: str) -> str:
        """Determine category for a task."""
        for category, tasks in self.task_categories.items():
            if any(t in task for t in tasks):
                return category
        return 'other'
    
    def print_usage_report(self):
        """Print detailed usage report with Rich formatting."""
        console.print("\n[bold cyan]📊 Token Usage Report[/bold cyan]")
        console.print("=" * 80)
        
        # Summary table
        summary_table = Table(title="Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="yellow")
        
        summary_table.add_row("Total Input Tokens", f"{self.total_input_tokens:,}")
        summary_table.add_row("Total Output Tokens", f"{self.total_output_tokens:,}")
        summary_table.add_row("Total Tokens", f"{self.total_input_tokens + self.total_output_tokens:,}")
        summary_table.add_row("Total Cost", f"${self.total_cost:.4f}")
        summary_table.add_row("Provider", self.provider)
        
        console.print(summary_table)
        
        # Category breakdown
        if any(self.call_counts_by_category.values()):
            cat_table = Table(title="\nUsage by Category", show_header=True, header_style="bold magenta")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Calls", justify="right", style="white")
            cat_table.add_column("Input Tokens", justify="right", style="green")
            cat_table.add_column("Output Tokens", justify="right", style="blue")
            cat_table.add_column("Cost", justify="right", style="yellow")
            
            for category in self.task_categories:
                if self.call_counts_by_category[category] > 0:
                    cat_table.add_row(
                        category.title(),
                        str(self.call_counts_by_category[category]),
                        f"{self.token_counts_by_category[category]['input']:,}",
                        f"{self.token_counts_by_category[category]['output']:,}",
                        f"${self.cost_by_category[category]:.4f}"
                    )
            
            console.print(cat_table)
        
        # Top expensive tasks
        if self.cost_by_task:
            sorted_tasks = sorted(
                self.cost_by_task.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            task_table = Table(title="\nTop 5 Most Expensive Tasks", show_header=True, header_style="bold magenta")
            task_table.add_column("Task", style="cyan")
            task_table.add_column("Calls", justify="right", style="white")
            task_table.add_column("Avg Cost/Call", justify="right", style="yellow")
            task_table.add_column("Total Cost", justify="right", style="red")
            
            for task, cost in sorted_tasks:
                calls = self.call_counts.get(task, 1)
                avg_cost = cost / calls if calls > 0 else 0
                task_table.add_row(
                    task[:40] + "..." if len(task) > 40 else task,
                    str(calls),
                    f"${avg_cost:.5f}",
                    f"${cost:.4f}"
                )
            
            console.print(task_table)
        
        console.print("\n" + "=" * 80)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "provider": self.provider,
            "call_counts": dict(self.call_counts),
            "cost_by_category": dict(self.cost_by_category),
        }