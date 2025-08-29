"""Visualization agent for generating Mermaid diagrams and HTML."""

import base64
import json
import re
import zlib
from typing import Any

from rich.console import Console

from src.agents.base_agent import BaseAgent
from src.core.types import ExtractionResult

console = Console()


class VisualizationAgent(BaseAgent):
    """Agent for generating Mermaid mindmaps and HTML visualizations."""
    
    def __init__(self, *args, **kwargs):
        """Initialize visualization agent."""
        super().__init__(
            name="VisualizationAgent",
            description="Generates Mermaid diagrams and interactive HTML",
            *args,
            **kwargs
        )
        
        # Emoji selection for nodes
        self.default_emojis = {
            'document': '📄',
            'topic': '🎯',
            'subtopic': '📌',
            'detail': '💡',
            'high': '⭐',
            'medium': '📍',
            'low': '▪️'
        }
    
    async def process(self, input_data: dict, **kwargs) -> ExtractionResult:
        """Generate visualizations from mindmap data.
        
        Args:
            input_data: Dict with 'mindmap' data
            
        Returns:
            ExtractionResult with Mermaid syntax and HTML
        """
        mindmap_data = input_data.get('mindmap', {})
        
        console.print("[blue]🎨 Generating visualizations...[/blue]")
        
        try:
            # Add emojis to nodes
            await self._add_emojis(mindmap_data)
            
            # Generate Mermaid syntax
            mermaid_syntax = self._generate_mermaid(mindmap_data)
            
            # Generate interactive HTML
            html_content = self._generate_html(mermaid_syntax)
            
            # Generate Markdown outline
            markdown_outline = self._generate_markdown(mermaid_syntax)
            
            return ExtractionResult(
                success=True,
                data={
                    "mermaid": mermaid_syntax,
                    "html": html_content,
                    "markdown": markdown_outline,
                    "node_count": self._count_nodes(mindmap_data)
                }
            )
        except Exception as e:
            console.print(f"[red]❌ Visualization generation failed: {e}[/red]")
            return ExtractionResult(
                success=False,
                error=str(e)
            )
    
    async def _add_emojis(self, mindmap_data: dict):
        """Add emojis to mindmap nodes."""
        central_theme = mindmap_data.get('central_theme', {})
        
        # Process topics
        for topic in central_theme.get('subtopics', []):
            if not topic.get('emoji'):
                topic['emoji'] = await self._select_emoji(
                    topic.get('name', ''),
                    'topic'
                )
            
            # Process subtopics
            for subtopic in topic.get('subtopics', []):
                if not subtopic.get('emoji'):
                    subtopic['emoji'] = await self._select_emoji(
                        subtopic.get('name', ''),
                        'subtopic'
                    )
                
                # Process details
                for detail in subtopic.get('details', []):
                    if not detail.get('emoji'):
                        importance = detail.get('importance', 'medium')
                        detail['emoji'] = self.default_emojis.get(
                            importance,
                            self.default_emojis['detail']
                        )
    
    async def _select_emoji(self, text: str, node_type: str) -> str:
        """Select appropriate emoji for a node."""
        # For now, use simple emoji selection
        # Could be enhanced with LLM-based selection
        
        emoji_map = {
            # Technical
            'api': '🔌', 'database': '💾', 'security': '🔒',
            'performance': '⚡', 'architecture': '🏗️', 'cloud': '☁️',
            'network': '🌐', 'code': '💻', 'algorithm': '🧮',
            
            # Business
            'strategy': '♟️', 'growth': '📈', 'market': '🏪',
            'finance': '💰', 'customer': '👥', 'product': '📦',
            'sales': '💼', 'analytics': '📊', 'team': '👫',
            
            # Scientific
            'research': '🔬', 'experiment': '🧪', 'data': '📊',
            'hypothesis': '💭', 'analysis': '🔍', 'methodology': '📋',
            'results': '📈', 'conclusion': '✅', 'theory': '🧠',
            
            # General
            'overview': '🌍', 'introduction': '👋', 'summary': '📝',
            'implementation': '🔧', 'design': '🎨', 'testing': '🧪',
            'documentation': '📚', 'configuration': '⚙️', 'deployment': '🚀'
        }
        
        # Check for keywords
        text_lower = text.lower()
        for keyword, emoji in emoji_map.items():
            if keyword in text_lower:
                return emoji
        
        # Default by type
        return self.default_emojis.get(node_type, '📌')
    
    def _generate_mermaid(self, mindmap_data: dict) -> str:
        """Generate Mermaid mindmap syntax."""
        lines = ["mindmap"]
        
        # Add root node
        lines.append("    ((📄))")
        
        # Add topics and their content
        central_theme = mindmap_data.get('central_theme', {})
        for topic in central_theme.get('subtopics', []):
            self._add_mermaid_node(topic, lines, indent_level=2)
        
        return "\n".join(lines)
    
    def _add_mermaid_node(
        self,
        node: dict,
        lines: list[str],
        indent_level: int
    ):
        """Recursively add nodes to Mermaid diagram."""
        indent = "    " * indent_level
        
        # Determine node shape based on level
        if indent_level == 2:  # Topic
            node_text = f"{node.get('emoji', '🎯')} {node.get('name', '')}"
            lines.append(f"{indent}(({node_text}))")
        elif indent_level == 3:  # Subtopic
            node_text = f"{node.get('emoji', '📌')} {node.get('name', '')}"
            lines.append(f"{indent}({node_text})")
        else:  # Detail
            node_text = node.get('text', '')
            emoji = node.get('emoji', '💡')
            # Escape special characters
            node_text = node_text.replace('(', '\\(').replace(')', '\\)')
            lines.append(f"{indent}[{emoji} {node_text}]")
        
        # Add subtopics
        for subtopic in node.get('subtopics', []):
            self._add_mermaid_node(subtopic, lines, indent_level + 1)
        
        # Add details
        for detail in node.get('details', []):
            self._add_mermaid_node(detail, lines, indent_level + 1)
    
    def _generate_html(self, mermaid_syntax: str) -> str:
        """Generate interactive HTML with Mermaid diagram."""
        # Create compressed URL for Mermaid Live Editor
        data = {
            "code": mermaid_syntax,
            "mermaid": {"theme": "default"}
        }
        json_string = json.dumps(data)
        compressed = zlib.compress(json_string.encode('utf-8'), level=9)
        base64_string = base64.urlsafe_b64encode(compressed).decode('utf-8').rstrip('=')
        edit_url = f'https://mermaid.live/edit#pako:{base64_string}'
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Mindmap Visualization</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
    <!-- Tailwind CSS -->
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    
    <!-- Mermaid JS -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.min.js"></script>
    
    <style>
        body {{ margin: 0; padding: 0; }}
        #mermaid-container {{
            width: 100%;
            height: calc(100vh - 64px);
            overflow: auto;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 2rem;
        }}
        .mermaid {{
            max-width: 100%;
        }}
    </style>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <div class="flex items-center justify-between p-4 bg-white shadow-sm border-b">
        <h1 class="text-xl font-bold text-gray-800">🧠 Mindmap Visualization</h1>
        <div class="flex gap-2">
            <button onclick="downloadSVG()" class="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition">
                📥 Download SVG
            </button>
            <a href="{edit_url}" target="_blank" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition">
                ✏️ Edit in Mermaid Live
            </a>
        </div>
    </div>
    
    <!-- Mermaid Container -->
    <div id="mermaid-container">
        <pre class="mermaid">
{mermaid_syntax}
        </pre>
    </div>
    
    <script>
        // Initialize Mermaid
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            mindmap: {{
                useMaxWidth: true,
                padding: 20
            }}
        }});
        
        // Download SVG function
        function downloadSVG() {{
            const svg = document.querySelector('.mermaid svg');
            if (!svg) return;
            
            const svgData = new XMLSerializer().serializeToString(svg);
            const blob = new Blob([svgData], {{type: 'image/svg+xml'}});
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = 'mindmap.svg';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
        
        // Auto-resize on window resize
        window.addEventListener('resize', () => {{
            mermaid.init();
        }});
    </script>
</body>
</html>"""
        
        return html
    
    def _generate_markdown(self, mermaid_syntax: str) -> str:
        """Convert Mermaid syntax to Markdown outline."""
        lines = []
        
        # Parse Mermaid syntax
        mermaid_lines = mermaid_syntax.split('\n')[1:]  # Skip 'mindmap' header
        
        for line in mermaid_lines:
            if not line.strip():
                continue
            
            # Count indentation
            indent_level = len(re.match(r'^\s*', line).group()) // 4
            content = line.strip()
            
            # Parse different node types
            if indent_level == 1 and '((📄))' in content:
                continue  # Skip root node
            
            elif indent_level == 2:  # Main topics
                match = re.search(r'\(\((.*?)\)\)', content)
                if match:
                    topic = match.group(1).strip()
                    if lines:
                        lines.append("")  # Add spacing
                    lines.append(f"# {topic}")
                    lines.append("")
            
            elif indent_level == 3:  # Subtopics
                match = re.search(r'\((.*?)\)', content)
                if match:
                    subtopic = match.group(1).strip()
                    lines.append(f"## {subtopic}")
                    lines.append("")
            
            elif indent_level == 4:  # Details
                match = re.search(r'\[(.*?)\]', content)
                if match:
                    detail = match.group(1).strip()
                    # Remove escape characters
                    detail = detail.replace('\\(', '(').replace('\\)', ')')
                    lines.append(f"- {detail}")
                    lines.append("")
        
        # Clean up spacing
        markdown = "\n".join(lines)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        return markdown.strip()
    
    def _count_nodes(self, mindmap_data: dict) -> dict:
        """Count nodes at each level."""
        counts = {'topics': 0, 'subtopics': 0, 'details': 0}
        
        central_theme = mindmap_data.get('central_theme', {})
        
        for topic in central_theme.get('subtopics', []):
            counts['topics'] += 1
            
            for subtopic in topic.get('subtopics', []):
                counts['subtopics'] += 1
                
                counts['details'] += len(subtopic.get('details', []))
        
        return counts
    
    async def _make_llm_call(self, prompt: str, max_tokens: int, temperature: float) -> Any:
        """Provider-specific implementation (not needed for visualization)."""
        # Visualization doesn't need LLM calls
        pass