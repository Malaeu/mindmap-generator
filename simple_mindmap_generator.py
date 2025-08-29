#!/usr/bin/env python3
"""Generate mindmap from conversation about mathematical infinity - no API required."""

import re
from collections import Counter, defaultdict


class ConversationMindmapGenerator:
    """Generate mindmap from conversation using text analysis."""
    
    def __init__(self):
        self.topics = defaultdict(list)
        self.key_concepts = set()
        self.relationships = []
        
    def extract_key_concepts(self, text: str) -> list[str]:
        """Extract mathematical and technical concepts from text."""
        
        # Mathematical concepts to look for
        math_patterns = [
            r'\bалеф[-\s]?ноль\b|\bℵ₀\b|\baleph[-\s]?null\b|\baleph[-\s]?zero\b',
            r'\bконтинуум\b|\bcontinuum\b|\b2\^ℵ₀\b',
            r'\bдзета[-\s]?функц\w+\b|\bzeta[-\s]?function\b|\bζ\(s\)\b',
            r'\bгипотез\w+\s+Риман\w*\b|\bRiemann\s+hypothesis\b|\bRH\b',
            r'\bбиекц\w+\b|\bbijection\b',
            r'\bсчетн\w+\s+бесконечност\w*\b|\bcountable\s+infinit\w*\b',
            r'\bнесчетн\w+\b|\buncountable\b',
            r'\bкардинальност\w*\b|\bcardinality\b',
            r'\bординал\w*\b|\bordinal\w*\b',
            r'\bдиагональн\w+\s+(метод|лемм\w+)\b|\bdiagonal\s+(method|argument)\b',
            r'\bКантор\w*\b|\bCantor\b',
            r'\bГильберт\w*\b|\bHilbert\b',
            r'\bЭйлер\w*\b|\bEuler\b',
            r'\bадел\w+\b|\badel\w+\b',
            r'\bp-адическ\w+\b|\bp-adic\b',
            r'\bL-функц\w+\b|\bL-function\b',
            r'\bспектр\w*\b|\bspectr\w+\b',
            r'\bоператор\w*\b|\boperator\b',
            r'\bсамосопряж\w+\b|\bself-adjoint\b',
            r'\bголоморфн\w+\b|\bholomorphic\b',
            r'\bаналитическ\w+\s+продолжен\w+\b|\banalytic\s+continuation\b',
            r'\bтэта[-\s]?функц\w+\b|\btheta[-\s]?function\b',
            r'\bПоиссон\w*\b|\bPoisson\b',
            r'\bВейл\w*\b|\bWeyl\b',
            r'\bТейт\w*\b|\bTate\b',
            r'\bГаусс\w*\b|\bGauss\w*\b|\bGUE\b',
            r'\bпрост\w+\s+числ\w*\b|\bprime\s+number\w*\b',
            r'\bнул\w+\s+дзет\w+\b|\bzeros?\s+of\s+zeta\b',
            r'\bкритическ\w+\s+(прям\w+|лини\w+)\b|\bcritical\s+(line|strip)\b',
            r'\bфункциональн\w+\s+уравнен\w+\b|\bfunctional\s+equation\b',
            r'\bЛенглендс\w*\b|\bLanglands\b',
            r'\bавтоморфн\w+\b|\bautomorphic\b',
            r'\bпредставлен\w+\s+Галуа\b|\bGalois\s+representation\b',
            r'\bявн\w+\s+формул\w+\b|\bexplicit\s+formula\b',
            r'\bмеллинов\w+\b|\bMellin\b',
            r'\bФурье\b|\bFourier\b',
            r'\bгильбертов\w+\s+пространств\w+\b|\bHilbert\s+space\b',
        ]
        
        concepts = []
        text_lower = text.lower()
        
        for pattern in math_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concept = match.group().strip()
                # Normalize concept name
                if 'алеф' in concept.lower() or 'aleph' in concept.lower() or 'ℵ₀' in concept:
                    concepts.append('Aleph-null (ℵ₀)')
                elif 'континуум' in concept.lower() or 'continuum' in concept.lower():
                    concepts.append('Continuum')
                elif 'дзета' in concept.lower() or 'zeta' in concept.lower() or 'ζ' in concept:
                    concepts.append('Zeta Function')
                elif 'риман' in concept.lower() or 'riemann' in concept.lower():
                    if 'гипотез' in concept.lower() or 'hypothesis' in concept.lower():
                        concepts.append('Riemann Hypothesis')
                    else:
                        concepts.append('Riemann')
                elif 'биекц' in concept.lower() or 'bijection' in concept.lower():
                    concepts.append('Bijection')
                elif 'кантор' in concept.lower() or 'cantor' in concept.lower():
                    concepts.append('Cantor')
                elif 'гильберт' in concept.lower() or 'hilbert' in concept.lower():
                    concepts.append('Hilbert')
                elif 'эйлер' in concept.lower() or 'euler' in concept.lower():
                    concepts.append('Euler Product')
                elif 'адел' in concept.lower() or 'adel' in concept.lower():
                    concepts.append('Adeles')
                elif 'p-ади' in concept.lower() or 'p-adic' in concept.lower():
                    concepts.append('p-adic Numbers')
                elif 'спектр' in concept.lower() or 'spectr' in concept.lower():
                    concepts.append('Spectral Theory')
                elif 'оператор' in concept.lower() or 'operator' in concept.lower():
                    concepts.append('Operators')
                elif 'тэта' in concept.lower() or 'theta' in concept.lower():
                    concepts.append('Theta Function')
                elif 'поиссон' in concept.lower() or 'poisson' in concept.lower():
                    concepts.append('Poisson Summation')
                elif 'вейл' in concept.lower() or 'weyl' in concept.lower():
                    concepts.append('Weyl Formula')
                elif 'гаусс' in concept.lower() or 'gauss' in concept.lower() or 'gue' in concept.lower():
                    concepts.append('Random Matrix Theory (GUE)')
                elif 'прост' in concept.lower() or 'prime' in concept.lower():
                    concepts.append('Prime Numbers')
                elif 'нул' in concept.lower() and ('дзет' in concept.lower() or 'zeta' in concept.lower()):
                    concepts.append('Zeta Zeros')
                elif 'критическ' in concept.lower() or 'critical' in concept.lower():
                    concepts.append('Critical Line')
                elif 'ленглендс' in concept.lower() or 'langlands' in concept.lower():
                    concepts.append('Langlands Program')
                elif 'автоморф' in concept.lower() or 'automorphic' in concept.lower():
                    concepts.append('Automorphic Forms')
                elif 'галуа' in concept.lower() or 'galois' in concept.lower():
                    concepts.append('Galois Representations')
                elif 'явн' in concept.lower() and 'формул' in concept.lower() or 'explicit' in concept.lower() and 'formula' in concept.lower():
                    concepts.append('Explicit Formula')
                elif 'функциональн' in concept.lower() and 'уравнен' in concept.lower():
                    concepts.append('Functional Equation')
                elif 'самосопряж' in concept.lower() or 'self-adjoint' in concept.lower():
                    concepts.append('Self-Adjoint')
                elif 'голоморфн' in concept.lower() or 'holomorphic' in concept.lower():
                    concepts.append('Holomorphic')
                elif 'аналитическ' in concept.lower() and 'продолжен' in concept.lower():
                    concepts.append('Analytic Continuation')
                elif 'меллинов' in concept.lower() or 'mellin' in concept.lower():
                    concepts.append('Mellin Transform')
                elif 'фурье' in concept.lower() or 'fourier' in concept.lower():
                    concepts.append('Fourier Analysis')
                elif 'гильбертов' in concept.lower() and 'пространств' in concept.lower():
                    concepts.append('Hilbert Space')
                elif 'счетн' in concept.lower() and 'бесконечност' in concept.lower():
                    concepts.append('Countable Infinity')
                elif 'несчетн' in concept.lower():
                    concepts.append('Uncountable Infinity')
                elif 'кардинальност' in concept.lower() or 'cardinality' in concept.lower():
                    concepts.append('Cardinality')
                elif 'ординал' in concept.lower() or 'ordinal' in concept.lower():
                    concepts.append('Ordinals')
                elif 'диагональн' in concept.lower() and ('метод' in concept.lower() or 'лемм' in concept.lower()):
                    concepts.append('Diagonal Argument')
                else:
                    # Skip if it contains Cyrillic characters
                    if not any(ord(c) > 127 for c in concept):
                        clean_concept = ' '.join(concept.split())
                        if len(clean_concept) > 2:
                            concepts.append(clean_concept.title())
        
        return list(set(concepts))  # Remove duplicates
    
    def analyze_conversation(self, markdown_file: str):
        """Analyze conversation and extract structure."""
        
        with open(markdown_file, encoding='utf-8') as f:
            content = f.read()
        
        # Split by messages
        messages = content.split('---')
        
        # Track conversation flow
        conversation_topics = []
        
        for msg in messages:
            if not msg.strip():
                continue
                
            # Extract concepts from this message
            concepts = self.extract_key_concepts(msg)
            
            # Group by topic area
            if any('бесконечност' in msg.lower() or 'infinit' in msg.lower() for _ in [1]):
                if 'счетн' in msg.lower() or 'countable' in msg.lower():
                    self.topics['Countable Infinity'].extend(concepts)
                elif 'несчетн' in msg.lower() or 'uncountable' in msg.lower():
                    self.topics['Uncountable Infinity'].extend(concepts)
                else:
                    self.topics['Mathematical Infinity'].extend(concepts)
            
            if 'риман' in msg.lower() or 'riemann' in msg.lower():
                self.topics['Riemann Hypothesis'].extend(concepts)
            
            if 'дзета' in msg.lower() or 'zeta' in msg.lower() or 'ζ' in msg:
                self.topics['Zeta Function Theory'].extend(concepts)
            
            if 'адел' in msg.lower() or 'adel' in msg.lower() or 'p-ади' in msg.lower() or 'p-adic' in msg.lower():
                self.topics['Adelic Theory'].extend(concepts)
            
            if 'спектр' in msg.lower() or 'spectr' in msg.lower() or 'оператор' in msg.lower():
                self.topics['Spectral Theory'].extend(concepts)
            
            if 'ленглендс' in msg.lower() or 'langlands' in msg.lower():
                self.topics['Langlands Program'].extend(concepts)
            
            if 'vme' in msg.lower() or 'memory' in msg.lower() and 'error' in msg.lower():
                self.topics['VME Architecture'].extend(concepts)
            
            if 'тэта' in msg.lower() or 'theta' in msg.lower() or 'поиссон' in msg.lower():
                self.topics['Analytic Bridges'].extend(concepts)
            
            # Extract code blocks
            code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', msg, re.DOTALL)
            if code_blocks:
                self.topics['Implementation'].append('Code Examples')
                for lang, code in code_blocks:
                    if lang.lower() in ['python', 'py']:
                        self.topics['Implementation'].append('Python Implementation')
                    elif lang.lower() in ['r']:
                        self.topics['Implementation'].append('R Visualization')
            
            # Store all unique concepts
            self.key_concepts.update(concepts)
    
    def generate_mermaid(self) -> str:
        """Generate Mermaid mindmap diagram."""
        
        def sanitize_label(text: str) -> str:
            """Sanitize text for Mermaid node labels."""
            # Remove or replace problematic characters
            text = text.replace('ℵ₀', 'Aleph-0')
            text = text.replace('ℵ', 'Aleph')
            text = text.replace('×', 'x')
            text = text.replace('ζ', 'zeta')
            text = text.replace('ξ', 'xi')
            text = text.replace('Γ', 'Gamma')
            text = text.replace('π', 'pi')
            text = text.replace('∞', 'infinity')
            text = text.replace('²', '^2')
            text = text.replace('³', '^3')
            text = text.replace('₀', '0')
            text = text.replace('₁', '1')
            text = text.replace('₂', '2')
            # Remove all Cyrillic characters and replace with English equivalents
            cyrillic_map = {
                'А': 'A', 'а': 'a', 'Б': 'B', 'б': 'b', 'В': 'V', 'в': 'v',
                'Г': 'G', 'г': 'g', 'Д': 'D', 'д': 'd', 'Е': 'E', 'е': 'e',
                'Ё': 'E', 'ё': 'e', 'Ж': 'Zh', 'ж': 'zh', 'З': 'Z', 'з': 'z',
                'И': 'I', 'и': 'i', 'Й': 'Y', 'й': 'y', 'К': 'K', 'к': 'k',
                'Л': 'L', 'л': 'l', 'М': 'M', 'м': 'm', 'Н': 'N', 'н': 'n',
                'О': 'O', 'о': 'o', 'П': 'P', 'п': 'p', 'Р': 'R', 'р': 'r',
                'С': 'S', 'с': 's', 'Т': 'T', 'т': 't', 'У': 'U', 'у': 'u',
                'Ф': 'F', 'ф': 'f', 'Х': 'Kh', 'х': 'kh', 'Ц': 'Ts', 'ц': 'ts',
                'Ч': 'Ch', 'ч': 'ch', 'Ш': 'Sh', 'ш': 'sh', 'Щ': 'Shch', 'щ': 'shch',
                'Ъ': '', 'ъ': '', 'Ы': 'Y', 'ы': 'y', 'Ь': '', 'ь': '',
                'Э': 'E', 'э': 'e', 'Ю': 'Yu', 'ю': 'yu', 'Я': 'Ya', 'я': 'ya'
            }
            for cyr, lat in cyrillic_map.items():
                text = text.replace(cyr, lat)
            # Replace quotes and special chars
            text = text.replace('"', "'")
            # CRITICAL FIX: Replace brackets with parentheses to avoid Mermaid syntax errors
            text = text.replace('[', '(')
            text = text.replace(']', ')')
            text = text.replace('<', '(')
            text = text.replace('>', ')')
            text = text.replace('&', 'and')
            text = text.replace('#', 'num')
            # Remove any remaining non-ASCII characters
            text = ''.join(c if ord(c) < 128 else '' for c in text)
            # Clean up extra spaces
            text = ' '.join(text.split())
            return text if text else "Node"
        
        mermaid = """graph TB
    %% Mathematical Infinity Exploration Mindmap
    
    Root[Mathematical Infinity and Riemann Hypothesis]
    
    %% Main branches
"""
        
        # Create nodes for main topics
        node_id = 1
        topic_nodes = {}
        
        for topic, concepts in self.topics.items():
            if concepts:
                topic_id = f"T{node_id}"
                topic_nodes[topic] = topic_id
                # Clean topic name for display
                clean_topic = sanitize_label(topic.replace('_', ' '))
                mermaid += f"    {topic_id}[{clean_topic}]\n"
                mermaid += f"    Root --> {topic_id}\n"
                node_id += 1
        
        mermaid += "\n    %% Concepts under each topic\n"
        
        # Add concepts under each topic
        concept_id = 1
        for topic, concepts in self.topics.items():
            if topic in topic_nodes:
                # Count and deduplicate concepts
                concept_counts = Counter(concepts)
                # Get top concepts
                top_concepts = concept_counts.most_common(10)
                
                for concept, count in top_concepts:
                    if concept and len(concept) > 2:
                        c_id = f"C{concept_id}"
                        # Sanitize concept for Mermaid
                        safe_concept = sanitize_label(concept)
                        if count > 1:
                            safe_concept += f" x{count}"
                        mermaid += f"    {c_id}[{safe_concept}]\n"
                        mermaid += f"    {topic_nodes[topic]} --> {c_id}\n"
                        concept_id += 1
        
        # Add relationships between related topics
        mermaid += "\n    %% Cross-connections\n"
        
        # Define relationships
        relationships = [
            ('Riemann Hypothesis', 'Zeta Function Theory', 'analyzes'),
            ('Zeta Function Theory', 'Prime Numbers', 'encodes'),
            ('Spectral Theory', 'Riemann Hypothesis', 'approaches'),
            ('Adelic Theory', 'Zeta Function Theory', 'generalizes'),
            ('Langlands Program', 'Automorphic Forms', 'connects'),
            ('Mathematical Infinity', 'Countable Infinity', 'includes'),
            ('Mathematical Infinity', 'Uncountable Infinity', 'includes'),
            ('Analytic Bridges', 'Zeta Function Theory', 'transforms'),
            ('VME Architecture', 'Implementation', 'guides'),
        ]
        
        for source, target, rel_type in relationships:
            if source in topic_nodes and target in self.topics:
                if target not in topic_nodes:
                    # Create node if doesn't exist
                    t_id = f"T{node_id}"
                    topic_nodes[target] = t_id
                    clean_target = sanitize_label(target)
                    mermaid += f"    {t_id}[{clean_target}]\n"
                    node_id += 1
                # Add dotted connection
                mermaid += f"    {topic_nodes[source]} -.-> {topic_nodes[target]}\n"
        
        # Add styling
        mermaid += """
    %% Styling
    classDef mainTopic fill:#e1f5fe,stroke:#0288d1,stroke-width:3px
    classDef concept fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef implementation fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class Root mainTopic
"""
        
        # Apply styles to topic nodes
        for topic, node_id in topic_nodes.items():
            if 'Implementation' in topic or 'VME' in topic:
                mermaid += f"    class {node_id} implementation\n"
            else:
                mermaid += f"    class {node_id} mainTopic\n"
        
        return mermaid
    
    def generate_html(self, mermaid_content: str, title: str = "Mathematical Infinity Mindmap") -> str:
        """Generate interactive HTML with Mermaid diagram."""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
            max-width: 95%;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        #mindmap {{
            width: 100%;
            overflow-x: auto;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            background: #fafafa;
        }}
        .controls {{
            margin: 20px 0;
            text-align: center;
        }}
        button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
            font-size: 14px;
            transition: background 0.3s;
        }}
        button:hover {{
            background: #5a67d8;
        }}
        .info {{
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
            padding: 10px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            font-size: 12px;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 {title}</h1>
        <div class="subtitle">Interactive Knowledge Graph from Chat Conversation</div>
        
        <div class="info">
            <strong>📊 Visualization:</strong> This mindmap represents the key mathematical concepts 
            discussed in the conversation about infinity, the Riemann Hypothesis, and their deep connections.
            The diagram shows main topics, subtopics, and their relationships.
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(self.topics)}</div>
                <div class="stat-label">Main Topics</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.key_concepts)}</div>
                <div class="stat-label">Key Concepts</div>
            </div>
            <div class="stat">
                <div class="stat-value">{sum(len(v) for v in self.topics.values())}</div>
                <div class="stat-label">Total Connections</div>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="zoomIn()">🔍 Zoom In</button>
            <button onclick="zoomOut()">🔍 Zoom Out</button>
            <button onclick="resetZoom()">↺ Reset</button>
            <button onclick="downloadSVG()">💾 Download SVG</button>
        </div>
        
        <div id="mindmap">
            <div class="mermaid">
{mermaid_content}
            </div>
        </div>
    </div>
    
    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
        
        let currentZoom = 1;
        
        function zoomIn() {{
            currentZoom *= 1.2;
            updateZoom();
        }}
        
        function zoomOut() {{
            currentZoom *= 0.8;
            updateZoom();
        }}
        
        function resetZoom() {{
            currentZoom = 1;
            updateZoom();
        }}
        
        function updateZoom() {{
            const svg = document.querySelector('#mindmap svg');
            if (svg) {{
                svg.style.transform = `scale(${{currentZoom}})`;
                svg.style.transformOrigin = 'center';
            }}
        }}
        
        function downloadSVG() {{
            const svg = document.querySelector('#mindmap svg');
            if (svg) {{
                const svgData = new XMLSerializer().serializeToString(svg);
                const svgBlob = new Blob([svgData], {{type: 'image/svg+xml;charset=utf-8'}});
                const svgUrl = URL.createObjectURL(svgBlob);
                const downloadLink = document.createElement('a');
                downloadLink.href = svgUrl;
                downloadLink.download = 'mindmap.svg';
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            }}
        }}
    </script>
</body>
</html>"""
        
        return html
    
    def generate_markdown_outline(self) -> str:
        """Generate a structured markdown outline."""
        
        outline = """# Mathematical Infinity & Riemann Hypothesis - Knowledge Map

## 📚 Executive Summary

This conversation explores the deep connections between mathematical infinity, the Riemann Hypothesis, 
and various mathematical frameworks that bridge discrete and continuous mathematics.

## 🗺️ Main Topic Areas

"""
        
        for topic in sorted(self.topics.keys()):
            concepts = self.topics[topic]
            if concepts:
                outline += f"### {topic}\n\n"
                
                # Count and deduplicate
                concept_counts = Counter(concepts)
                top_concepts = concept_counts.most_common(15)
                
                for concept, count in top_concepts:
                    if concept and len(concept) > 2:
                        if count > 1:
                            outline += f"- **{concept}** (mentioned {count} times)\n"
                        else:
                            outline += f"- {concept}\n"
                
                outline += "\n"
        
        outline += """## 🔗 Key Relationships

### Infinity Hierarchy
- **Countable Infinity (ℵ₀)**: Natural numbers, integers, rationals
- **Uncountable Infinity (Continuum)**: Real numbers, 2^ℵ₀
- **Continuum Hypothesis**: Independence from ZFC axioms

### Riemann Hypothesis Connections
- **Zeta Function**: Encodes prime distribution
- **Spectral Theory**: Hilbert-Pólya conjecture
- **Random Matrices**: GUE statistics match zeta zeros
- **Adelic Theory**: Unifies p-adic and real analysis

### Analytic Bridges
- **Theta Functions**: Poisson summation formula
- **Functional Equation**: Symmetry ξ(s) = ξ(1-s)
- **Explicit Formula**: Links zeros to prime counting

### Modern Approaches
- **Langlands Program**: Connects representation theory to number theory
- **VME Architecture**: Computational framework for RH
- **Tate's Thesis**: Adelic reformulation of zeta

## 💡 Key Insights

1. **Bijection techniques** establish equivalence between infinite sets
2. **Diagonal argument** proves uncountability of reals
3. **Euler product** reveals multiplicative structure of primes
4. **Spectral interpretation** suggests quantum mechanical approach
5. **Adelic viewpoint** unifies local and global perspectives

## 🔬 Technical Implementations

The conversation includes practical code examples for:
- Pairing functions for integer bijections
- Symmetry verification of xi function
- Visualization of critical zeros
- VME architecture implementation

## 📖 References & Concepts Index

**Mathematicians**: Riemann, Cantor, Hilbert, Euler, Weyl, Tate, Langlands
**Functions**: Zeta, Theta, Gamma, L-functions
**Theories**: Spectral, Adelic, Random Matrix, Galois
**Methods**: Diagonal argument, Analytic continuation, Poisson summation

---
*Generated from conversation analysis - {len(self.key_concepts)} unique concepts identified*
"""
        
        return outline

# Main execution
if __name__ == "__main__":
    generator = ConversationMindmapGenerator()
    
    # Analyze the conversation
    print("Analyzing conversation...")
    generator.analyze_conversation("infinity_conversation.md")
    
    # Generate outputs
    print(f"Found {len(generator.topics)} main topics")
    print(f"Identified {len(generator.key_concepts)} unique concepts")
    
    # Generate Mermaid diagram
    mermaid = generator.generate_mermaid()
    with open("mindmap_outputs/infinity_mindmap.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid)
    print("✓ Generated Mermaid diagram: mindmap_outputs/infinity_mindmap.mmd")
    
    # Generate HTML
    html = generator.generate_html(mermaid)
    with open("mindmap_outputs/infinity_mindmap.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✓ Generated interactive HTML: mindmap_outputs/infinity_mindmap.html")
    
    # Generate Markdown outline
    outline = generator.generate_markdown_outline()
    with open("mindmap_outputs/infinity_mindmap_outline.md", "w", encoding="utf-8") as f:
        f.write(outline)
    print("✓ Generated markdown outline: mindmap_outputs/infinity_mindmap_outline.md")
    
    print("\n🎉 Mindmap generation complete!")
    print("Open mindmap_outputs/infinity_mindmap.html in a browser to view the interactive diagram")