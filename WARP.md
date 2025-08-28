# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an **intelligent mindmap generator** that transforms text documents into hierarchical, visual mindmaps using Large Language Models. The system is sophisticated, employing non-linear exploration patterns and advanced content deduplication techniques.

### Core Architecture

The system uses an **undirected graph exploration model** rather than traditional linear LLM pipelines:
- **Multi-dimensional processing**: Parallel exploration of different document aspects
- **Feedback loops**: Quality evaluation and verification mechanisms  
- **Heuristic-guided decisions**: Adaptive depth vs breadth exploration
- **Reality checking**: Verification against source content to prevent hallucination

### Key Components

1. **MindMapGenerator**: Main orchestrator class with async processing
2. **DocumentOptimizer**: Multi-provider LLM client manager (OpenAI, Anthropic, DeepSeek, Gemini)
3. **TokenUsageTracker**: Comprehensive cost tracking and reporting
4. **DocumentType**: Intelligent document classification system
5. **MinimalDatabaseStub**: Simplified data persistence layer

## Essential Commands

### Environment Setup
```bash
# Install Python 3.12 and create virtual environment
pyenv local 3.12
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Configuration
```bash
# Configure API keys in .env file
cp .env.example .env  # Edit with your API keys
```

Required environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `DEEPSEEK_API_KEY`
- `GEMINI_API_KEY`
- `API_PROVIDER` (Options: "OPENAI", "CLAUDE", "DEEPSEEK", "GEMINI")

### Running the Generator
```bash
# Basic execution
python mindmap_generator.py

# The input file is configured in main() function:
# Change input_file variable to your target document
```

### Analyzing Outputs
Generated files appear in `mindmap_outputs/` directory:
- `{filename}_mindmap__{provider}.txt` - Mermaid syntax
- `{filename}_mindmap__{provider}.html` - Interactive visualization
- `{filename}_mindmap_outline__{provider}.md` - Markdown outline

## Development Workflow

### Code Structure
- **Single-file architecture**: All logic contained in `mindmap_generator.py` (~6000 lines)
- **Async-first design**: Heavy use of asyncio for concurrent processing
- **Provider abstraction**: Unified interface across different LLM providers
- **Enhanced logging**: Color-coded, stage-specific progress tracking

### Key Processing Stages
1. **Document Type Detection**: Automatically identifies content type (technical, legal, scientific, etc.)
2. **Chunking System**: Intelligent boundary-aware content splitting with overlap
3. **Topic Extraction**: Multi-phase hierarchical concept extraction
4. **Redundancy Detection**: Advanced semantic deduplication using fuzzy matching + LLM analysis
5. **Reality Checking**: Source verification to prevent confabulation
6. **Output Generation**: Multi-format export (Mermaid, HTML, Markdown)

### Testing Strategy
```bash
# Test with provided sample documents
python mindmap_generator.py  # Uses Durnovo memo by default

# Test with small document
# Edit main() to use: "sample_input_document_as_markdown__small.md"
```

### Debugging and Monitoring
The system provides extensive logging with:
- **Color-coded progress indicators**
- **Real-time token usage tracking**
- **Cost projections per provider**
- **Completion ratio monitoring**
- **Quality verification metrics**

### Performance Optimization
- **Token usage tracking**: Detailed cost analysis by task category
- **Caching mechanisms**: Content and emoji selection caching
- **Adaptive processing**: Dynamic resource allocation based on document complexity
- **Early stopping**: Intelligent completion detection to avoid over-processing

## Advanced Features

### Multi-Provider Support
The system dynamically switches between LLM providers:
- **OpenAI**: Efficient, balanced processing (GPT-4o-mini)
- **Anthropic**: Detailed contextual analysis (Claude 3.5 Haiku)
- **DeepSeek**: Comprehensive extraction with cost efficiency
- **Gemini**: Thematic organization focus (Gemini 2.0 Flash Lite)

### Document Type Specialization
Specialized processing for different content types:
- **Technical**: System components, interfaces, implementations
- **Scientific**: Methodology, results, theoretical frameworks
- **Legal**: Principles, rights, obligations, procedures
- **Business**: Strategy, market analysis, implementation plans
- **Academic**: Theoretical frameworks, scholarly arguments
- **Narrative**: Plot, characters, themes

### Quality Assurance
- **Reality Check System**: Prevents AI hallucination by verifying content against source
- **Semantic Redundancy Detection**: Multi-layer deduplication (textual, fuzzy, token-based, LLM-semantic)
- **Hierarchical Verification**: Cross-level consistency checking
- **Confidence Scoring**: Quality metrics influence content inclusion

## Configuration Limits

### Processing Limits (configurable in MindMapGenerator.__init__)
- **max_topics**: 6 (main themes)
- **max_subtopics**: 4 per topic
- **max_details**: 8 per subtopic
- **similarity_threshold**: Prevents duplicate content (topic: 75%, subtopic: 70%, detail: 65%)

### Token Limits by Provider
- **OpenAI**: 8,192 tokens
- **Claude**: 200,000 tokens  
- **DeepSeek**: 8,192 tokens
- **Gemini**: 8,192 tokens

### Cost Tracking
Real-time cost calculation per provider with detailed breakdowns by:
- Task categories (topics, subtopics, details, verification, etc.)
- Token usage (input/output)
- Provider-specific pricing models

## Sample Documents

The repository includes test cases demonstrating cross-provider capabilities:
- **Durnovo Memo**: Historical document predicting WWI and Russian Revolution
- **Small Sample**: Minimal test document for quick validation

Each sample has pre-generated outputs from all four supported providers for comparison.

