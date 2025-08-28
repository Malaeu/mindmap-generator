# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced Python-based mindmap generator that transforms documents into hierarchical knowledge graphs using multiple LLM providers (OpenAI, Anthropic, DeepSeek, Gemini). The system employs a unique non-directional graph exploration model with reality checking and multi-layered redundancy detection.

## Development Setup & Commands

```bash
# Environment setup (Python 3.12 currently, target 3.13 for migration)
python -m venv .venv
source .venv/bin/activate

# Install dependencies (MIGRATE TO UV + pyproject.toml)
pip install -r requirements.txt

# Run the generator
python mindmap_generator.py

# Output location
# Generated files appear in mindmap_outputs/ as .txt (Mermaid), .html (interactive), .md (outline)
```

## Architecture & Code Structure

### Current State (NEEDS REFACTORING)
- **MONOLITHIC**: Single 4,439-line file `mindmap_generator.py` (~207KB)
- **NO TESTS**: No testing framework implemented
- **LEGACY DEPS**: Uses requirements.txt instead of pyproject.toml
- **NO LINTING CONFIG**: ruff installed but not configured

### Core Components in mindmap_generator.py

1. **MindMapGenerator** (main orchestrator)
   - Async processing with aiofiles/aiolimiter
   - Multi-stage document processing pipeline
   - Generates Mermaid diagrams with interactive HTML

2. **DocumentOptimizer** (LLM client manager)
   - Multi-provider support (OpenAI, Anthropic, DeepSeek, Gemini)
   - Provider-specific prompt optimization
   - Token usage tracking per operation type

3. **TokenUsageTracker** (metrics)
   - Tracks tokens by category: topics, subtopics, details, similarity, verification, emoji
   - Cost calculation per provider
   - Performance statistics

4. **DocumentType** (classification system)
   - Technical, Scientific, Legal, Business, Academic, Narrative
   - Type-specific processing strategies

5. **MinimalDatabaseStub** (persistence)
   - SQLModel/aiosqlite for lightweight storage
   - Caches processed chunks and relationships

### Unique Features

- **Non-directional graph exploration**: Explores multiple document aspects in parallel
- **Reality checking**: Verifies generated content against source to prevent hallucinations  
- **Multi-layer redundancy detection**: Text similarity, fuzzy matching, token analysis, LLM semantic checking
- **Overlapping chunks**: Preserves context across document segments
- **Progressive refinement**: Builds understanding iteratively

## Configuration

### Environment Variables (.env)
```ini
API_PROVIDER="OPENAI"        # Active provider: OPENAI, ANTHROPIC, DEEPSEEK, GEMINI
OPENAI_API_KEY="sk-..."     # Required API keys
ANTHROPIC_API_KEY="..."
DEEPSEEK_API_KEY="..."  
GEMINI_API_KEY="..."
```

### Input Configuration
Edit `main()` function to set input file:
```python
input_file = "sample_input_document_as_markdown__durnovo_memo.md"
```

## Critical Migration Tasks

### PRIORITY 1: Modernize Python Setup
```bash
# Switch to Python 3.13 + uv + pyproject.toml
uv init
uv add [all dependencies from requirements.txt]
uv run ruff check --fix --unsafe-fixes
```

### PRIORITY 2: Break Up Monolith
Target structure:
```
src/
├── core/
│   ├── generator.py       # MindMapGenerator class
│   ├── optimizer.py       # DocumentOptimizer
│   ├── tracker.py         # TokenUsageTracker
│   └── types.py          # DocumentType, data models
├── processors/
│   ├── chunking.py       # Document chunking logic
│   ├── similarity.py     # Redundancy detection
│   └── verification.py   # Reality checking
├── providers/
│   ├── base.py          # Abstract provider
│   ├── openai.py        # OpenAI implementation
│   ├── anthropic.py     # Anthropic implementation
│   └── ...
└── utils/
    ├── logging.py        # Rich console output
    └── database.py       # Database operations
```

### PRIORITY 3: Add Testing
```bash
# Setup pytest with proper fixtures
uv add --dev pytest pytest-asyncio pytest-mock
# Create tests/ directory with comprehensive test coverage
```

## Working with the Codebase

### Key Processing Flow
1. Document ingestion → Type detection
2. Smart chunking with overlap
3. Parallel topic extraction per chunk  
4. Subtopic generation with redundancy checking
5. Detail extraction with semantic deduplication
6. Reality verification against source
7. Mermaid diagram generation
8. HTML visualization rendering

### Output Files
- `{filename}_mindmap_{provider}.txt` - Raw Mermaid syntax
- `{filename}_mindmap_{provider}.html` - Interactive visualization
- `{filename}_mindmap_{provider}.md` - Markdown outline

### Provider-Specific Behaviors
- **OpenAI**: Best for technical documents, structured output
- **Anthropic**: Superior context understanding, nuanced topics
- **DeepSeek**: Cost-effective, good for large documents
- **Gemini**: Strong at creative/narrative content

## Important Notes

- NEVER delete files without permission
- Always preserve the non-directional exploration architecture
- Maintain backwards compatibility during refactoring
- Use Rich library for all console output
- Test with sample documents before major changes
- Token tracking is critical - preserve all metrics

## CRITICAL: Problem Analysis Protocol

When code doesn't work - STOP! Don't simplify, ANALYZE using this template:

### 1. Show EXACT problematic code (not entire file)
### 2. List symptoms (hanging/error/wrong result) 
### 3. Ask KEY QUESTIONS:
   - Computational complexity (O(n) vs O(n²)?)
   - Mathematical correctness (right formulas/signs/normalization?)
   - Numerical stability (small divisions/cancellation?)
   - Conceptual issues (do we understand the problem?)

### 4. Provide DEBUG DATA (intermediate values)
### 5. Form HYPOTHESES (ranked by probability)
### 6. Ask Ylsha SPECIFIC QUESTIONS with context

**DON'T**: Simplify code / Replace methods / Ignore warnings / Make assumptions
**DO**: Show exact problem / Give concrete numbers / Form testable hypotheses / Ask right questions

See PROBLEM_ANALYSIS_TEMPLATE.md for detailed protocol.