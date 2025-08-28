"""Basic tests for the agentic mindmap system."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.core.types import DocumentType, Topic, Subtopic, Detail
from src.agents.document_agent import DocumentAnalyzer
from src.agents.extraction_agent import TopicExtractor
from src.agents.verification_agent import VerificationAgent
from src.agents.visualization_agent import VisualizationAgent


# Create a mock LLM provider for testing
def create_mock_provider():
    """Create a mock LLM provider."""
    mock = MagicMock()
    mock.name = "MockProvider"
    return mock


# Sample document for testing
SAMPLE_DOC = """
# Software Architecture Document

## System Overview
Our application is a cloud-based solution designed for enterprise customers.
It provides real-time data processing, advanced analytics, and secure storage.

## Key Components

### API Gateway
The API Gateway handles all incoming requests and routes them to appropriate services.
It implements rate limiting, authentication, and request validation.

### Database Layer
We use PostgreSQL for relational data and Redis for caching.
The database is configured with master-slave replication for high availability.

### Processing Engine
The processing engine uses Apache Spark for distributed computing.
It can handle millions of events per second with low latency.

## Security Features
- End-to-end encryption for all data transmissions
- Multi-factor authentication for user access
- Regular security audits and penetration testing
"""


@pytest.mark.asyncio
async def test_document_analyzer():
    """Test document analysis and chunking."""
    analyzer = DocumentAnalyzer(llm_provider=create_mock_provider())
    
    result = await analyzer.process(SAMPLE_DOC)
    
    assert result.success
    assert result.data['document_type'] == DocumentType.TECHNICAL
    assert len(result.data['chunks']) > 0
    assert result.data['document_length'] == len(SAMPLE_DOC)


@pytest.mark.asyncio
async def test_topic_extractor():
    """Test topic extraction."""
    extractor = TopicExtractor(llm_provider=create_mock_provider())
    
    result = await extractor.process({
        'content': SAMPLE_DOC,
        'document_type': DocumentType.TECHNICAL
    })
    
    assert result.success
    assert len(result.data['topics']) >= 2
    assert isinstance(result.data['topics'][0], Topic)


@pytest.mark.asyncio
async def test_verification_agent():
    """Test content verification and deduplication."""
    agent = VerificationAgent(llm_provider=create_mock_provider())
    
    # Create sample mindmap with duplicates
    mindmap = {
        'central_theme': {
            'name': 'Test',
            'subtopics': [
                {
                    'name': 'Component A',
                    'subtopics': [
                        {'name': 'Subcomponent 1', 'details': []},
                        {'name': 'Subcomponent 1', 'details': []},  # Duplicate
                    ]
                },
                {
                    'name': 'Component A',  # Duplicate topic
                    'subtopics': []
                }
            ]
        }
    }
    
    result = await agent.process({
        'content': mindmap,
        'source': SAMPLE_DOC
    })
    
    assert result.success
    
    # Check deduplication worked
    processed = result.data['mindmap']['central_theme']
    assert len(processed['subtopics']) == 1  # Should remove duplicate topic
    assert len(processed['subtopics'][0]['subtopics']) == 1  # Should remove duplicate subtopic


@pytest.mark.asyncio
async def test_visualization_agent():
    """Test Mermaid and HTML generation."""
    agent = VisualizationAgent(llm_provider=create_mock_provider())
    
    mindmap = {
        'central_theme': {
            'name': 'Test Document',
            'subtopics': [
                {
                    'name': 'Topic 1',
                    'emoji': '🎯',
                    'subtopics': [
                        {
                            'name': 'Subtopic 1',
                            'emoji': '📌',
                            'details': [
                                {
                                    'text': 'Detail about subtopic',
                                    'importance': 'high',
                                    'emoji': '⭐'
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }
    
    result = await agent.process({'mindmap': mindmap})
    
    assert result.success
    assert 'mindmap' in result.data['mermaid']
    assert 'html' in result.data
    assert 'Topic 1' in result.data['markdown']  # Check topic name is in markdown
    assert result.data['node_count']['topics'] == 1
    assert result.data['node_count']['subtopics'] == 1
    assert result.data['node_count']['details'] == 1


def test_types():
    """Test type definitions."""
    topic = Topic(name="Test Topic")
    assert topic.name == "Test Topic"
    assert topic.verified == False
    assert topic.confidence == 1.0
    assert len(topic.subtopics) == 0
    
    subtopic = Subtopic(name="Test Subtopic", parent_topic="Test Topic")
    assert subtopic.name == "Test Subtopic"
    assert subtopic.parent_topic == "Test Topic"
    
    detail = Detail(text="Test detail text", importance="high")
    assert detail.text == "Test detail text"
    assert detail.importance == "high"


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_document_analyzer())
    asyncio.run(test_topic_extractor())
    asyncio.run(test_verification_agent())
    asyncio.run(test_visualization_agent())
    test_types()
    
    print("✅ All tests passed!")