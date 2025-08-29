"""Core type definitions for the mindmap generator."""

from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(Enum):
    """Document type classifications for content extraction strategies."""
    
    TECHNICAL = auto()
    SCIENTIFIC = auto()
    LEGAL = auto()
    BUSINESS = auto()
    ACADEMIC = auto()
    NARRATIVE = auto()
    UNKNOWN = auto()


class Topic(BaseModel):
    """A main topic extracted from the document."""
    
    name: str
    emoji: str | None = None
    subtopics: list['Subtopic'] = Field(default_factory=list)
    verified: bool = False
    confidence: float = 1.0


class Subtopic(BaseModel):
    """A subtopic under a main topic."""
    
    name: str
    emoji: str | None = None
    details: list['Detail'] = Field(default_factory=list)
    verified: bool = False
    parent_topic: str | None = None


class Detail(BaseModel):
    """A specific detail under a subtopic."""
    
    text: str
    importance: str = "medium"  # high, medium, low
    emoji: str | None = None
    verified: bool = False
    parent_subtopic: str | None = None


class MindmapData(BaseModel):
    """Complete mindmap data structure."""
    
    central_theme: Topic
    document_type: DocumentType
    total_topics: int = 0
    total_subtopics: int = 0
    total_details: int = 0
    generation_metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    
    index: int
    start_position: int
    end_position: int
    overlap_size: int = 0
    hash: str | None = None
    processed: bool = False


class ExtractionResult(BaseModel):
    """Result from an extraction operation."""
    
    success: bool
    data: Any | None = None
    error: str | None = None
    tokens_used: dict[str, int] = Field(default_factory=dict)
    duration_ms: float | None = None


# Enable forward references
Topic.model_rebuild()
Subtopic.model_rebuild()