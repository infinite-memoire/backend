"""
AI Processing System Package

This package contains the complete multi-agent AI processing pipeline for
converting audio transcripts into organized book chapters.

Components:
- semantic_chunker: Semantic text analysis and chunking
- graph_builder: Storyline graph construction and Neo4j integration  
- agents: Multi-agent system for content generation and harmonization
- orchestrator: Central coordination and workflow management

Usage:
    from ai_implementation import orchestrator
    
    session_id = await orchestrator.process_transcript(
        transcript="...",
        user_id="user123",
        user_preferences={"writing_style": "memoir"}
    )
    
    # Check status
    status = await orchestrator.get_session_status(session_id)
    
    # Get results when complete
    results = await orchestrator.get_session_results(session_id)
"""

from .semantic_chunker import SemanticChunker, SemanticChunk, ProcessedTranscript
from .graph_builder import GraphBuilder, StorylineNode, RelationshipEdge
from .agents import (
    ChapterWriterAgent,
    ChapterHarmonizerAgent, 
    FollowupQuestionsAgent,
    TaskRequest,
    TaskResult,
    AgentMessage
)
from .orchestrator import AgentOrchestrator, ProcessingSession, ProgressUpdate

# Global instances for easy access
semantic_chunker = SemanticChunker()
graph_builder = GraphBuilder()
orchestrator = AgentOrchestrator()

__all__ = [
    # Classes
    "SemanticChunker",
    "SemanticChunk", 
    "ProcessedTranscript",
    "GraphBuilder",
    "StorylineNode",
    "RelationshipEdge",
    "ChapterWriterAgent",
    "ChapterHarmonizerAgent",
    "FollowupQuestionsAgent",
    "AgentOrchestrator",
    "ProcessingSession",
    "ProgressUpdate",
    "TaskRequest",
    "TaskResult",
    "AgentMessage",
    
    # Global instances
    "semantic_chunker",
    "graph_builder", 
    "orchestrator"
]