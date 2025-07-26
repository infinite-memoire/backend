"""
AI Processing Data Models
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class ProcessingStage(str, Enum):
    """Processing stage enumeration"""
    INITIALIZATION = "initialization"
    SEMANTIC_PROCESSING = "semantic_processing"
    GRAPH_CONSTRUCTION = "graph_construction"
    AGENT_COORDINATION = "agent_coordination"
    CONTENT_GENERATION = "content_generation"
    QUALITY_ASSURANCE = "quality_assurance"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"

class AISessionModel(BaseModel):
    """AI processing session model for Firestore"""
    session_id: str
    user_id: str
    audio_session_id: Optional[str] = None
    transcript_length: int
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    current_stage: ProcessingStage = ProcessingStage.INITIALIZATION
    progress_percentage: float = 0.0
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "processing"
    error_message: Optional[str] = None

class SemanticChunkModel(BaseModel):
    """Semantic chunk model"""
    id: str
    content: str
    start_position: float
    end_position: float
    word_count: int
    speaker: Optional[str] = None
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    temporal_info: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

class StorylineModel(BaseModel):
    """Storyline model"""
    id: str
    summary: str
    temporal_info: Dict[str, Any] = Field(default_factory=dict)
    participants: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    centrality_score: float = 0.0
    confidence: float = 0.0
    entity_density: float = 0.0
    main_storyline: bool = False

class ChapterModel(BaseModel):
    """Generated chapter model"""
    storyline_id: str
    title: str
    content: str
    word_count: int
    quality_score: float
    source_chunks: List[str] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    harmonization_applied: bool = False
    changes_applied: int = 0

class FollowupQuestionModel(BaseModel):
    """Follow-up question model"""
    id: str
    storyline_id: Optional[str] = None
    category: str
    question: str
    context: str
    priority_score: float
    reasoning: str

class QuestionAnswerModel(BaseModel):
    """User answer to follow-up question"""
    session_id: str
    question_id: str
    answer: str
    confidence: float = 1.0
    answered_at: datetime
    processed: bool = False

class ProcessingSummaryModel(BaseModel):
    """Processing summary model"""
    session_id: str
    user_id: str
    processing_time_seconds: float
    transcript_stats: Dict[str, Any]
    graph_stats: Dict[str, Any]
    content_stats: Dict[str, Any]
    interaction_stats: Dict[str, Any]
    harmonization_applied: bool
    errors_encountered: int

class UserPreferencesModel(BaseModel):
    """User preferences for AI processing"""
    writing_style: str = "memoir"
    tone: str = "reflective"
    perspective: str = "first_person"
    chunk_size: int = 20
    preserve_speakers: bool = True
    merge_similar: bool = True
    target_word_count: int = 1200
    quality_tier: str = "accurate"  # "fast" or "accurate"

class ProcessingConfigModel(BaseModel):
    """Configuration for AI processing"""
    anthropic_api_key: Optional[str] = None
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    max_concurrent_sessions: int = 5
    session_timeout_hours: int = 4
    checkpoint_interval_minutes: int = 5