"""
Content Models for Output Management

Data models for books, chapters, and content metadata used throughout
the output management and publishing system.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class ChapterStatus(str, Enum):
    """Status of a chapter in the workflow"""
    DRAFT = "draft"
    GENERATED = "generated"
    REVIEWED = "reviewed"
    PUBLISHED = "published"
    FAILED = "failed"


class BookStatus(str, Enum):
    """Status of a book"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PUBLISHED = "published"


class ContentMetadata(BaseModel):
    """Metadata for content pieces"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    version: str = "1.0"
    source: Optional[str] = None


class SourceReferences(BaseModel):
    """References to source material for content generation"""
    audio_session_ids: List[str] = Field(default_factory=list)
    transcript_chunk_ids: List[str] = Field(default_factory=list)
    storyline_node_ids: List[str] = Field(default_factory=list)
    source_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ChapterContent(BaseModel):
    """Content of a chapter"""
    markdown_text: str = Field(..., min_length=1)
    word_count: Optional[int] = None
    themes: List[str] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.word_count is None:
            self.word_count = len(self.markdown_text.split())


class ChapterModel(BaseModel):
    """Model for a book chapter"""
    id: Optional[str] = None
    book_id: str = Field(..., min_length=1)
    book_version: str = Field(default="v1.0")
    chapter_number: int = Field(..., ge=1)
    title: str = Field(..., min_length=1, max_length=200)
    content: ChapterContent
    status: ChapterStatus = ChapterStatus.DRAFT
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    generation_agent: Optional[str] = None
    source_references: SourceReferences = Field(default_factory=SourceReferences)
    generation_metadata: Optional[Dict[str, Any]] = None
    
    @validator('chapter_number')
    def validate_chapter_number(cls, v):
        if v < 1:
            raise ValueError('Chapter number must be positive')
        return v
    
    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('Chapter title cannot be empty')
        return v.strip()


class BookSettings(BaseModel):
    """Settings for a book"""
    auto_generate_metadata: bool = True
    default_template: str = "book.html5"
    enable_analytics: bool = True
    allow_public_access: bool = False
    content_warnings: List[str] = Field(default_factory=list)
    target_audience: str = "general"
    language: str = "en"


class BookVersion(BaseModel):
    """Version information for a book"""
    created_at: datetime = Field(default_factory=datetime.now)
    chapter_count: int = Field(default=0, ge=0)
    total_word_count: int = Field(default=0, ge=0)
    status: str = "active"
    base_version: Optional[str] = None


class BookModel(BaseModel):
    """Model for a book"""
    id: Optional[str] = None
    owner_user_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(default="", max_length=2000)
    status: BookStatus = BookStatus.DRAFT
    current_version: str = Field(default="v1.0")
    versions: Dict[str, BookVersion] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    settings: Optional[BookSettings] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize default version if not provided
        if not self.versions:
            self.versions = {
                self.current_version: BookVersion()
            }
        # Initialize default settings if not provided
        if self.settings is None:
            self.settings = BookSettings()
    
    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('Book title cannot be empty')
        return v.strip()
    
    @validator('owner_user_id')
    def validate_owner(cls, v):
        if not v.strip():
            raise ValueError('Owner user ID cannot be empty')
        return v.strip()


class BookStatistics(BaseModel):
    """Statistics for a book"""
    book_id: str
    title: str
    total_chapters: int = 0
    total_words: int = 0
    estimated_reading_time: int = 0  # in minutes
    average_quality_score: float = 0.0
    current_version: str = "v1.0"
    total_versions: int = 1
    created_at: datetime
    last_updated: datetime


class ChapterCreateRequest(BaseModel):
    """Request model for creating a chapter"""
    book_id: str = Field(..., min_length=1)
    chapter_number: int = Field(..., ge=1)
    title: str = Field(..., min_length=1, max_length=200)
    markdown_text: str = Field(..., min_length=1)
    themes: List[str] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ChapterUpdateRequest(BaseModel):
    """Request model for updating a chapter"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    markdown_text: Optional[str] = Field(None, min_length=1)
    themes: Optional[List[str]] = None
    participants: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    status: Optional[ChapterStatus] = None


class BookCreateRequest(BaseModel):
    """Request model for creating a book"""
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(default="", max_length=2000)
    settings: Optional[BookSettings] = None


class BookUpdateRequest(BaseModel):
    """Request model for updating a book"""
    title: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[BookStatus] = None
    settings: Optional[BookSettings] = None


class BookSearchRequest(BaseModel):
    """Request model for searching books"""
    query: str = Field(..., min_length=1, max_length=100)
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class BookListResponse(BaseModel):
    """Response model for book listings"""
    books: List[BookModel]
    total_count: int
    page_size: int
    page_number: int
    has_more: bool


class ChapterListResponse(BaseModel):
    """Response model for chapter listings"""
    chapters: List[ChapterModel]
    book_id: str
    book_version: str
    total_count: int


# Utility functions for model conversion
def dict_to_book_model(data: Dict[str, Any]) -> BookModel:
    """Convert dictionary to BookModel"""
    # Convert versions dict
    if "versions" in data:
        versions = {}
        for version_key, version_data in data["versions"].items():
            if isinstance(version_data, dict):
                versions[version_key] = BookVersion(**version_data)
            else:
                versions[version_key] = version_data
        data["versions"] = versions
    
    # Convert settings
    if "settings" in data and isinstance(data["settings"], dict):
        data["settings"] = BookSettings(**data["settings"])
    
    return BookModel(**data)


def dict_to_chapter_model(data: Dict[str, Any]) -> ChapterModel:
    """Convert dictionary to ChapterModel"""
    # Convert content
    if "content" in data and isinstance(data["content"], dict):
        data["content"] = ChapterContent(**data["content"])
    
    # Convert source references
    if "source_references" in data and isinstance(data["source_references"], dict):
        data["source_references"] = SourceReferences(**data["source_references"])
    
    return ChapterModel(**data)


def book_model_to_dict(book: BookModel) -> Dict[str, Any]:
    """Convert BookModel to dictionary for storage"""
    data = book.dict()
    
    # Convert datetime objects
    for field in ["created_at", "updated_at"]:
        if field in data and isinstance(data[field], datetime):
            data[field] = data[field]
    
    # Convert versions
    if "versions" in data:
        versions = {}
        for version_key, version_data in data["versions"].items():
            if isinstance(version_data, BookVersion):
                versions[version_key] = version_data.dict()
            else:
                versions[version_key] = version_data
        data["versions"] = versions
    
    return data


def chapter_model_to_dict(chapter: ChapterModel) -> Dict[str, Any]:
    """Convert ChapterModel to dictionary for storage"""
    data = chapter.dict()
    
    # Convert nested models
    if "content" in data and isinstance(data["content"], dict):
        # Content is already a dict from Pydantic
        pass
    
    if "source_references" in data and isinstance(data["source_references"], dict):
        # Source references is already a dict from Pydantic
        pass
    
    return data