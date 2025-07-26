"""
Publishing Models

Data models for the publishing workflow system including publication metadata,
settings, status tracking, and workflow management.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PublicationStatus(Enum):
    """Status of a publication workflow"""
    DRAFT = "draft"
    READY_FOR_REVIEW = "ready_for_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    UNPUBLISHED = "unpublished"
    REJECTED = "rejected"


class PublicationVisibility(Enum):
    """Visibility levels for published books"""
    PRIVATE = "private"
    UNLISTED = "unlisted"
    PUBLIC = "public"
    FEATURED = "featured"


@dataclass
class PublicationMetadata:
    """Publication metadata for a book"""
    book_id: str
    title: str
    subtitle: Optional[str] = None
    description: str = ""
    author_name: str = ""
    author_bio: str = ""
    category: str = "memoir"
    tags: List[str] = field(default_factory=list)
    cover_image_url: Optional[str] = None
    estimated_reading_time: int = 0  # in minutes
    publication_date: Optional[datetime] = None
    isbn: Optional[str] = None
    language: str = "en"
    content_warnings: List[str] = field(default_factory=list)
    target_audience: str = "general"


@dataclass
class PublicationSettings:
    """Publication settings for a book"""
    visibility: PublicationVisibility = PublicationVisibility.PUBLIC
    allow_comments: bool = True
    allow_downloads: bool = True
    price: float = 0.0  # MVP: Free books only
    copyright_notice: str = ""
    license_type: str = "all_rights_reserved"
    marketplace_listing: bool = True
    analytics_enabled: bool = True


@dataclass
class PublishingWorkflow:
    """Publishing workflow instance"""
    workflow_id: str
    book_id: str
    user_id: str
    status: PublicationStatus = PublicationStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    steps_completed: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    publication_metadata: Dict[str, Any] = field(default_factory=dict)
    publication_settings: Dict[str, Any] = field(default_factory=dict)
    publication_url: Optional[str] = None
    marketplace_id: Optional[str] = None
    preview_url: Optional[str] = None
    submitted_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    error_message: Optional[str] = None