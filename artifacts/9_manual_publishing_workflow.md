# Manual Publishing Workflow Implementation

## Publishing Workflow Architecture

### Publishing State Management
```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

class PublicationStatus(Enum):
    DRAFT = "draft"
    READY_FOR_REVIEW = "ready_for_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    UNPUBLISHED = "unpublished"
    REJECTED = "rejected"

class PublicationVisibility(Enum):
    PRIVATE = "private"
    UNLISTED = "unlisted"
    PUBLIC = "public"
    FEATURED = "featured"

@dataclass
class PublicationMetadata:
    book_id: str
    title: str
    subtitle: Optional[str] = None
    description: str = ""
    author_name: str = ""
    author_bio: str = ""
    category: str = "memoir"
    tags: List[str] = None
    cover_image_url: Optional[str] = None
    estimated_reading_time: int = 0  # in minutes
    publication_date: Optional[datetime] = None
    isbn: Optional[str] = None
    language: str = "en"
    content_warnings: List[str] = None
    target_audience: str = "general"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.content_warnings is None:
            self.content_warnings = []

@dataclass
class PublicationSettings:
    visibility: PublicationVisibility
    allow_comments: bool = True
    allow_downloads: bool = True
    price: float = 0.0  # MVP: Free books only
    copyright_notice: str = ""
    license_type: str = "all_rights_reserved"
    marketplace_listing: bool = True
    analytics_enabled: bool = True
```

### Publishing Workflow Service
```python
class PublishingWorkflowService:
    def __init__(self, storage_service, conversion_service, validation_service):
        self.storage = storage_service
        self.converter = conversion_service
        self.validator = validation_service
    
    async def initiate_publishing_workflow(self, book_id: str, user_id: str) -> str:
        """Start the manual publishing workflow for a book"""
        
        # Validate book ownership
        book = await self.storage.get_book_metadata(book_id)
        if book["owner_user_id"] != user_id:
            raise PermissionError("User does not own this book")
        
        # Check if book has completed chapters
        chapters = await self.storage.get_book_chapters(book_id, book["current_version"])
        if not chapters:
            raise ValueError("Book must have at least one chapter to publish")
        
        # Create publishing workflow instance
        workflow_id = f"pub_{book_id}_{int(datetime.now().timestamp())}"
        
        workflow_data = {
            "workflow_id": workflow_id,
            "book_id": book_id,
            "user_id": user_id,
            "status": PublicationStatus.DRAFT.value,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "steps_completed": [],
            "validation_results": {},
            "publication_metadata": {},
            "publication_settings": {}
        }
        
        await self.storage.create_publishing_workflow(workflow_data)
        return workflow_id
    
    async def update_publication_metadata(self, 
                                         workflow_id: str, 
                                         metadata: PublicationMetadata) -> dict:
        """Update the publication metadata for a book"""
        
        workflow = await self.storage.get_publishing_workflow(workflow_id)
        
        # Validate metadata
        validation_result = await self._validate_publication_metadata(metadata)
        
        if validation_result["is_valid"]:
            workflow["publication_metadata"] = metadata.__dict__
            workflow["steps_completed"].append("metadata_updated")
            workflow["updated_at"] = datetime.now()
            
            await self.storage.update_publishing_workflow(workflow)
            
            return {
                "status": "success",
                "message": "Publication metadata updated successfully",
                "validation_result": validation_result
            }
        else:
            return {
                "status": "validation_failed",
                "message": "Metadata validation failed",
                "validation_result": validation_result
            }
    
    async def update_publication_settings(self, 
                                         workflow_id: str, 
                                         settings: PublicationSettings) -> dict:
        """Update the publication settings for a book"""
        
        workflow = await self.storage.get_publishing_workflow(workflow_id)
        
        workflow["publication_settings"] = settings.__dict__
        workflow["steps_completed"].append("settings_updated")
        workflow["updated_at"] = datetime.now()
        
        await self.storage.update_publishing_workflow(workflow)
        
        return {
            "status": "success",
            "message": "Publication settings updated successfully"
        }
    
    async def preview_publication(self, workflow_id: str) -> dict:
        """Generate a preview of how the book will appear when published"""
        
        workflow = await self.storage.get_publishing_workflow(workflow_id)
        book_id = workflow["book_id"]
        
        # Get book content
        book_metadata = await self.storage.get_book_metadata(book_id)
        chapters = await self.storage.get_book_chapters(book_id, book_metadata["current_version"])
        
        # Generate HTML preview
        html_content = await self.converter.convert_book_to_html(
            chapters=chapters,
            book_metadata={**book_metadata, **workflow["publication_metadata"]},
            template_name="marketplace_preview.html5"
        )
        
        # Store preview
        preview_id = f"preview_{workflow_id}_{int(datetime.now().timestamp())}"
        preview_path = f"./previews/{preview_id}.html"
        
        Path(preview_path).parent.mkdir(exist_ok=True)
        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        workflow["steps_completed"].append("preview_generated")
        workflow["preview_url"] = f"/preview/{preview_id}"
        await self.storage.update_publishing_workflow(workflow)
        
        return {
            "preview_url": f"/preview/{preview_id}",
            "preview_id": preview_id,
            "status": "success"
        }
    
    async def submit_for_publication(self, workflow_id: str) -> dict:
        """Submit the book for publication after all checks"""
        
        workflow = await self.storage.get_publishing_workflow(workflow_id)
        
        # Run final validation
        validation_results = await self._run_publication_validation(workflow)
        
        if not validation_results["is_ready_for_publication"]:
            return {
                "status": "validation_failed",
                "message": "Book is not ready for publication",
                "validation_results": validation_results
            }
        
        # Update workflow status
        workflow["status"] = PublicationStatus.READY_FOR_REVIEW.value
        workflow["steps_completed"].append("submitted_for_publication")
        workflow["validation_results"] = validation_results
        workflow["submitted_at"] = datetime.now()
        
        await self.storage.update_publishing_workflow(workflow)
        
        # For MVP: Auto-approve since there's no manual review process
        return await self._auto_approve_publication(workflow_id)
    
    async def _auto_approve_publication(self, workflow_id: str) -> dict:
        """Auto-approve publication for MVP (no manual review)"""
        
        workflow = await self.storage.get_publishing_workflow(workflow_id)
        
        try:
            # Generate final HTML version
            final_html = await self._generate_final_publication(workflow)
            
            # Create marketplace entry
            marketplace_entry = await self._create_marketplace_entry(workflow, final_html)
            
            # Update book status
            await self._update_book_publication_status(workflow["book_id"], "published")
            
            # Update workflow
            workflow["status"] = PublicationStatus.PUBLISHED.value
            workflow["published_at"] = datetime.now()
            workflow["marketplace_id"] = marketplace_entry["id"]
            workflow["publication_url"] = marketplace_entry["url"]
            
            await self.storage.update_publishing_workflow(workflow)
            
            return {
                "status": "published",
                "message": "Book has been successfully published",
                "publication_url": marketplace_entry["url"],
                "marketplace_id": marketplace_entry["id"]
            }
            
        except Exception as e:
            # Handle publication failure
            workflow["status"] = PublicationStatus.REJECTED.value
            workflow["error_message"] = str(e)
            await self.storage.update_publishing_workflow(workflow)
            
            return {
                "status": "publication_failed",
                "message": f"Publication failed: {str(e)}"
            }
```

### Publication Validation System
```python
class PublicationValidator:
    def __init__(self, storage_service):
        self.storage = storage_service
    
    async def validate_publication_metadata(self, metadata: PublicationMetadata) -> dict:
        """Validate publication metadata completeness and quality"""
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "score": 100
        }
        
        # Required field validation
        if not metadata.title or len(metadata.title.strip()) < 3:
            validation_results["errors"].append("Title is required and must be at least 3 characters")
            validation_results["is_valid"] = False
            validation_results["score"] -= 20
        
        if not metadata.description or len(metadata.description.strip()) < 50:
            validation_results["errors"].append("Description is required and must be at least 50 characters")
            validation_results["is_valid"] = False
            validation_results["score"] -= 15
        
        if not metadata.author_name or len(metadata.author_name.strip()) < 2:
            validation_results["errors"].append("Author name is required")
            validation_results["is_valid"] = False
            validation_results["score"] -= 10
        
        # Quality recommendations (warnings)
        if not metadata.subtitle:
            validation_results["warnings"].append("Consider adding a subtitle for better discoverability")
            validation_results["score"] -= 5
        
        if len(metadata.tags) < 3:
            validation_results["warnings"].append("Consider adding more tags (3-10 recommended)")
            validation_results["score"] -= 5
        
        if not metadata.cover_image_url:
            validation_results["warnings"].append("Cover image will improve book appeal")
            validation_results["score"] -= 10
        
        if len(metadata.description) < 200:
            validation_results["warnings"].append("Longer descriptions tend to perform better (200+ chars recommended)")
            validation_results["score"] -= 5
        
        return validation_results
    
    async def validate_book_content(self, book_id: str, version: str) -> dict:
        """Validate the book content for publication readiness"""
        
        chapters = await self.storage.get_book_chapters(book_id, version)
        
        validation_results = {
            "is_ready": True,
            "issues": [],
            "recommendations": [],
            "stats": {
                "total_chapters": len(chapters),
                "total_word_count": 0,
                "average_chapter_length": 0,
                "estimated_reading_time": 0
            }
        }
        
        if len(chapters) == 0:
            validation_results["issues"].append("Book must have at least one chapter")
            validation_results["is_ready"] = False
            return validation_results
        
        total_words = 0
        for chapter in chapters:
            content = chapter.get("content", {}).get("markdown_text", "")
            word_count = len(content.split())
            total_words += word_count
            
            # Chapter-specific validation
            if word_count < 100:
                validation_results["issues"].append(
                    f"Chapter '{chapter.get('title', 'Untitled')}' is very short ({word_count} words)"
                )
            
            if not chapter.get("title"):
                validation_results["issues"].append("All chapters should have titles")
        
        validation_results["stats"]["total_word_count"] = total_words
        validation_results["stats"]["average_chapter_length"] = total_words // len(chapters)
        validation_results["stats"]["estimated_reading_time"] = total_words // 200  # 200 WPM
        
        # Content quality recommendations
        if total_words < 5000:
            validation_results["recommendations"].append(
                "Consider expanding content. Books under 5,000 words may appear incomplete"
            )
        
        if len(chapters) == 1:
            validation_results["recommendations"].append(
                "Consider breaking long content into multiple chapters for better readability"
            )
        
        return validation_results
```

### Marketplace Entry Management
```python
class MarketplaceService:
    def __init__(self, storage_service):
        self.storage = storage_service
    
    async def create_marketplace_entry(self, workflow: dict, html_content: str) -> dict:
        """Create a new marketplace entry for a published book"""
        
        marketplace_id = f"book_{workflow['book_id']}_{int(datetime.now().timestamp())}"
        
        # Extract metadata
        metadata = workflow["publication_metadata"]
        settings = workflow["publication_settings"]
        
        marketplace_entry = {
            "id": marketplace_id,
            "book_id": workflow["book_id"],
            "title": metadata["title"],
            "subtitle": metadata.get("subtitle"),
            "description": metadata["description"],
            "author_name": metadata["author_name"],
            "author_bio": metadata.get("author_bio", ""),
            "category": metadata.get("category", "memoir"),
            "tags": metadata.get("tags", []),
            "cover_image_url": metadata.get("cover_image_url"),
            "estimated_reading_time": metadata.get("estimated_reading_time", 0),
            "language": metadata.get("language", "en"),
            "content_warnings": metadata.get("content_warnings", []),
            "target_audience": metadata.get("target_audience", "general"),
            
            "visibility": settings["visibility"],
            "allow_comments": settings.get("allow_comments", True),
            "allow_downloads": settings.get("allow_downloads", True),
            "price": settings.get("price", 0.0),
            "copyright_notice": settings.get("copyright_notice", ""),
            "license_type": settings.get("license_type", "all_rights_reserved"),
            
            "publication_date": datetime.now(),
            "view_count": 0,
            "download_count": 0,
            "rating_average": 0.0,
            "rating_count": 0,
            "featured": False,
            
            "html_content": html_content,
            "status": "active"
        }
        
        await self.storage.create_marketplace_entry(marketplace_entry)
        
        return {
            "id": marketplace_id,
            "url": f"/marketplace/book/{marketplace_id}",
            "reading_url": f"/read/{marketplace_id}"
        }
    
    async def update_marketplace_visibility(self, marketplace_id: str, 
                                          visibility: PublicationVisibility) -> dict:
        """Update the visibility of a marketplace entry"""
        
        entry = await self.storage.get_marketplace_entry(marketplace_id)
        entry["visibility"] = visibility.value
        entry["updated_at"] = datetime.now()
        
        await self.storage.update_marketplace_entry(entry)
        
        return {
            "status": "success",
            "message": f"Visibility updated to {visibility.value}"
        }
    
    async def unpublish_book(self, marketplace_id: str, user_id: str) -> dict:
        """Unpublish a book from the marketplace"""
        
        entry = await self.storage.get_marketplace_entry(marketplace_id)
        
        # Verify ownership through book
        book = await self.storage.get_book_metadata(entry["book_id"])
        if book["owner_user_id"] != user_id:
            raise PermissionError("User does not own this book")
        
        entry["status"] = "unpublished"
        entry["unpublished_at"] = datetime.now()
        
        await self.storage.update_marketplace_entry(entry)
        
        return {
            "status": "success",
            "message": "Book has been unpublished from marketplace"
        }
```

### Publishing API Routes
```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(tags=["Publishing"])

class PublicationMetadataRequest(BaseModel):
    title: str
    subtitle: Optional[str] = None
    description: str
    author_name: str
    author_bio: str = ""
    category: str = "memoir"
    tags: List[str] = []
    cover_image_url: Optional[str] = None
    language: str = "en"
    content_warnings: List[str] = []
    target_audience: str = "general"

class PublicationSettingsRequest(BaseModel):
    visibility: str = "public"
    allow_comments: bool = True
    allow_downloads: bool = True
    price: float = 0.0
    copyright_notice: str = ""
    license_type: str = "all_rights_reserved"
    marketplace_listing: bool = True

@router.post("/start-publishing/{book_id}")
async def start_publishing_workflow(book_id: str, current_user=Depends(get_current_user)) -> dict:
    """Start the publishing workflow for a book"""
    
    try:
        workflow_id = await publishing_service.initiate_publishing_workflow(
            book_id, current_user.id
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Publishing workflow initiated"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/publishing/{workflow_id}/metadata")
async def update_publication_metadata(
    workflow_id: str,
    metadata_request: PublicationMetadataRequest,
    current_user=Depends(get_current_user)
) -> dict:
    """Update publication metadata"""
    
    metadata = PublicationMetadata(**metadata_request.dict())
    result = await publishing_service.update_publication_metadata(workflow_id, metadata)
    
    return result

@router.put("/publishing/{workflow_id}/settings")
async def update_publication_settings(
    workflow_id: str,
    settings_request: PublicationSettingsRequest,
    current_user=Depends(get_current_user)
) -> dict:
    """Update publication settings"""
    
    settings = PublicationSettings(**settings_request.dict())
    result = await publishing_service.update_publication_settings(workflow_id, settings)
    
    return result

@router.post("/publishing/{workflow_id}/preview")
async def generate_publication_preview(
    workflow_id: str,
    current_user=Depends(get_current_user)
) -> dict:
    """Generate a preview of the published book"""
    
    result = await publishing_service.preview_publication(workflow_id)
    return result

@router.post("/publishing/{workflow_id}/publish")
async def publish_book(
    workflow_id: str,
    current_user=Depends(get_current_user)
) -> dict:
    """Submit the book for publication"""
    
    result = await publishing_service.submit_for_publication(workflow_id)
    return result

@router.get("/publishing/{workflow_id}/status")
async def get_publishing_status(
    workflow_id: str,
    current_user=Depends(get_current_user)
) -> dict:
    """Get the current status of a publishing workflow"""
    
    workflow = await storage_service.get_publishing_workflow(workflow_id)
    
    return {
        "workflow_id": workflow_id,
        "status": workflow["status"],
        "steps_completed": workflow["steps_completed"],
        "validation_results": workflow.get("validation_results", {}),
        "publication_url": workflow.get("publication_url"),
        "created_at": workflow["created_at"],
        "updated_at": workflow["updated_at"]
    }

@router.post("/marketplace/{marketplace_id}/unpublish")
async def unpublish_book(
    marketplace_id: str,
    current_user=Depends(get_current_user)
) -> dict:
    """Unpublish a book from the marketplace"""
    
    result = await marketplace_service.unpublish_book(marketplace_id, current_user.id)
    return result
```

This manual publishing workflow provides a comprehensive system for authors to publish their AI-generated books to the marketplace with proper validation, preview capabilities, and full control over publication settings.