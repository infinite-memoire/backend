"""
Publishing Workflow Service

Handles the complete publishing workflow for books including metadata management,
validation, preview generation, and marketplace publication.
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.models.publishing import (
    PublicationMetadata, PublicationSettings, PublicationStatus,
    PublicationVisibility, PublishingWorkflow
)
from app.utils.logging_utils import get_logger

logger = get_logger("publishing_service")


class PublishingWorkflowService:
    """Service for managing publishing workflows"""
    
    def __init__(self, storage_service, conversion_service, validation_service):
        self.storage = storage_service
        self.converter = conversion_service
        self.validator = validation_service
    
    async def initiate_publishing_workflow(self, book_id: str, user_id: str, 
                                         auto_generate_metadata: bool = True) -> str:
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
        workflow_id = f"pub_{book_id}_{uuid.uuid4().hex[:8]}"
        
        workflow = PublishingWorkflow(
            workflow_id=workflow_id,
            book_id=book_id,
            user_id=user_id,
            status=PublicationStatus.DRAFT,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        await self.storage.create_publishing_workflow(workflow.__dict__)
        
        # Auto-generate metadata if requested
        if auto_generate_metadata:
            await self._auto_generate_metadata(workflow_id, book)
        
        logger.info(f"Publishing workflow initiated: {workflow_id}")
        return workflow_id
    
    async def get_workflow(self, workflow_id: str) -> Optional[PublishingWorkflow]:
        """Get publishing workflow by ID"""
        workflow_data = await self.storage.get_publishing_workflow(workflow_id)
        if not workflow_data:
            return None
        
        return PublishingWorkflow(**workflow_data)
    
    async def update_publication_metadata(self, 
                                         workflow_id: str, 
                                         metadata: PublicationMetadata) -> Dict[str, Any]:
        """Update the publication metadata for a book"""
        
        workflow_data = await self.storage.get_publishing_workflow(workflow_id)
        if not workflow_data:
            raise ValueError("Publishing workflow not found")
        
        # Validate metadata
        validation_result = await self.validator.validate_publication_metadata(metadata)
        
        if validation_result["is_valid"]:
            workflow_data["publication_metadata"] = metadata.__dict__
            if "metadata_updated" not in workflow_data["steps_completed"]:
                workflow_data["steps_completed"].append("metadata_updated")
            workflow_data["updated_at"] = datetime.now()
            workflow_data["validation_results"]["metadata"] = validation_result
            
            await self.storage.update_publishing_workflow(workflow_data)
            
            logger.info(f"Publication metadata updated: {workflow_id}")
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
                                         settings: PublicationSettings) -> Dict[str, Any]:
        """Update the publication settings for a book"""
        
        workflow_data = await self.storage.get_publishing_workflow(workflow_id)
        if not workflow_data:
            raise ValueError("Publishing workflow not found")
        
        workflow_data["publication_settings"] = settings.__dict__
        if "settings_updated" not in workflow_data["steps_completed"]:
            workflow_data["steps_completed"].append("settings_updated")
        workflow_data["updated_at"] = datetime.now()
        
        await self.storage.update_publishing_workflow(workflow_data)
        
        logger.info(f"Publication settings updated: {workflow_id}")
        return {
            "status": "success",
            "message": "Publication settings updated successfully"
        }
    
    async def generate_preview(self, workflow_id: str, template_name: str = "book.html5") -> Dict[str, str]:
        """Generate a preview of how the book will appear when published"""
        
        workflow_data = await self.storage.get_publishing_workflow(workflow_id)
        if not workflow_data:
            raise ValueError("Publishing workflow not found")
        
        book_id = workflow_data["book_id"]
        
        # Get book content
        book_metadata = await self.storage.get_book_metadata(book_id)
        chapters = await self.storage.get_book_chapters(book_id, book_metadata["current_version"])
        
        # Merge book metadata with publication metadata
        merged_metadata = {**book_metadata, **workflow_data.get("publication_metadata", {})}
        
        # Generate HTML preview
        html_content = await self.converter.convert_book_to_html(
            chapters=chapters,
            book_metadata=merged_metadata,
            template_name=template_name
        )
        
        # Store preview
        preview_id = f"preview_{workflow_id}_{int(datetime.now().timestamp())}"
        preview_path = f"./previews/{preview_id}.html"
        
        Path(preview_path).parent.mkdir(exist_ok=True)
        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Update workflow
        if "preview_generated" not in workflow_data["steps_completed"]:
            workflow_data["steps_completed"].append("preview_generated")
        workflow_data["preview_url"] = f"/preview/{preview_id}"
        workflow_data["updated_at"] = datetime.now()
        await self.storage.update_publishing_workflow(workflow_data)
        
        logger.info(f"Preview generated: {preview_id}")
        return {
            "preview_url": f"/preview/{preview_id}",
            "preview_id": preview_id,
            "status": "success"
        }
    
    async def validate_publication_readiness(self, workflow_id: str) -> Dict[str, Any]:
        """Validate publication readiness"""
        
        workflow_data = await self.storage.get_publishing_workflow(workflow_id)
        if not workflow_data:
            raise ValueError("Publishing workflow not found")
        
        book_id = workflow_data["book_id"]
        book_metadata = await self.storage.get_book_metadata(book_id)
        
        # Validate content
        content_validation = await self.validator.validate_book_content(
            book_id, book_metadata["current_version"]
        )
        
        # Validate metadata if exists
        metadata_validation = {"is_valid": True, "errors": [], "warnings": [], "score": 100}
        if workflow_data.get("publication_metadata"):
            metadata = PublicationMetadata(**workflow_data["publication_metadata"])
            metadata_validation = await self.validator.validate_publication_metadata(metadata)
        
        # Combine results
        combined_validation = {
            "is_valid": content_validation["is_ready"] and metadata_validation["is_valid"],
            "is_ready_for_publication": content_validation["is_ready"] and metadata_validation["is_valid"],
            "score": min(metadata_validation["score"], 100 if content_validation["is_ready"] else 50),
            "errors": content_validation["issues"] + metadata_validation["errors"],
            "warnings": content_validation["recommendations"] + metadata_validation["warnings"],
            "recommendations": content_validation["recommendations"],
            "content_stats": content_validation["stats"]
        }
        
        # Update workflow with validation results
        workflow_data["validation_results"]["combined"] = combined_validation
        workflow_data["updated_at"] = datetime.now()
        await self.storage.update_publishing_workflow(workflow_data)
        
        return combined_validation
    
    async def submit_for_publication(self, workflow_id: str) -> Dict[str, Any]:
        """Submit the book for publication after all checks"""
        
        workflow_data = await self.storage.get_publishing_workflow(workflow_id)
        if not workflow_data:
            raise ValueError("Publishing workflow not found")
        
        # Run final validation
        validation_results = await self.validate_publication_readiness(workflow_id)
        
        if not validation_results["is_ready_for_publication"]:
            return {
                "status": "validation_failed",
                "message": "Book is not ready for publication",
                "validation_results": validation_results
            }
        
        # Update workflow status
        workflow_data["status"] = PublicationStatus.READY_FOR_REVIEW.value
        if "submitted_for_publication" not in workflow_data["steps_completed"]:
            workflow_data["steps_completed"].append("submitted_for_publication")
        workflow_data["validation_results"]["final"] = validation_results
        workflow_data["submitted_at"] = datetime.now()
        workflow_data["updated_at"] = datetime.now()
        
        await self.storage.update_publishing_workflow(workflow_data)
        
        # For MVP: Auto-approve since there's no manual review process
        return await self._auto_approve_publication(workflow_id)
    
    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a publishing workflow"""
        
        workflow_data = await self.storage.get_publishing_workflow(workflow_id)
        if not workflow_data:
            raise ValueError("Publishing workflow not found")
        
        workflow_data["status"] = "cancelled"
        workflow_data["updated_at"] = datetime.now()
        
        await self.storage.update_publishing_workflow(workflow_data)
        logger.info(f"Publishing workflow cancelled: {workflow_id}")
    
    async def get_user_workflows(self, user_id: str, status_filter: Optional[str] = None, 
                               limit: int = 20) -> List[PublishingWorkflow]:
        """Get all workflows for a user"""
        
        workflows_data = await self.storage.get_user_publishing_workflows(
            user_id, status_filter, limit
        )
        
        return [PublishingWorkflow(**data) for data in workflows_data]
    
    async def _auto_generate_metadata(self, workflow_id: str, book_metadata: Dict[str, Any]) -> None:
        """Auto-generate publication metadata from book content"""
        
        # Basic metadata extraction from book
        metadata = PublicationMetadata(
            book_id=book_metadata["book_id"],
            title=book_metadata.get("title", "Untitled"),
            description=book_metadata.get("description", ""),
            author_name=book_metadata.get("author_name", ""),
            language=book_metadata.get("language", "en")
        )
        
        # Update workflow with auto-generated metadata
        workflow_data = await self.storage.get_publishing_workflow(workflow_id)
        workflow_data["publication_metadata"] = metadata.__dict__
        if "auto_metadata_generated" not in workflow_data["steps_completed"]:
            workflow_data["steps_completed"].append("auto_metadata_generated")
        
        await self.storage.update_publishing_workflow(workflow_data)
        logger.info(f"Auto-generated metadata for workflow: {workflow_id}")
    
    async def _auto_approve_publication(self, workflow_id: str) -> Dict[str, Any]:
        """Auto-approve publication for MVP (no manual review)"""
        
        workflow_data = await self.storage.get_publishing_workflow(workflow_id)
        
        try:
            # Generate final HTML version
            final_html = await self._generate_final_publication(workflow_data)
            
            # Create marketplace entry
            from app.services.marketplace import MarketplaceService
            marketplace_service = MarketplaceService(self.storage)
            marketplace_entry = await marketplace_service.create_marketplace_entry(
                workflow_data, final_html
            )
            
            # Update book status
            await self._update_book_publication_status(workflow_data["book_id"], "published")
            
            # Update workflow
            workflow_data["status"] = PublicationStatus.PUBLISHED.value
            workflow_data["published_at"] = datetime.now()
            workflow_data["marketplace_id"] = marketplace_entry["id"]
            workflow_data["publication_url"] = marketplace_entry["url"]
            workflow_data["updated_at"] = datetime.now()
            
            await self.storage.update_publishing_workflow(workflow_data)
            
            logger.info(f"Book published successfully: {workflow_id}")
            return {
                "status": "published",
                "message": "Book has been successfully published",
                "publication_url": marketplace_entry["url"],
                "marketplace_id": marketplace_entry["id"]
            }
            
        except Exception as e:
            # Handle publication failure
            workflow_data["status"] = PublicationStatus.REJECTED.value
            workflow_data["error_message"] = str(e)
            workflow_data["updated_at"] = datetime.now()
            await self.storage.update_publishing_workflow(workflow_data)
            
            logger.error(f"Publication failed: {workflow_id}, error: {str(e)}")
            return {
                "status": "publication_failed",
                "message": f"Publication failed: {str(e)}"
            }
    
    async def _generate_final_publication(self, workflow_data: Dict[str, Any]) -> str:
        """Generate final HTML for publication"""
        
        book_id = workflow_data["book_id"]
        book_metadata = await self.storage.get_book_metadata(book_id)
        chapters = await self.storage.get_book_chapters(book_id, book_metadata["current_version"])
        
        # Merge all metadata
        merged_metadata = {
            **book_metadata,
            **workflow_data.get("publication_metadata", {}),
            **workflow_data.get("publication_settings", {})
        }
        
        # Generate final HTML
        return await self.converter.convert_book_to_html(
            chapters=chapters,
            book_metadata=merged_metadata,
            template_name="marketplace_final.html5"
        )
    
    async def _update_book_publication_status(self, book_id: str, status: str) -> None:
        """Update the publication status of a book"""
        
        book_metadata = await self.storage.get_book_metadata(book_id)
        book_metadata["publication_status"] = status
        book_metadata["published_at"] = datetime.now() if status == "published" else None
        
        await self.storage.update_book_metadata(book_id, book_metadata)


class PublicationValidator:
    """Service for validating publication readiness"""
    
    def __init__(self, storage_service):
        self.storage = storage_service
    
    async def validate_publication_metadata(self, metadata: PublicationMetadata) -> Dict[str, Any]:
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
    
    async def validate_book_content(self, book_id: str, version: str) -> Dict[str, Any]:
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
        validation_results["stats"]["average_chapter_length"] = total_words // len(chapters) if chapters else 0
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