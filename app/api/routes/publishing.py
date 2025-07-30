"""
Publishing API Routes

Handles all publishing-related endpoints including workflow management,
metadata updates, preview generation, and marketplace publishing.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import asyncio

from app.services.publishing import PublishingWorkflowService, PublicationValidator
from app.services.content_storage import ContentStorageService, validate_book_access
from app.services.html_conversion import HTMLConversionService
from app.services.marketplace import MarketplaceService
from app.models.publishing import (
    PublicationMetadata, PublicationSettings, PublicationStatus,
    PublicationVisibility, PublishingWorkflow
)
from app.utils.auth_utils import get_current_user, User
from app.utils.logging_utils import get_logger, log_performance
from app.utils.exceptions import ValidationError, PublishingError
from app.dependencies import get_services

logger = get_logger("publishing_api")
router = APIRouter(tags=["Publishing"], prefix="/api/publishing")


# Request/Response Models
class StartPublishingRequest(BaseModel):
    """Request to start publishing workflow"""
    book_id: str = Field(..., description="ID of the book to publish")
    auto_generate_metadata: bool = Field(True, description="Auto-generate missing metadata")


class PublicationMetadataRequest(BaseModel):
    """Request to update publication metadata"""
    title: str = Field(..., min_length=3, max_length=200, description="Book title")
    subtitle: Optional[str] = Field(None, max_length=300, description="Book subtitle")
    description: str = Field(..., min_length=50, max_length=2000, description="Book description")
    author_name: str = Field(..., min_length=2, max_length=100, description="Author name")
    author_bio: str = Field("", max_length=1000, description="Author biography")
    category: str = Field("memoir", description="Book category")
    tags: List[str] = Field(default_factory=list, description="Book tags")
    cover_image_url: Optional[str] = Field(None, description="Cover image URL")
    language: str = Field("en", description="Book language")
    content_warnings: List[str] = Field(default_factory=list, description="Content warnings")
    target_audience: str = Field("general", description="Target audience")

    @validator('tags')
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 tags allowed')
        return [tag.strip().lower() for tag in v if tag.strip()]

    @validator('category')
    def validate_category(cls, v):
        allowed_categories = [
            'memoir', 'biography', 'autobiography', 'family_history',
            'business', 'personal_development', 'travel', 'history', 'other'
        ]
        if v not in allowed_categories:
            raise ValueError(f'Category must be one of: {", ".join(allowed_categories)}')
        return v


class PublicationSettingsRequest(BaseModel):
    """Request to update publication settings"""
    visibility: str = Field("public", description="Publication visibility")
    allow_comments: bool = Field(True, description="Allow reader comments")
    allow_downloads: bool = Field(True, description="Allow book downloads")
    price: float = Field(0.0, ge=0.0, le=1000.0, description="Book price (MVP: free only)")
    copyright_notice: str = Field("", max_length=500, description="Copyright notice")
    license_type: str = Field("all_rights_reserved", description="License type")
    marketplace_listing: bool = Field(True, description="List in marketplace")
    analytics_enabled: bool = Field(True, description="Enable analytics")

    @validator('visibility')
    def validate_visibility(cls, v):
        allowed = ['private', 'unlisted', 'public', 'featured']
        if v not in allowed:
            raise ValueError(f'Visibility must be one of: {", ".join(allowed)}')
        return v

    @validator('price')
    def validate_price(cls, v):
        # MVP: Only free books allowed
        if v != 0.0:
            raise ValueError('Only free books are supported in MVP')
        return v


class PublishingStatusResponse(BaseModel):
    """Response for publishing status"""
    workflow_id: str
    book_id: str
    status: str
    steps_completed: List[str]
    validation_results: Dict[str, Any]
    publication_url: Optional[str] = None
    marketplace_id: Optional[str] = None
    preview_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class ValidationResultResponse(BaseModel):
    """Response for validation results"""
    is_valid: bool
    is_ready_for_publication: bool
    score: int
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    content_stats: Dict[str, Any]


# API Routes
@router.post("/start/{book_id}", response_model=Dict[str, str])
@log_performance(logger)
async def start_publishing_workflow(
    book_id: str,
    request: StartPublishingRequest,
    current_user: User = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
) -> Dict[str, str]:
    """
    Start the publishing workflow for a book

    This endpoint initiates the complete publishing pipeline:
    - Validates book ownership and completeness
    - Creates publishing workflow instance
    - Optionally auto-generates metadata
    - Returns workflow ID for tracking progress
    """
    try:
        publishing_service = services["publishing"]
        content_service = services["content_storage"]

        # Validate book ownership
        if not await validate_book_access(content_service, book_id, current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't own this book"
            )

        logger.info("Starting publishing workflow",
                   book_id=book_id, user_id=current_user.id,
                   auto_generate=request.auto_generate_metadata)

        # Start workflow
        workflow_id = await publishing_service.initiate_publishing_workflow(
            book_id=book_id,
            user_id=current_user.id,
            auto_generate_metadata=request.auto_generate_metadata
        )

        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Publishing workflow initiated successfully"
        }

    except ValueError as e:
        logger.warning("Publishing workflow validation failed",
                      book_id=book_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to start publishing workflow",
                    book_id=book_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to start publishing workflow")


@router.put("/{workflow_id}/metadata", response_model=Dict[str, Any])
@log_performance(logger)
async def update_publication_metadata(
    workflow_id: str,
    metadata_request: PublicationMetadataRequest,
    current_user: User = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
) -> Dict[str, Any]:
    """
    Update publication metadata for a workflow

    Updates all metadata required for publication including:
    - Title, subtitle, description
    - Author information
    - Categorization and tags
    - Content warnings and audience info
    """
    try:
        publishing_service = services["publishing"]

        # Validate workflow ownership
        workflow = await publishing_service.get_workflow(workflow_id)
        if not workflow or workflow.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't own this workflow"
            )

        logger.info("Updating publication metadata",
                   workflow_id=workflow_id, user_id=current_user.id)

        # Convert request to metadata object
        metadata = PublicationMetadata(**metadata_request.dict())

        # Update metadata
        result = await publishing_service.update_publication_metadata(workflow_id, metadata)

        return result

    except ValidationError as e:
        logger.warning("Metadata validation failed",
                      workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to update publication metadata",
                    workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to update metadata")


@router.put("/{workflow_id}/settings", response_model=Dict[str, Any])
@log_performance(logger)
async def update_publication_settings(
    workflow_id: str,
    settings_request: PublicationSettingsRequest,
    current_user: User = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
) -> Dict[str, Any]:
    """
    Update publication settings for a workflow

    Configures publication options including:
    - Visibility and access controls
    - Pricing and licensing
    - Comments and downloads
    - Analytics preferences
    """
    try:
        publishing_service = services["publishing"]

        # Validate workflow ownership
        workflow = await publishing_service.get_workflow(workflow_id)
        if not workflow or workflow.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't own this workflow"
            )

        logger.info("Updating publication settings",
                   workflow_id=workflow_id, user_id=current_user.id)

        # Convert request to settings object
        settings = PublicationSettings(**settings_request.dict())

        # Update settings
        result = await publishing_service.update_publication_settings(workflow_id, settings)

        return result

    except Exception as e:
        logger.error("Failed to update publication settings",
                    workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to update settings")


@router.post("/{workflow_id}/validate", response_model=ValidationResultResponse)
@log_performance(logger)
async def validate_publication(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
) -> ValidationResultResponse:
    """
    Validate publication readiness

    Performs comprehensive validation including:
    - Content completeness and quality
    - Metadata validation
    - Publishing requirements check
    - Quality score assessment
    """
    try:
        publishing_service = services["publishing"]
        validator = services["validator"]

        # Validate workflow ownership
        workflow = await publishing_service.get_workflow(workflow_id)
        if not workflow or workflow.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't own this workflow"
            )

        logger.info("Validating publication",
                   workflow_id=workflow_id, user_id=current_user.id)

        # Run validation
        validation_results = await publishing_service.validate_publication_readiness(workflow_id)

        return ValidationResultResponse(**validation_results)

    except Exception as e:
        logger.error("Failed to validate publication",
                    workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to validate publication")


@router.post("/{workflow_id}/preview", response_model=Dict[str, str])
@log_performance(logger)
async def generate_publication_preview(
    workflow_id: str,
    template_name: str = "book.html5",
    current_user: User = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
) -> Dict[str, str]:
    """
    Generate a preview of the published book

    Creates an HTML preview showing how the book will appear
    when published, using the specified template.
    """
    try:
        publishing_service = services["publishing"]

        # Validate workflow ownership
        workflow = await publishing_service.get_workflow(workflow_id)
        if not workflow or workflow.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't own this workflow"
            )

        logger.info("Generating publication preview",
                   workflow_id=workflow_id, template=template_name, user_id=current_user.id)

        # Generate preview
        result = await publishing_service.generate_preview(workflow_id, template_name)

        return result

    except Exception as e:
        logger.error("Failed to generate preview",
                    workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to generate preview")


@router.post("/{workflow_id}/publish", response_model=Dict[str, Any])
@log_performance(logger)
async def publish_book(
    workflow_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
) -> Dict[str, Any]:
    """
    Submit the book for publication

    Performs final validation and publishes the book to the marketplace.
    This is the final step in the publishing workflow.
    """
    try:
        publishing_service = services["publishing"]

        # Validate workflow ownership
        workflow = await publishing_service.get_workflow(workflow_id)
        if not workflow or workflow.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't own this workflow"
            )

        logger.info("Publishing book",
                   workflow_id=workflow_id, user_id=current_user.id)

        # Submit for publication (includes validation)
        result = await publishing_service.submit_for_publication(workflow_id)

        if result["status"] == "validation_failed":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=result
            )

        return result

    except PublishingError as e:
        logger.warning("Publishing validation failed",
                      workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to publish book",
                    workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to publish book")


@router.get("/{workflow_id}/status", response_model=PublishingStatusResponse)
@log_performance(logger)
async def get_publishing_status(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
) -> PublishingStatusResponse:
    """
    Get the current status of a publishing workflow

    Returns detailed information about workflow progress,
    validation results, and any errors encountered.
    """
    try:
        publishing_service = services["publishing"]

        # Get workflow
        workflow = await publishing_service.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Publishing workflow not found"
            )

        # Validate ownership
        if workflow.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't own this workflow"
            )

        logger.debug("Retrieved publishing status",
                    workflow_id=workflow_id, status=workflow.status)

        return PublishingStatusResponse(
            workflow_id=workflow.workflow_id,
            book_id=workflow.book_id,
            status=workflow.status.value,
            steps_completed=workflow.steps_completed,
            validation_results=workflow.validation_results,
            publication_url=workflow.publication_url,
            marketplace_id=workflow.marketplace_id,
            preview_url=workflow.preview_url,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            error_message=workflow.error_message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get publishing status",
                    workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to get status")


@router.delete("/{workflow_id}", response_model=Dict[str, str])
@log_performance(logger)
async def cancel_publishing_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
) -> Dict[str, str]:
    """
    Cancel a publishing workflow

    Cancels an active publishing workflow and cleans up
    any temporary resources created during the process.
    """
    try:
        publishing_service = services["publishing"]

        # Validate workflow ownership
        workflow = await publishing_service.get_workflow(workflow_id)
        if not workflow or workflow.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't own this workflow"
            )

        # Check if workflow can be cancelled
        if workflow.status in [PublicationStatus.PUBLISHED, PublicationStatus.REJECTED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel workflow in {workflow.status.value} status"
            )

        logger.info("Cancelling publishing workflow",
                   workflow_id=workflow_id, user_id=current_user.id)

        # Cancel workflow
        await publishing_service.cancel_workflow(workflow_id)

        return {
            "status": "cancelled",
            "message": "Publishing workflow has been cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel publishing workflow",
                    workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to cancel workflow")


@router.get("/workflows", response_model=List[PublishingStatusResponse])
@log_performance(logger)
async def list_user_workflows(
    current_user: User = Depends(get_current_user),
    status_filter: Optional[str] = None,
    limit: int = 20,
    services: Dict[str, Any] = Depends(get_services)
) -> List[PublishingStatusResponse]:
    """
    List all publishing workflows for the current user

    Returns a list of workflows owned by the user, optionally
    filtered by status and limited to the specified count.
    """
    try:
        publishing_service = services["publishing"]

        logger.info("Listing user workflows",
                   user_id=current_user.id, status_filter=status_filter, limit=limit)

        # Get workflows
        workflows = await publishing_service.get_user_workflows(
            user_id=current_user.id,
            status_filter=status_filter,
            limit=limit
        )

        # Convert to response format
        response_list = []
        for workflow in workflows:
            response_list.append(PublishingStatusResponse(
                workflow_id=workflow.workflow_id,
                book_id=workflow.book_id,
                status=workflow.status.value,
                steps_completed=workflow.steps_completed,
                validation_results=workflow.validation_results,
                publication_url=workflow.publication_url,
                marketplace_id=workflow.marketplace_id,
                preview_url=workflow.preview_url,
                created_at=workflow.created_at,
                updated_at=workflow.updated_at,
                error_message=workflow.error_message
            ))

        return response_list

    except Exception as e:
        logger.error("Failed to list user workflows",
                    user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to list workflows")


# Error handlers
@router.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    """Handle validation errors with detailed messages"""
    logger.warning("Validation error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation failed",
            "details": str(exc),
            "type": "validation_error"
        }
    )


@router.exception_handler(PublishingError)
async def publishing_error_handler(request, exc: PublishingError):
    """Handle publishing-specific errors"""
    logger.warning("Publishing error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Publishing failed",
            "details": str(exc),
            "type": "publishing_error"
        }
    )