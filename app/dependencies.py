"""
Dependency Injection

FastAPI dependency injection for services and utilities.
"""

from typing import Dict, Any
from functools import lru_cache

from app.services.content_storage import ContentStorageService
from app.services.publishing import PublishingWorkflowService, PublicationValidator
from app.services.html_conversion import HTMLConversionService
from app.services.marketplace import MarketplaceService
from app.utils.logging import get_logger

logger = get_logger("dependencies")


@lru_cache()
def get_content_storage_service() -> ContentStorageService:
    """Get content storage service instance"""
    # In a real implementation, this would initialize with proper database connections
    # For now, return a mock service
    return ContentStorageService()


@lru_cache()
def get_html_conversion_service() -> HTMLConversionService:
    """Get HTML conversion service instance"""
    return HTMLConversionService()


@lru_cache()
def get_publication_validator() -> PublicationValidator:
    """Get publication validator instance"""
    storage_service = get_content_storage_service()
    return PublicationValidator(storage_service)


@lru_cache()
def get_publishing_workflow_service() -> PublishingWorkflowService:
    """Get publishing workflow service instance"""
    storage_service = get_content_storage_service()
    conversion_service = get_html_conversion_service()
    validator = get_publication_validator()
    
    return PublishingWorkflowService(
        storage_service=storage_service,
        conversion_service=conversion_service,
        validation_service=validator
    )


@lru_cache()
def get_marketplace_service() -> MarketplaceService:
    """Get marketplace service instance"""
    storage_service = get_content_storage_service()
    return MarketplaceService(storage_service)


def get_services() -> Dict[str, Any]:
    """Get all services as a dependency"""
    return {
        "content_storage": get_content_storage_service(),
        "publishing": get_publishing_workflow_service(),
        "html_conversion": get_html_conversion_service(),
        "validator": get_publication_validator(),
        "marketplace": get_marketplace_service()
    }


# Service dependencies for individual injection
def get_publishing_service() -> PublishingWorkflowService:
    """Individual publishing service dependency"""
    return get_publishing_workflow_service()


def get_validator_service() -> PublicationValidator:
    """Individual validator service dependency"""
    return get_publication_validator()


def get_storage_service() -> ContentStorageService:
    """Individual storage service dependency"""
    return get_content_storage_service()


def get_conversion_service() -> HTMLConversionService:
    """Individual conversion service dependency"""
    return get_html_conversion_service()


def get_marketplace_service_dep() -> MarketplaceService:
    """Individual marketplace service dependency"""
    return get_marketplace_service()


# Health check dependencies
async def check_services_health() -> Dict[str, str]:
    """Check health of all services"""
    health_status = {}
    
    try:
        # Check content storage
        storage_service = get_content_storage_service()
        # In a real implementation, ping the database
        health_status["content_storage"] = "healthy"
    except Exception as e:
        health_status["content_storage"] = f"unhealthy: {str(e)}"
    
    try:
        # Check HTML conversion
        conversion_service = get_html_conversion_service()
        health_status["html_conversion"] = "healthy"
    except Exception as e:
        health_status["html_conversion"] = f"unhealthy: {str(e)}"
    
    try:
        # Check publishing service
        publishing_service = get_publishing_workflow_service()
        health_status["publishing"] = "healthy"
    except Exception as e:
        health_status["publishing"] = f"unhealthy: {str(e)}"
    
    try:
        # Check marketplace service
        marketplace_service = get_marketplace_service()
        health_status["marketplace"] = "healthy"
    except Exception as e:
        health_status["marketplace"] = f"unhealthy: {str(e)}"
    
    return health_status