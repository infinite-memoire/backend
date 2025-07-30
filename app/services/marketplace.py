"""
Marketplace Service

Handles marketplace entry management including creation, updates, visibility controls,
and unpublishing functionality for published books.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.models.publishing import PublicationVisibility
from app.utils.logging_utils import get_logger

logger = get_logger("marketplace_service")


class MarketplaceService:
    """Service for managing marketplace entries"""
    
    def __init__(self, storage_service):
        self.storage = storage_service
    
    async def create_marketplace_entry(self, workflow: Dict[str, Any], html_content: str) -> Dict[str, str]:
        """Create a new marketplace entry for a published book"""
        
        marketplace_id = f"book_{workflow['book_id']}_{uuid.uuid4().hex[:8]}"
        
        # Extract metadata and settings
        metadata = workflow.get("publication_metadata", {})
        settings = workflow.get("publication_settings", {})
        
        marketplace_entry = {
            "id": marketplace_id,
            "book_id": workflow["book_id"],
            "workflow_id": workflow["workflow_id"],
            "user_id": workflow["user_id"],
            
            # Publication metadata
            "title": metadata.get("title", "Untitled"),
            "subtitle": metadata.get("subtitle"),
            "description": metadata.get("description", ""),
            "author_name": metadata.get("author_name", "Unknown Author"),
            "author_bio": metadata.get("author_bio", ""),
            "category": metadata.get("category", "memoir"),
            "tags": metadata.get("tags", []),
            "cover_image_url": metadata.get("cover_image_url"),
            "estimated_reading_time": metadata.get("estimated_reading_time", 0),
            "language": metadata.get("language", "en"),
            "content_warnings": metadata.get("content_warnings", []),
            "target_audience": metadata.get("target_audience", "general"),
            
            # Publication settings
            "visibility": settings.get("visibility", "public"),
            "allow_comments": settings.get("allow_comments", True),
            "allow_downloads": settings.get("allow_downloads", True),
            "price": settings.get("price", 0.0),
            "copyright_notice": settings.get("copyright_notice", ""),
            "license_type": settings.get("license_type", "all_rights_reserved"),
            "analytics_enabled": settings.get("analytics_enabled", True),
            
            # Marketplace metadata
            "publication_date": datetime.now(),
            "view_count": 0,
            "download_count": 0,
            "rating_average": 0.0,
            "rating_count": 0,
            "featured": False,
            "trending": False,
            
            # Content and status
            "html_content": html_content,
            "status": "active",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        await self.storage.create_marketplace_entry(marketplace_entry)
        
        logger.info(f"Marketplace entry created: {marketplace_id}")
        return {
            "id": marketplace_id,
            "url": f"/marketplace/book/{marketplace_id}",
            "reading_url": f"/read/{marketplace_id}"
        }
    
    async def get_marketplace_entry(self, marketplace_id: str) -> Optional[Dict[str, Any]]:
        """Get marketplace entry by ID"""
        return await self.storage.get_marketplace_entry(marketplace_id)
    
    async def update_marketplace_entry(self, marketplace_entry: Dict[str, Any]) -> None:
        """Update marketplace entry"""
        marketplace_entry["updated_at"] = datetime.now()
        await self.storage.update_marketplace_entry(marketplace_entry)
    
    async def update_marketplace_visibility(self, marketplace_id: str, 
                                          visibility: PublicationVisibility, 
                                          user_id: str) -> Dict[str, str]:
        """Update the visibility of a marketplace entry"""
        
        entry = await self.storage.get_marketplace_entry(marketplace_id)
        if not entry:
            raise ValueError("Marketplace entry not found")
        
        # Verify ownership
        if entry["user_id"] != user_id:
            raise PermissionError("User does not own this marketplace entry")
        
        entry["visibility"] = visibility.value
        entry["updated_at"] = datetime.now()
        
        await self.storage.update_marketplace_entry(entry)
        
        logger.info(f"Marketplace visibility updated: {marketplace_id} -> {visibility.value}")
        return {
            "status": "success",
            "message": f"Visibility updated to {visibility.value}"
        }
    
    async def unpublish_book(self, marketplace_id: str, user_id: str) -> Dict[str, str]:
        """Unpublish a book from the marketplace"""
        
        entry = await self.storage.get_marketplace_entry(marketplace_id)
        if not entry:
            raise ValueError("Marketplace entry not found")
        
        # Verify ownership
        if entry["user_id"] != user_id:
            raise PermissionError("User does not own this marketplace entry")
        
        # Update status
        entry["status"] = "unpublished"
        entry["unpublished_at"] = datetime.now()
        entry["updated_at"] = datetime.now()
        
        await self.storage.update_marketplace_entry(entry)
        
        # Also update the book's publication status
        book_metadata = await self.storage.get_book_metadata(entry["book_id"])
        book_metadata["publication_status"] = "unpublished"
        book_metadata["updated_at"] = datetime.now()
        await self.storage.update_book_metadata(book_metadata)
        
        logger.info(f"Book unpublished from marketplace: {marketplace_id}")
        return {
            "status": "success",
            "message": "Book has been unpublished from marketplace"
        }
    
    async def republish_book(self, marketplace_id: str, user_id: str) -> Dict[str, str]:
        """Republish a previously unpublished book"""
        
        entry = await self.storage.get_marketplace_entry(marketplace_id)
        if not entry:
            raise ValueError("Marketplace entry not found")
        
        # Verify ownership
        if entry["user_id"] != user_id:
            raise PermissionError("User does not own this marketplace entry")
        
        if entry["status"] != "unpublished":
            raise ValueError("Book is not in unpublished state")
        
        # Update status
        entry["status"] = "active"
        entry["republished_at"] = datetime.now()
        entry["updated_at"] = datetime.now()
        
        await self.storage.update_marketplace_entry(entry)
        
        # Also update the book's publication status
        book_metadata = await self.storage.get_book_metadata(entry["book_id"])
        book_metadata["publication_status"] = "published"
        book_metadata["updated_at"] = datetime.now()
        await self.storage.update_book_metadata(book_metadata)
        
        logger.info(f"Book republished to marketplace: {marketplace_id}")
        return {
            "status": "success",
            "message": "Book has been republished to marketplace"
        }
    
    async def update_marketplace_metadata(self, marketplace_id: str, user_id: str,
                                        metadata_updates: Dict[str, Any]) -> Dict[str, str]:
        """Update marketplace metadata (title, description, etc.)"""
        
        entry = await self.storage.get_marketplace_entry(marketplace_id)
        if not entry:
            raise ValueError("Marketplace entry not found")
        
        # Verify ownership
        if entry["user_id"] != user_id:
            raise PermissionError("User does not own this marketplace entry")
        
        # Allowed fields for update
        allowed_fields = {
            "title", "subtitle", "description", "author_name", "author_bio",
            "category", "tags", "cover_image_url", "content_warnings",
            "target_audience", "copyright_notice"
        }
        
        # Update allowed fields
        for field, value in metadata_updates.items():
            if field in allowed_fields:
                entry[field] = value
        
        entry["updated_at"] = datetime.now()
        await self.storage.update_marketplace_entry(entry)
        
        logger.info(f"Marketplace metadata updated: {marketplace_id}")
        return {
            "status": "success",
            "message": "Marketplace metadata updated successfully"
        }
    
    async def increment_view_count(self, marketplace_id: str) -> None:
        """Increment view count for a marketplace entry"""
        
        entry = await self.storage.get_marketplace_entry(marketplace_id)
        if entry and entry["status"] == "active":
            entry["view_count"] = entry.get("view_count", 0) + 1
            entry["last_viewed_at"] = datetime.now()
            await self.storage.update_marketplace_entry(entry)
    
    async def increment_download_count(self, marketplace_id: str) -> None:
        """Increment download count for a marketplace entry"""
        
        entry = await self.storage.get_marketplace_entry(marketplace_id)
        if entry and entry["status"] == "active" and entry.get("allow_downloads", True):
            entry["download_count"] = entry.get("download_count", 0) + 1
            entry["last_downloaded_at"] = datetime.now()
            await self.storage.update_marketplace_entry(entry)
    
    async def search_marketplace(self, query: str = "", category: Optional[str] = None,
                               tags: Optional[List[str]] = None, language: Optional[str] = None,
                               limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """Search marketplace entries"""
        
        search_params = {
            "query": query,
            "category": category,
            "tags": tags,
            "language": language,
            "status": "active",
            "visibility": ["public", "featured"],
            "limit": limit,
            "offset": offset
        }
        
        results = await self.storage.search_marketplace_entries(search_params)
        
        return {
            "results": results["entries"],
            "total_count": results["total_count"],
            "page": offset // limit + 1,
            "total_pages": (results["total_count"] + limit - 1) // limit
        }
    
    async def get_featured_books(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get featured books for homepage"""
        
        return await self.storage.get_featured_marketplace_entries(limit)
    
    async def get_trending_books(self, limit: int = 10, days: int = 7) -> List[Dict[str, Any]]:
        """Get trending books based on recent activity"""
        
        return await self.storage.get_trending_marketplace_entries(limit, days)
    
    async def get_user_published_books(self, user_id: str, 
                                     include_unpublished: bool = False) -> List[Dict[str, Any]]:
        """Get all published books for a user"""
        
        status_filter = ["active"] if not include_unpublished else ["active", "unpublished"]
        return await self.storage.get_user_marketplace_entries(user_id, status_filter)
    
    async def update_book_rating(self, marketplace_id: str, rating: float, 
                               user_id: str) -> Dict[str, str]:
        """Update book rating (future feature)"""
        
        # This would be implemented when rating system is added
        # For now, just return success
        return {
            "status": "success",
            "message": "Rating system not implemented in MVP"
        }
    
    async def add_comment(self, marketplace_id: str, user_id: str, 
                         comment: str) -> Dict[str, str]:
        """Add comment to a book (future feature)"""
        
        # This would be implemented when comment system is added
        # For now, just return success
        return {
            "status": "success",
            "message": "Comment system not implemented in MVP"
        }
    
    async def get_marketplace_analytics(self, marketplace_id: str, 
                                      user_id: str) -> Dict[str, Any]:
        """Get analytics for a marketplace entry"""
        
        entry = await self.storage.get_marketplace_entry(marketplace_id)
        if not entry:
            raise ValueError("Marketplace entry not found")
        
        # Verify ownership
        if entry["user_id"] != user_id:
            raise PermissionError("User does not own this marketplace entry")
        
        if not entry.get("analytics_enabled", True):
            raise PermissionError("Analytics are disabled for this book")
        
        return {
            "marketplace_id": marketplace_id,
            "view_count": entry.get("view_count", 0),
            "download_count": entry.get("download_count", 0),
            "rating_average": entry.get("rating_average", 0.0),
            "rating_count": entry.get("rating_count", 0),
            "publication_date": entry.get("publication_date"),
            "last_viewed_at": entry.get("last_viewed_at"),
            "last_downloaded_at": entry.get("last_downloaded_at")
        }