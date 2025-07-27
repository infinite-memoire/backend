"""
Content Storage Service for Output Management

Handles storage and retrieval of book content, chapters, and metadata
in Firestore with optimized query patterns and data consistency.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from google.cloud import firestore
from app.models.content import BookModel, ChapterModel, ContentMetadata
from app.utils.logging import get_logger
import asyncio
import json

logger = get_logger(__name__)


class ContentStorageService:
    """Service for managing book and chapter content storage in Firestore"""
    
    def __init__(self, firestore_client: firestore.Client):
        self.db = firestore_client
        
    async def create_book(self, user_id: str, book_data: BookModel) -> str:
        """Create a new book with initial metadata"""
        try:
            book_id = book_data.id or f"book_{user_id}_{int(datetime.now().timestamp())}"
            
            book_document = {
                "id": book_id,
                "owner_user_id": user_id,
                "title": book_data.title,
                "description": book_data.description,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "status": "draft",
                "current_version": "v1.0",
                "versions": {
                    "v1.0": {
                        "created_at": datetime.now(),
                        "chapter_count": 0,
                        "total_word_count": 0,
                        "status": "active"
                    }
                },
                "metadata": book_data.metadata or {},
                "settings": book_data.settings or {}
            }
            
            doc_ref = self.db.collection('books').document(book_id)
            doc_ref.set(book_document)
            
            logger.info("Created book", book_id=book_id, user_id=user_id)
            return book_id
            
        except Exception as e:
            logger.error("Failed to create book", error=str(e), user_id=user_id)
            raise
    
    async def get_book_metadata(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve book metadata by ID"""
        try:
            doc_ref = self.db.collection('books').document(book_id)
            doc = doc_ref.get()
            
            if doc.exists:
                logger.debug("Retrieved book metadata", book_id=book_id)
                return doc.to_dict()
            else:
                logger.warning("Book not found", book_id=book_id)
                return None
                
        except Exception as e:
            logger.error("Failed to get book metadata", book_id=book_id, error=str(e))
            raise
    
    async def update_book_metadata(self, book_id: str, updates: Dict[str, Any]) -> None:
        """Update book metadata with version tracking"""
        try:
            doc_ref = self.db.collection('books').document(book_id)
            
            update_data = {
                **updates,
                "updated_at": datetime.now()
            }
            
            doc_ref.update(update_data)
            logger.info("Updated book metadata", book_id=book_id, fields=list(updates.keys()))
            
        except Exception as e:
            logger.error("Failed to update book metadata", book_id=book_id, error=str(e))
            raise
    
    async def store_chapter(self, chapter_data: ChapterModel) -> str:
        """Store chapter content with metadata and source references"""
        try:
            chapter_id = chapter_data.id or f"chapter_{chapter_data.book_id}_{chapter_data.chapter_number}"
            
            chapter_document = {
                "id": chapter_id,
                "book_id": chapter_data.book_id,
                "book_version": chapter_data.book_version,
                "chapter_number": chapter_data.chapter_number,
                "title": chapter_data.title,
                "content": {
                    "markdown_text": chapter_data.content.markdown_text,
                    "word_count": len(chapter_data.content.markdown_text.split()),
                    "themes": chapter_data.content.themes,
                    "participants": chapter_data.content.participants,
                    "tags": chapter_data.content.tags
                },
                "metadata": {
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "status": chapter_data.status,
                    "quality_score": chapter_data.quality_score,
                    "generation_agent": chapter_data.generation_agent
                },
                "source_references": {
                    "audio_session_ids": chapter_data.source_references.audio_session_ids,
                    "transcript_chunk_ids": chapter_data.source_references.transcript_chunk_ids,
                    "storyline_node_ids": chapter_data.source_references.storyline_node_ids,
                    "source_confidence": chapter_data.source_references.source_confidence
                },
                "generation_metadata": chapter_data.generation_metadata or {}
            }
            
            doc_ref = self.db.collection('chapters').document(chapter_id)
            doc_ref.set(chapter_document)
            
            # Update book chapter count
            await self._update_book_chapter_count(chapter_data.book_id, chapter_data.book_version)
            
            logger.info("Stored chapter", chapter_id=chapter_id, book_id=chapter_data.book_id)
            return chapter_id
            
        except Exception as e:
            logger.error("Failed to store chapter", error=str(e), book_id=chapter_data.book_id)
            raise
    
    async def get_chapter(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chapter by ID"""
        try:
            doc_ref = self.db.collection('chapters').document(chapter_id)
            doc = doc_ref.get()
            
            if doc.exists:
                logger.debug("Retrieved chapter", chapter_id=chapter_id)
                return doc.to_dict()
            else:
                logger.warning("Chapter not found", chapter_id=chapter_id)
                return None
                
        except Exception as e:
            logger.error("Failed to get chapter", chapter_id=chapter_id, error=str(e))
            raise
    
    async def get_book_chapters(self, book_id: str, version: str = "latest") -> List[Dict[str, Any]]:
        """Retrieve all chapters for a book version, ordered by chapter number"""
        try:
            if version == "latest":
                book_meta = await self.get_book_metadata(book_id)
                if not book_meta:
                    return []
                version = book_meta.get("current_version", "v1.0")
            
            query = (self.db.collection('chapters')
                    .where('book_id', '==', book_id)
                    .where('book_version', '==', version)
                    .order_by('chapter_number'))
            
            docs = query.stream()
            chapters = [doc.to_dict() for doc in docs]
            
            logger.debug("Retrieved book chapters", 
                        book_id=book_id, version=version, count=len(chapters))
            return chapters
            
        except Exception as e:
            logger.error("Failed to get book chapters", 
                        book_id=book_id, version=version, error=str(e))
            raise
    
    async def update_chapter_content(self, chapter_id: str, content: str, 
                                   metadata_updates: Optional[Dict[str, Any]] = None) -> None:
        """Update chapter content and metadata"""
        try:
            doc_ref = self.db.collection('chapters').document(chapter_id)
            
            update_data = {
                "content.markdown_text": content,
                "content.word_count": len(content.split()),
                "metadata.updated_at": datetime.now()
            }
            
            if metadata_updates:
                for key, value in metadata_updates.items():
                    update_data[f"metadata.{key}"] = value
            
            doc_ref.update(update_data)
            logger.info("Updated chapter content", chapter_id=chapter_id)
            
        except Exception as e:
            logger.error("Failed to update chapter content", 
                        chapter_id=chapter_id, error=str(e))
            raise
    
    async def delete_chapter(self, chapter_id: str) -> None:
        """Delete a chapter and update book metadata"""
        try:
            # Get chapter info before deletion
            chapter = await self.get_chapter(chapter_id)
            if not chapter:
                logger.warning("Chapter not found for deletion", chapter_id=chapter_id)
                return
            
            # Delete the chapter
            doc_ref = self.db.collection('chapters').document(chapter_id)
            doc_ref.delete()
            
            # Update book chapter count
            await self._update_book_chapter_count(chapter["book_id"], chapter["book_version"])
            
            logger.info("Deleted chapter", chapter_id=chapter_id, book_id=chapter["book_id"])
            
        except Exception as e:
            logger.error("Failed to delete chapter", chapter_id=chapter_id, error=str(e))
            raise
    
    async def create_book_version(self, book_id: str, version: str, 
                                base_version: Optional[str] = None) -> str:
        """Create a new version of a book"""
        try:
            book_ref = self.db.collection('books').document(book_id)
            book_doc = book_ref.get()
            
            if not book_doc.exists:
                raise ValueError(f"Book {book_id} not found")
            
            book_data = book_doc.to_dict()
            
            # Create new version metadata
            version_metadata = {
                "created_at": datetime.now(),
                "chapter_count": 0,
                "total_word_count": 0,
                "status": "active",
                "base_version": base_version
            }
            
            # Update book versions
            book_data["versions"][version] = version_metadata
            book_data["current_version"] = version
            book_data["updated_at"] = datetime.now()
            
            book_ref.set(book_data)
            
            # If base version specified, copy chapters
            if base_version:
                await self._copy_chapters_to_version(book_id, base_version, version)
            
            logger.info("Created book version", book_id=book_id, version=version, base_version=base_version)
            return version
            
        except Exception as e:
            logger.error("Failed to create book version", 
                        book_id=book_id, version=version, error=str(e))
            raise
    
    async def get_user_books(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all books owned by a user"""
        try:
            query = (self.db.collection('books')
                    .where('owner_user_id', '==', user_id)
                    .order_by('updated_at', direction=firestore.Query.DESCENDING)
                    .limit(limit))
            
            docs = query.stream()
            books = [doc.to_dict() for doc in docs]
            
            logger.debug("Retrieved user books", user_id=user_id, count=len(books))
            return books
            
        except Exception as e:
            logger.error("Failed to get user books", user_id=user_id, error=str(e))
            raise
    
    async def search_books(self, query_text: str, filters: Optional[Dict[str, Any]] = None,
                          limit: int = 20) -> List[Dict[str, Any]]:
        """Search books by title, description, or content"""
        try:
            # For MVP, implement basic title search
            # In production, would use full-text search service
            
            query = self.db.collection('books')
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if field in ['status', 'owner_user_id']:
                        query = query.where(field, '==', value)
            
            # Get all matching documents (in production, use search service)
            docs = query.stream()
            books = []
            
            query_lower = query_text.lower()
            
            for doc in docs:
                book_data = doc.to_dict()
                title = book_data.get('title', '').lower()
                description = book_data.get('description', '').lower()
                
                if query_lower in title or query_lower in description:
                    books.append(book_data)
                    
                if len(books) >= limit:
                    break
            
            logger.debug("Searched books", query=query_text, count=len(books))
            return books
            
        except Exception as e:
            logger.error("Failed to search books", query=query_text, error=str(e))
            raise
    
    async def _update_book_chapter_count(self, book_id: str, version: str) -> None:
        """Update the chapter count for a book version"""
        try:
            chapters = await self.get_book_chapters(book_id, version)
            chapter_count = len(chapters)
            total_word_count = sum(
                chapter.get("content", {}).get("word_count", 0) 
                for chapter in chapters
            )
            
            book_ref = self.db.collection('books').document(book_id)
            book_ref.update({
                f"versions.{version}.chapter_count": chapter_count,
                f"versions.{version}.total_word_count": total_word_count,
                "updated_at": datetime.now()
            })
            
        except Exception as e:
            logger.error("Failed to update book chapter count", 
                        book_id=book_id, version=version, error=str(e))
            # Don't raise - this is a secondary operation
    
    async def _copy_chapters_to_version(self, book_id: str, source_version: str, 
                                      target_version: str) -> None:
        """Copy chapters from one version to another"""
        try:
            source_chapters = await self.get_book_chapters(book_id, source_version)
            
            for chapter in source_chapters:
                # Create new chapter with updated version
                new_chapter_id = f"chapter_{book_id}_{target_version}_{chapter['chapter_number']}"
                
                chapter_data = {
                    **chapter,
                    "id": new_chapter_id,
                    "book_version": target_version,
                    "metadata": {
                        **chapter["metadata"],
                        "created_at": datetime.now(),
                        "copied_from": f"{source_version}:{chapter['id']}"
                    }
                }
                
                doc_ref = self.db.collection('chapters').document(new_chapter_id)
                doc_ref.set(chapter_data)
            
            logger.info("Copied chapters between versions", 
                       book_id=book_id, source=source_version, target=target_version, 
                       count=len(source_chapters))
                       
        except Exception as e:
            logger.error("Failed to copy chapters between versions", 
                        book_id=book_id, source=source_version, target=target_version, 
                        error=str(e))
            raise


# Utility functions for content operations
async def validate_book_access(storage_service: ContentStorageService, 
                              book_id: str, user_id: str) -> bool:
    """Validate that a user has access to a book"""
    book = await storage_service.get_book_metadata(book_id)
    return book and book.get("owner_user_id") == user_id


async def get_book_statistics(storage_service: ContentStorageService, 
                             book_id: str) -> Dict[str, Any]:
    """Get comprehensive statistics for a book"""
    book = await storage_service.get_book_metadata(book_id)
    if not book:
        return {}
    
    chapters = await storage_service.get_book_chapters(book_id, book["current_version"])
    
    total_words = sum(chapter.get("content", {}).get("word_count", 0) for chapter in chapters)
    avg_quality = sum(chapter.get("quality_score", 0) for chapter in chapters) / len(chapters) if chapters else 0
    
    return {
        "book_id": book_id,
        "title": book["title"],
        "total_chapters": len(chapters),
        "total_words": total_words,
        "estimated_reading_time": total_words // 200,  # 200 WPM
        "average_quality_score": round(avg_quality, 2),
        "current_version": book["current_version"],
        "total_versions": len(book.get("versions", {})),
        "created_at": book["created_at"],
        "last_updated": book["updated_at"]
    }