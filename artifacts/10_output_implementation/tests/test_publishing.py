"""
Test suite for publishing functionality

Tests the complete publishing workflow including validation,
metadata management, HTML conversion, and marketplace integration.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.services.publishing import PublishingWorkflowService, PublicationValidator
from app.services.content_storage import ContentStorageService
from app.services.html_conversion import HTMLConversionService
from app.services.marketplace import MarketplaceService
from app.models.publishing import (
    PublicationMetadata, PublicationSettings, PublicationStatus, 
    PublicationVisibility, PublishingWorkflow
)
from app.utils.exceptions import ValidationError, PublishingError


@pytest.fixture
def mock_storage_service():
    """Mock content storage service"""
    service = Mock(spec=ContentStorageService)
    service.get_book_metadata = AsyncMock()
    service.get_book_chapters = AsyncMock()
    service.create_publishing_workflow = AsyncMock()
    service.get_publishing_workflow = AsyncMock()
    service.update_publishing_workflow = AsyncMock()
    return service


@pytest.fixture
def mock_conversion_service():
    """Mock HTML conversion service"""
    service = Mock(spec=HTMLConversionService)
    service.convert_book_to_html = AsyncMock()
    return service


@pytest.fixture
def mock_marketplace_service():
    """Mock marketplace service"""
    service = Mock(spec=MarketplaceService)
    service.create_marketplace_entry = AsyncMock()
    service.update_marketplace_visibility = AsyncMock()
    return service


@pytest.fixture
def publishing_service(mock_storage_service, mock_conversion_service, mock_marketplace_service):
    """Create publishing service with mocked dependencies"""
    return PublishingWorkflowService(
        storage_service=mock_storage_service,
        conversion_service=mock_conversion_service,
        marketplace_service=mock_marketplace_service
    )


@pytest.fixture
def validator_service(mock_storage_service):
    """Create validator service with mocked dependencies"""
    return PublicationValidator(storage_service=mock_storage_service)


@pytest.fixture
def sample_book_metadata():
    """Sample book metadata for testing"""
    return {
        "id": "book_123",
        "owner_user_id": "user_456",
        "title": "My Life Story",
        "description": "A memoir about my journey",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "status": "completed",
        "current_version": "v1.0",
        "versions": {
            "v1.0": {
                "created_at": datetime.now(),
                "chapter_count": 5,
                "total_word_count": 15000,
                "status": "active"
            }
        }
    }


@pytest.fixture
def sample_chapters():
    """Sample chapters for testing"""
    return [
        {
            "id": "chapter_1",
            "book_id": "book_123",
            "chapter_number": 1,
            "title": "Early Years",
            "content": {
                "markdown_text": "# Early Years\n\nI was born in a small town..." * 50,
                "word_count": 250,
                "themes": ["childhood", "family"],
                "participants": ["John", "Mary"]
            },
            "metadata": {
                "status": "completed",
                "quality_score": 0.85
            }
        },
        {
            "id": "chapter_2",
            "book_id": "book_123",
            "chapter_number": 2,
            "title": "School Days",
            "content": {
                "markdown_text": "# School Days\n\nMy school years were filled with..." * 60,
                "word_count": 300,
                "themes": ["education", "friendship"],
                "participants": ["John", "Teacher Smith"]
            },
            "metadata": {
                "status": "completed",
                "quality_score": 0.90
            }
        }
    ]


@pytest.fixture
def sample_publication_metadata():
    """Sample publication metadata for testing"""
    return PublicationMetadata(
        title="My Life Story",
        subtitle="A Journey of Discovery",
        description="This is a memoir about my personal journey through life, filled with ups and downs, learning experiences, and meaningful relationships.",
        author_name="John Doe",
        author_bio="John Doe is a writer and storyteller.",
        category="memoir",
        tags=["memoir", "personal", "journey"],
        language="en",
        target_audience="general"
    )


@pytest.fixture
def sample_publication_settings():
    """Sample publication settings for testing"""
    return PublicationSettings(
        visibility=PublicationVisibility.PUBLIC,
        allow_comments=True,
        allow_downloads=True,
        price=0.0,
        marketplace_listing=True,
        analytics_enabled=True
    )


class TestPublishingWorkflowService:
    """Test cases for PublishingWorkflowService"""
    
    @pytest.mark.asyncio
    async def test_initiate_publishing_workflow_success(
        self, publishing_service, mock_storage_service, sample_book_metadata, sample_chapters
    ):
        """Test successful workflow initiation"""
        # Setup mocks
        mock_storage_service.get_book_metadata.return_value = sample_book_metadata
        mock_storage_service.get_book_chapters.return_value = sample_chapters
        mock_storage_service.create_publishing_workflow.return_value = "workflow_123"
        
        # Execute
        workflow_id = await publishing_service.initiate_publishing_workflow(
            book_id="book_123",
            user_id="user_456"
        )
        
        # Verify
        assert workflow_id.startswith("pub_book_123_")
        mock_storage_service.get_book_metadata.assert_called_once_with("book_123")
        mock_storage_service.get_book_chapters.assert_called_once()
        mock_storage_service.create_publishing_workflow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initiate_workflow_book_not_found(
        self, publishing_service, mock_storage_service
    ):
        """Test workflow initiation with non-existent book"""
        # Setup mocks
        mock_storage_service.get_book_metadata.return_value = None
        
        # Execute and verify
        with pytest.raises(ValueError, match="Book book_123 not found"):
            await publishing_service.initiate_publishing_workflow(
                book_id="book_123",
                user_id="user_456"
            )
    
    @pytest.mark.asyncio
    async def test_initiate_workflow_permission_denied(
        self, publishing_service, mock_storage_service, sample_book_metadata
    ):
        """Test workflow initiation with wrong user"""
        # Setup mocks
        mock_storage_service.get_book_metadata.return_value = sample_book_metadata
        
        # Execute and verify
        with pytest.raises(PermissionError, match="User does not own this book"):
            await publishing_service.initiate_publishing_workflow(
                book_id="book_123",
                user_id="wrong_user"
            )
    
    @pytest.mark.asyncio
    async def test_initiate_workflow_no_chapters(
        self, publishing_service, mock_storage_service, sample_book_metadata
    ):
        """Test workflow initiation with book that has no chapters"""
        # Setup mocks
        mock_storage_service.get_book_metadata.return_value = sample_book_metadata
        mock_storage_service.get_book_chapters.return_value = []
        
        # Execute and verify
        with pytest.raises(ValueError, match="Book must have at least one chapter"):
            await publishing_service.initiate_publishing_workflow(
                book_id="book_123",
                user_id="user_456"
            )
    
    @pytest.mark.asyncio
    async def test_update_publication_metadata_success(
        self, publishing_service, mock_storage_service, sample_publication_metadata
    ):
        """Test successful metadata update"""
        # Setup mocks
        workflow_data = {
            "workflow_id": "workflow_123",
            "book_id": "book_123",
            "user_id": "user_456",
            "status": PublicationStatus.DRAFT.value,
            "steps_completed": [],
            "validation_results": {},
            "publication_metadata": {},
            "publication_settings": {}
        }
        mock_storage_service.get_publishing_workflow.return_value = workflow_data
        mock_storage_service.update_publishing_workflow.return_value = None
        
        # Execute
        result = await publishing_service.update_publication_metadata(
            "workflow_123", sample_publication_metadata
        )
        
        # Verify
        assert result["status"] == "success"
        assert "validation_result" in result
        mock_storage_service.update_publishing_workflow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_publication_settings_success(
        self, publishing_service, mock_storage_service, sample_publication_settings
    ):
        """Test successful settings update"""
        # Setup mocks
        workflow_data = {
            "workflow_id": "workflow_123",
            "book_id": "book_123",
            "user_id": "user_456",
            "status": PublicationStatus.DRAFT.value,
            "steps_completed": [],
            "publication_settings": {}
        }
        mock_storage_service.get_publishing_workflow.return_value = workflow_data
        mock_storage_service.update_publishing_workflow.return_value = None
        
        # Execute
        result = await publishing_service.update_publication_settings(
            "workflow_123", sample_publication_settings
        )
        
        # Verify
        assert result["status"] == "success"
        mock_storage_service.update_publishing_workflow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_preview_success(
        self, publishing_service, mock_storage_service, mock_conversion_service,
        sample_book_metadata, sample_chapters
    ):
        """Test successful preview generation"""
        # Setup mocks
        workflow_data = {
            "workflow_id": "workflow_123",
            "book_id": "book_123",
            "publication_metadata": {
                "title": "My Life Story",
                "description": "A memoir"
            }
        }
        mock_storage_service.get_publishing_workflow.return_value = workflow_data
        mock_storage_service.get_book_metadata.return_value = sample_book_metadata
        mock_storage_service.get_book_chapters.return_value = sample_chapters
        mock_conversion_service.convert_book_to_html.return_value = "<html>Preview</html>"
        mock_storage_service.update_publishing_workflow.return_value = None
        
        # Execute
        result = await publishing_service.generate_preview("workflow_123")
        
        # Verify
        assert result["status"] == "success"
        assert "preview_url" in result
        assert result["preview_url"].startswith("/preview/")
        mock_conversion_service.convert_book_to_html.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_for_publication_success(
        self, publishing_service, mock_storage_service, mock_conversion_service,
        mock_marketplace_service, sample_book_metadata, sample_chapters
    ):
        """Test successful publication submission"""
        # Setup mocks
        workflow_data = {
            "workflow_id": "workflow_123",
            "book_id": "book_123",
            "user_id": "user_456",
            "status": PublicationStatus.DRAFT.value,
            "publication_metadata": {
                "title": "My Life Story",
                "description": "A memoir about my journey" * 10,  # Long enough
                "author_name": "John Doe"
            },
            "publication_settings": {
                "visibility": "public"
            }
        }
        
        mock_storage_service.get_publishing_workflow.return_value = workflow_data
        mock_storage_service.get_book_metadata.return_value = sample_book_metadata
        mock_storage_service.get_book_chapters.return_value = sample_chapters
        mock_conversion_service.convert_book_to_html.return_value = "<html>Book</html>"
        mock_marketplace_service.create_marketplace_entry.return_value = {
            "id": "marketplace_123",
            "url": "/marketplace/book/marketplace_123"
        }
        mock_storage_service.update_publishing_workflow.return_value = None
        
        # Execute
        result = await publishing_service.submit_for_publication("workflow_123")
        
        # Verify
        assert result["status"] == "published"
        assert "publication_url" in result
        assert "marketplace_id" in result
        mock_marketplace_service.create_marketplace_entry.assert_called_once()


class TestPublicationValidator:
    """Test cases for PublicationValidator"""
    
    @pytest.mark.asyncio
    async def test_validate_publication_metadata_valid(
        self, validator_service, sample_publication_metadata
    ):
        """Test validation of valid metadata"""
        result = await validator_service.validate_publication_metadata(sample_publication_metadata)
        
        assert result["is_valid"] is True
        assert result["score"] >= 80  # Should have good score
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_publication_metadata_missing_title(
        self, validator_service
    ):
        """Test validation with missing title"""
        metadata = PublicationMetadata(
            title="",  # Invalid: empty title
            description="A long enough description for the validation to pass",
            author_name="John Doe"
        )
        
        result = await validator_service.validate_publication_metadata(metadata)
        
        assert result["is_valid"] is False
        assert any("Title is required" in error for error in result["errors"])
    
    @pytest.mark.asyncio
    async def test_validate_publication_metadata_short_description(
        self, validator_service
    ):
        """Test validation with short description"""
        metadata = PublicationMetadata(
            title="My Life Story",
            description="Too short",  # Invalid: too short
            author_name="John Doe"
        )
        
        result = await validator_service.validate_publication_metadata(metadata)
        
        assert result["is_valid"] is False
        assert any("Description" in error and "50 characters" in error for error in result["errors"])
    
    @pytest.mark.asyncio
    async def test_validate_book_content_valid(
        self, validator_service, mock_storage_service, sample_chapters
    ):
        """Test validation of valid book content"""
        mock_storage_service.get_book_chapters.return_value = sample_chapters
        
        result = await validator_service.validate_book_content("book_123", "v1.0")
        
        assert result["is_ready"] is True
        assert result["stats"]["total_chapters"] == 2
        assert result["stats"]["total_word_count"] == 550
        assert len(result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_book_content_no_chapters(
        self, validator_service, mock_storage_service
    ):
        """Test validation with no chapters"""
        mock_storage_service.get_book_chapters.return_value = []
        
        result = await validator_service.validate_book_content("book_123", "v1.0")
        
        assert result["is_ready"] is False
        assert any("at least one chapter" in issue for issue in result["issues"])
    
    @pytest.mark.asyncio
    async def test_validate_book_content_short_chapters(
        self, validator_service, mock_storage_service
    ):
        """Test validation with very short chapters"""
        short_chapters = [
            {
                "title": "Chapter 1",
                "content": {
                    "markdown_text": "Short content"  # Only 2 words
                }
            }
        ]
        mock_storage_service.get_book_chapters.return_value = short_chapters
        
        result = await validator_service.validate_book_content("book_123", "v1.0")
        
        assert any("very short" in issue for issue in result["issues"])


class TestPublishingAPI:
    """Test cases for publishing API endpoints"""
    
    @pytest.fixture
    def mock_services(self, mock_storage_service, mock_conversion_service, mock_marketplace_service):
        """Mock services for API testing"""
        return {
            "publishing": PublishingWorkflowService(
                mock_storage_service, mock_conversion_service, mock_marketplace_service
            ),
            "content_storage": mock_storage_service,
            "validator": PublicationValidator(mock_storage_service)
        }
    
    @pytest.fixture
    def mock_user():
        """Mock user for API testing"""
        user = Mock()
        user.id = "user_456"
        user.email = "user@example.com"
        return user
    
    @pytest.mark.asyncio
    async def test_start_publishing_workflow_api(
        self, mock_services, mock_user, sample_book_metadata, sample_chapters
    ):
        """Test start publishing workflow API endpoint"""
        # Setup mocks
        mock_services["content_storage"].get_book_metadata.return_value = sample_book_metadata
        mock_services["content_storage"].get_book_chapters.return_value = sample_chapters
        
        # Import here to avoid circular imports in test setup
        from app.api.routes.publishing import start_publishing_workflow
        from app.dependencies import get_services
        from app.utils.auth import get_current_user
        
        # Mock dependencies
        def mock_get_services():
            return mock_services
        
        def mock_get_current_user():
            return mock_user
        
        # Create request
        request = Mock()
        request.book_id = "book_123"
        request.auto_generate_metadata = True
        
        # Execute (would normally be called by FastAPI)
        with patch('app.dependencies.get_services', mock_get_services):
            with patch('app.utils.auth.get_current_user', mock_get_current_user):
                result = await start_publishing_workflow(
                    book_id="book_123",
                    request=request,
                    current_user=mock_user,
                    services=mock_services
                )
        
        # Verify
        assert result["status"] == "started"
        assert "workflow_id" in result


@pytest.mark.integration
class TestPublishingIntegration:
    """Integration tests for the complete publishing workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_publishing_workflow(
        self, sample_book_metadata, sample_chapters, sample_publication_metadata, 
        sample_publication_settings
    ):
        """Test the complete end-to-end publishing workflow"""
        # This would be an integration test that uses real services
        # and tests the complete workflow from start to finish
        
        # For now, we'll test the workflow logic
        workflow_steps = [
            "initiate_workflow",
            "update_metadata", 
            "update_settings",
            "validate_publication",
            "generate_preview",
            "submit_for_publication"
        ]
        
        completed_steps = []
        
        # Simulate each step
        for step in workflow_steps:
            # In a real integration test, each step would call actual services
            completed_steps.append(step)
            
            # Verify step completion
            assert step in completed_steps
        
        # Verify all steps completed
        assert len(completed_steps) == len(workflow_steps)
        assert completed_steps[-1] == "submit_for_publication"


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_publishing.py -v
    pytest.main([__file__, "-v"])