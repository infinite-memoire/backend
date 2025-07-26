import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.upload_service import UploadSessionService
from app.models.upload_session import UploadStatus, ChunkInfo
from app.services.firestore import FirestoreService

@pytest.fixture
def upload_service():
    """Create upload service with mocked dependencies"""
    with patch('app.services.upload_service.FirestoreService') as mock_firestore:
        service = UploadSessionService()
        service.firestore_service = AsyncMock(spec=FirestoreService)
        return service

@pytest.mark.asyncio
async def test_create_upload_session(upload_service):
    """Test creating a new upload session"""
    # Arrange
    filename = "test_audio.mp3"
    file_size = 10 * 1024 * 1024  # 10MB
    content_type = "audio/mpeg"
    chunk_size = 5 * 1024 * 1024  # 5MB
    
    upload_service.firestore_service.create_audio_record.return_value = "audio_123"
    upload_service.firestore_service.upsert_document.return_value = None
    
    # Act
    session = await upload_service.create_upload_session(
        filename=filename,
        file_size=file_size,
        content_type=content_type,
        chunk_size=chunk_size
    )
    
    # Assert
    assert session.filename == filename
    assert session.total_size == file_size
    assert session.content_type == content_type
    assert session.chunk_size == chunk_size
    assert session.total_chunks == 2  # 10MB / 5MB = 2 chunks
    assert session.status == UploadStatus.INITIATED
    assert len(session.chunks) == 2
    assert session.chunks[0].chunk_size == chunk_size
    assert session.chunks[1].chunk_size == file_size - chunk_size  # Last chunk is smaller
    
    # Verify Firestore calls
    upload_service.firestore_service.create_audio_record.assert_called_once()
    upload_service.firestore_service.upsert_document.assert_called_once()

@pytest.mark.asyncio
async def test_upload_chunk_success(upload_service):
    """Test successful chunk upload"""
    # Arrange
    upload_id = "upload_123"
    chunk_index = 0
    chunk_data = b"test chunk data"
    
    # Mock existing session
    mock_session = MagicMock()
    mock_session.upload_id = upload_id
    mock_session.status = UploadStatus.INITIATED
    mock_session.chunks = [ChunkInfo(chunk_index=0, chunk_size=len(chunk_data), status="pending")]
    
    upload_service.get_upload_session = AsyncMock(return_value=mock_session)
    upload_service.firestore_service.upsert_document.return_value = None
    upload_service._persist_upload_session = AsyncMock()
    
    # Act
    result = await upload_service.upload_chunk(
        upload_id=upload_id,
        chunk_index=chunk_index,
        chunk_data=chunk_data
    )
    
    # Assert
    assert result is True
    assert mock_session.chunks[0].status == "uploaded"
    assert mock_session.chunks[0].checksum is not None
    assert mock_session.status == UploadStatus.IN_PROGRESS
    
    # Verify Firestore chunk storage
    upload_service.firestore_service.upsert_document.assert_called_once()
    args, kwargs = upload_service.firestore_service.upsert_document.call_args
    assert kwargs["collection"] == "upload_chunks"
    assert kwargs["document_id"] == f"{upload_id}_{chunk_index}"

@pytest.mark.asyncio
async def test_complete_upload_success(upload_service):
    """Test successful upload completion"""
    # Arrange
    upload_id = "upload_123"
    audio_id = "audio_123"
    
    # Mock session with all chunks uploaded
    mock_session = MagicMock()
    mock_session.upload_id = upload_id
    mock_session.audio_id = audio_id
    mock_session.total_chunks = 2
    mock_session.chunks = [
        ChunkInfo(chunk_index=0, chunk_size=1024, status="uploaded"),
        ChunkInfo(chunk_index=1, chunk_size=1024, status="uploaded")
    ]
    mock_session.status = UploadStatus.IN_PROGRESS
    
    upload_service.get_upload_session = AsyncMock(return_value=mock_session)
    upload_service.firestore_service.update_audio_record = AsyncMock()
    upload_service._persist_upload_session = AsyncMock()
    
    # Act
    result = await upload_service.complete_upload(upload_id)
    
    # Assert
    assert result is True
    assert mock_session.status == UploadStatus.COMPLETED
    assert mock_session.completed_at is not None
    
    # Verify audio record update
    upload_service.firestore_service.update_audio_record.assert_called_once_with(
        audio_id,
        {
            "upload_status": "completed",
            "processing_stage": "upload_completed",
            "completed_at": mock_session.completed_at
        }
    )

@pytest.mark.asyncio
async def test_complete_upload_missing_chunks(upload_service):
    """Test upload completion with missing chunks"""
    # Arrange
    upload_id = "upload_123"
    
    # Mock session with missing chunks
    mock_session = MagicMock()
    mock_session.upload_id = upload_id
    mock_session.total_chunks = 2
    mock_session.chunks = [
        ChunkInfo(chunk_index=0, chunk_size=1024, status="uploaded"),
        ChunkInfo(chunk_index=1, chunk_size=1024, status="pending")  # Missing chunk
    ]
    
    upload_service.get_upload_session = AsyncMock(return_value=mock_session)
    
    # Act & Assert
    with pytest.raises(ValueError, match="Missing chunks: \[1\]"):
        await upload_service.complete_upload(upload_id)

@pytest.mark.asyncio
async def test_get_upload_session_expired(upload_service):
    """Test retrieving an expired upload session"""
    # Arrange
    upload_id = "upload_123"
    
    # Mock expired session data
    expired_session_doc = {
        "upload_id": upload_id,
        "audio_id": "audio_123",
        "filename": "test.mp3",
        "total_size": 1024,
        "chunk_size": 512,
        "total_chunks": 2,
        "content_type": "audio/mpeg",
        "status": "in_progress",
        "chunks": [],
        "created_at": datetime.utcnow() - timedelta(hours=25),  # Created 25 hours ago
        "expires_at": datetime.utcnow() - timedelta(hours=1),   # Expired 1 hour ago
        "metadata": {}
    }
    
    upload_service.firestore_service.get_document.return_value = expired_session_doc
    upload_service._persist_upload_session = AsyncMock()
    
    # Act
    session = await upload_service.get_upload_session(upload_id)
    
    # Assert
    assert session is not None
    assert session.status == UploadStatus.EXPIRED
    
    # Verify session was updated
    upload_service._persist_upload_session.assert_called_once()

@pytest.mark.asyncio
async def test_upload_chunk_invalid_session(upload_service):
    """Test uploading chunk to non-existent session"""
    # Arrange
    upload_id = "invalid_upload"
    chunk_index = 0
    chunk_data = b"test data"
    
    upload_service.get_upload_session = AsyncMock(return_value=None)
    
    # Act & Assert
    with pytest.raises(ValueError, match="Upload session invalid_upload not found"):
        await upload_service.upload_chunk(
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunk_data=chunk_data
        )

@pytest.mark.asyncio
async def test_upload_chunk_out_of_range(upload_service):
    """Test uploading chunk with invalid index"""
    # Arrange
    upload_id = "upload_123"
    chunk_index = 5  # Out of range
    chunk_data = b"test data"
    
    # Mock session with only 2 chunks
    mock_session = MagicMock()
    mock_session.status = UploadStatus.INITIATED
    mock_session.chunks = [
        ChunkInfo(chunk_index=0, chunk_size=1024),
        ChunkInfo(chunk_index=1, chunk_size=1024)
    ]
    
    upload_service.get_upload_session = AsyncMock(return_value=mock_session)
    
    # Act & Assert
    with pytest.raises(ValueError, match="Chunk index 5 out of range"):
        await upload_service.upload_chunk(
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunk_data=chunk_data
        )

if __name__ == "__main__":
    pytest.main([__file__])