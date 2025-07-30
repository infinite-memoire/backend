# Chunked Upload Handler - Hierarchical Breakdown

Based on Docker deployment configuration and API endpoint requirements for large audio file uploads.

## 1. Chunked Upload Architecture
### 1.1 Upload Flow Design
#### 1.1.1 Multi-stage Upload Process
```
Client → Initiate Upload → Backend
       ↓
Backend → Create Upload Session → Firestore
       ↓
Client → Upload Chunks (parallel) → Backend
       ↓
Backend → Store Chunks → Firestore
       ↓
Client → Complete Upload → Backend
       ↓
Backend → Assemble & Process → Background Task
```

#### 1.1.2 Upload Session Management
```python
# app/models/upload_session.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

class UploadStatus(Enum):
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class ChunkInfo:
    chunk_index: int
    chunk_size: int
    uploaded_at: Optional[datetime] = None
    checksum: Optional[str] = None
    status: str = "pending"

@dataclass
class UploadSession:
    upload_id: str
    audio_id: str
    filename: str
    total_size: int
    chunk_size: int
    total_chunks: int
    content_type: str
    status: UploadStatus
    chunks: List[ChunkInfo]
    created_at: datetime
    expires_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict = None
```

### 1.2 Upload Validation System
#### 1.2.1 File Validation Rules

```python
# app/utils/upload_validation.py
from app.config.settings_config import get_settings
from app.utils.logging_utils import get_logger
import mimetypes
from typing import Tuple, Optional

logger = get_logger("upload_validation")


class UploadValidationError(Exception):
    """Exception raised for upload validation errors"""
    pass


class FileValidator:
    def __init__(self):
        self.settings = get_settings()

    def validate_upload_request(
            self,
            filename: str,
            file_size: int,
            content_type: str,
            chunk_size: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate upload request parameters"""

        try:
            # Validate file size
            max_size_bytes = self.settings.upload.max_upload_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                raise UploadValidationError(
                    f"File size {file_size} exceeds maximum allowed size {max_size_bytes}"
                )

            if file_size <= 0:
                raise UploadValidationError("File size must be greater than 0")

            # Validate content type
            if content_type not in self.settings.upload.allowed_audio_formats:
                raise UploadValidationError(
                    f"Content type {content_type} not allowed. "
                    f"Allowed types: {self.settings.upload.allowed_audio_formats}"
                )

            # Validate filename
            if not filename or len(filename.strip()) == 0:
                raise UploadValidationError("Filename cannot be empty")

            if len(filename) > 255:
                raise UploadValidationError("Filename too long (max 255 characters)")

            # Validate chunk size
            max_chunk_size = self.settings.upload.chunk_size_mb * 1024 * 1024
            if chunk_size > max_chunk_size:
                raise UploadValidationError(
                    f"Chunk size {chunk_size} exceeds maximum {max_chunk_size}"
                )

            if chunk_size <= 0:
                raise UploadValidationError("Chunk size must be greater than 0")

            # Validate file extension matches content type
            expected_content_type = mimetypes.guess_type(filename)[0]
            if expected_content_type and expected_content_type != content_type:
                logger.warning("Content type mismatch",
                               filename=filename,
                               declared_type=content_type,
                               expected_type=expected_content_type)

            logger.info("Upload validation successful",
                        filename=filename,
                        file_size=file_size,
                        content_type=content_type,
                        chunk_size=chunk_size)

            return True, None

        except UploadValidationError as e:
            logger.error("Upload validation failed",
                         filename=filename,
                         error_message=str(e))
            return False, str(e)
        except Exception as e:
            logger.error("Unexpected validation error",
                         filename=filename,
                         error_type=type(e).__name__,
                         error_message=str(e))
            return False, f"Validation error: {str(e)}"

    def validate_chunk_data(
            self,
            chunk_data: bytes,
            expected_size: int,
            chunk_index: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate individual chunk data"""

        try:
            actual_size = len(chunk_data)

            # For last chunk, size may be smaller
            if actual_size > expected_size:
                raise UploadValidationError(
                    f"Chunk {chunk_index} size {actual_size} exceeds expected {expected_size}"
                )

            if actual_size == 0:
                raise UploadValidationError(f"Chunk {chunk_index} is empty")

            logger.debug("Chunk validation successful",
                         chunk_index=chunk_index,
                         actual_size=actual_size,
                         expected_size=expected_size)

            return True, None

        except UploadValidationError as e:
            logger.error("Chunk validation failed",
                         chunk_index=chunk_index,
                         error_message=str(e))
            return False, str(e)


# Global validator instance
file_validator = FileValidator()
```

## 2. Upload Session Service
### 2.1 Session Management
#### 2.1.1 Upload Session Service

```python
# app/services/upload_service.py
import uuid
from datetime import datetime, timedelta
from typing import Optional, List
from app.services.firestore import FirestoreService
from app.models.upload_session import UploadSession, ChunkInfo, UploadStatus
from app.utils.logging_utils import get_logger, log_performance
from app.config.settings_config import get_settings
import hashlib

logger = get_logger("upload_service")


class UploadSessionService:
    def __init__(self):
        self.firestore_service = FirestoreService()
        self.settings = get_settings()

    @log_performance(logger)
    async def create_upload_session(
            self,
            filename: str,
            file_size: int,
            content_type: str,
            chunk_size: int
    ) -> UploadSession:
        """Create a new upload session"""

        try:
            upload_id = str(uuid.uuid4())
            audio_id = str(uuid.uuid4())

            # Calculate number of chunks
            total_chunks = (file_size + chunk_size - 1) // chunk_size

            # Create chunk info list
            chunks = [
                ChunkInfo(chunk_index=i, chunk_size=chunk_size)
                for i in range(total_chunks)
            ]

            # Adjust last chunk size
            if total_chunks > 0:
                last_chunk_size = file_size - (total_chunks - 1) * chunk_size
                chunks[-1].chunk_size = last_chunk_size

            # Create upload session
            session = UploadSession(
                upload_id=upload_id,
                audio_id=audio_id,
                filename=filename,
                total_size=file_size,
                chunk_size=chunk_size,
                total_chunks=total_chunks,
                content_type=content_type,
                status=UploadStatus.INITIATED,
                chunks=chunks,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24),
                metadata={}
            )

            # Store session in Firestore
            await self._persist_upload_session(session)

            # Create audio record
            audio_data = {
                "id": audio_id,
                "filename": filename,
                "file_size_bytes": file_size,
                "content_type": content_type,
                "upload_status": "initiated",
                "upload_id": upload_id,
                "created_at": datetime.utcnow(),
                "processing_stage": "upload_initiated"
            }

            await self.firestore_service.create_audio_record(audio_data)

            logger.info("Upload session created",
                        upload_id=upload_id,
                        audio_id=audio_id,
                        filename=filename,
                        total_chunks=total_chunks,
                        file_size=file_size)

            return session

        except Exception as e:
            logger.error("Failed to create upload session",
                         filename=filename,
                         file_size=file_size,
                         error_type=type(e).__name__,
                         error_message=str(e))
            raise

    @log_performance(logger)
    async def get_upload_session(self, upload_id: str) -> Optional[UploadSession]:
        """Retrieve upload session by ID"""

        try:
            session_doc = await self.firestore_service.get_document(
                collection="upload_sessions",
                document_id=upload_id
            )

            if not session_doc:
                logger.warning("Upload session not found", upload_id=upload_id)
                return None

            # Convert document to UploadSession object
            session = self._doc_to_upload_session(session_doc)

            # Check if session has expired
            if session.expires_at < datetime.utcnow():
                session.status = UploadStatus.EXPIRED
                await self._persist_upload_session(session)
                logger.warning("Upload session expired", upload_id=upload_id)

            return session

        except Exception as e:
            logger.error("Failed to retrieve upload session",
                         upload_id=upload_id,
                         error_type=type(e).__name__,
                         error_message=str(e))
            raise

    @log_performance(logger)
    async def upload_chunk(
            self,
            upload_id: str,
            chunk_index: int,
            chunk_data: bytes
    ) -> bool:
        """Upload a single chunk"""

        try:
            # Get upload session
            session = await self.get_upload_session(upload_id)
            if not session:
                raise ValueError(f"Upload session {upload_id} not found")

            if session.status != UploadStatus.INITIATED and session.status != UploadStatus.IN_PROGRESS:
                raise ValueError(f"Upload session {upload_id} is not active (status: {session.status})")

            if chunk_index >= len(session.chunks):
                raise ValueError(f"Chunk index {chunk_index} out of range")

            chunk_info = session.chunks[chunk_index]

            if chunk_info.status == "uploaded":
                logger.warning("Chunk already uploaded",
                               upload_id=upload_id,
                               chunk_index=chunk_index)
                return True

            # Calculate checksum
            checksum = hashlib.md5(chunk_data).hexdigest()

            # Store chunk data in Firestore
            chunk_doc_id = f"{upload_id}_{chunk_index}"
            chunk_doc = {
                "upload_id": upload_id,
                "chunk_index": chunk_index,
                "data": chunk_data,
                "checksum": checksum,
                "uploaded_at": datetime.utcnow(),
                "size": len(chunk_data)
            }

            await self.firestore_service.upsert_document(
                collection="upload_chunks",
                document_id=chunk_doc_id,
                data=chunk_doc
            )

            # Update chunk info
            chunk_info.status = "uploaded"
            chunk_info.uploaded_at = datetime.utcnow()
            chunk_info.checksum = checksum

            # Update session status
            if session.status == UploadStatus.INITIATED:
                session.status = UploadStatus.IN_PROGRESS

            # Persist updated session
            await self._persist_upload_session(session)

            logger.info("Chunk uploaded successfully",
                        upload_id=upload_id,
                        chunk_index=chunk_index,
                        chunk_size=len(chunk_data),
                        checksum=checksum)

            return True

        except Exception as e:
            logger.error("Failed to upload chunk",
                         upload_id=upload_id,
                         chunk_index=chunk_index,
                         error_type=type(e).__name__,
                         error_message=str(e))
            raise

    @log_performance(logger)
    async def complete_upload(self, upload_id: str) -> bool:
        """Complete the upload and trigger processing"""

        try:
            # Get upload session
            session = await self.get_upload_session(upload_id)
            if not session:
                raise ValueError(f"Upload session {upload_id} not found")

            # Check all chunks are uploaded
            uploaded_chunks = [chunk for chunk in session.chunks if chunk.status == "uploaded"]
            if len(uploaded_chunks) != session.total_chunks:
                missing_chunks = [i for i, chunk in enumerate(session.chunks) if chunk.status != "uploaded"]
                raise ValueError(f"Missing chunks: {missing_chunks}")

            # Update session status
            session.status = UploadStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            await self._persist_upload_session(session)

            # Update audio record
            await self.firestore_service.update_audio_record(session.audio_id, {
                "upload_status": "completed",
                "processing_stage": "upload_completed",
                "completed_at": datetime.utcnow()
            })

            # Trigger background processing
            from app.services.background_tasks import task_manager, process_audio_upload
            task_id = await task_manager.create_task(
                task_type="audio_upload_processing",
                task_function=process_audio_upload,
                task_args=(session.audio_id, upload_id),
                metadata={"upload_id": upload_id, "audio_id": session.audio_id}
            )

            logger.info("Upload completed successfully",
                        upload_id=upload_id,
                        audio_id=session.audio_id,
                        total_chunks=session.total_chunks,
                        processing_task_id=task_id)

            return True

        except Exception as e:
            # Update session with error
            if 'session' in locals():
                session.status = UploadStatus.FAILED
                session.error_message = str(e)
                await self._persist_upload_session(session)

            logger.error("Failed to complete upload",
                         upload_id=upload_id,
                         error_type=type(e).__name__,
                         error_message=str(e))
            raise

    async def _persist_upload_session(self, session: UploadSession):
        """Persist upload session to Firestore"""

        session_data = {
            "upload_id": session.upload_id,
            "audio_id": session.audio_id,
            "filename": session.filename,
            "total_size": session.total_size,
            "chunk_size": session.chunk_size,
            "total_chunks": session.total_chunks,
            "content_type": session.content_type,
            "status": session.status.value,
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "chunk_size": chunk.chunk_size,
                    "uploaded_at": chunk.uploaded_at,
                    "checksum": chunk.checksum,
                    "status": chunk.status
                }
                for chunk in session.chunks
            ],
            "created_at": session.created_at,
            "expires_at": session.expires_at,
            "completed_at": session.completed_at,
            "error_message": session.error_message,
            "metadata": session.metadata,
            "updated_at": datetime.utcnow()
        }

        await self.firestore_service.upsert_document(
            collection="upload_sessions",
            document_id=session.upload_id,
            data=session_data
        )

    def _doc_to_upload_session(self, doc: dict) -> UploadSession:
        """Convert Firestore document to UploadSession object"""

        chunks = [
            ChunkInfo(
                chunk_index=chunk_data["chunk_index"],
                chunk_size=chunk_data["chunk_size"],
                uploaded_at=chunk_data.get("uploaded_at"),
                checksum=chunk_data.get("checksum"),
                status=chunk_data.get("status", "pending")
            )
            for chunk_data in doc.get("chunks", [])
        ]

        return UploadSession(
            upload_id=doc["upload_id"],
            audio_id=doc["audio_id"],
            filename=doc["filename"],
            total_size=doc["total_size"],
            chunk_size=doc["chunk_size"],
            total_chunks=doc["total_chunks"],
            content_type=doc["content_type"],
            status=UploadStatus(doc["status"]),
            chunks=chunks,
            created_at=doc["created_at"],
            expires_at=doc["expires_at"],
            completed_at=doc.get("completed_at"),
            error_message=doc.get("error_message"),
            metadata=doc.get("metadata", {})
        )


# Global upload service instance
upload_service = UploadSessionService()
```

## 3. API Route Implementation
### 3.1 Upload Endpoints
#### 3.1.1 Chunked Upload Routes

```python
# app/api/routes/upload.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from app.services.upload_service import upload_service
from app.utils.upload_validation import file_validator
from app.utils.logging_utils import get_logger, log_performance
import asyncio

router = APIRouter(tags=["upload"])
logger = get_logger("upload_api")


class InitiateUploadRequest(BaseModel):
    filename: str
    file_size: int
    content_type: str
    chunk_size: Optional[int] = 5 * 1024 * 1024  # 5MB default


class InitiateUploadResponse(BaseModel):
    upload_id: str
    audio_id: str
    total_chunks: int
    chunk_size: int
    expires_at: str
    upload_urls: list


@router.post("/initiate", response_model=InitiateUploadResponse)
@log_performance(logger)
async def initiate_upload(request: InitiateUploadRequest):
    """Initiate a chunked upload session"""

    try:
        # Validate upload request
        is_valid, error_message = file_validator.validate_upload_request(
            filename=request.filename,
            file_size=request.file_size,
            content_type=request.content_type,
            chunk_size=request.chunk_size
        )

        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)

        # Create upload session
        session = await upload_service.create_upload_session(
            filename=request.filename,
            file_size=request.file_size,
            content_type=request.content_type,
            chunk_size=request.chunk_size
        )

        # Generate upload URLs for each chunk
        upload_urls = [
            f"/api/v1/upload/chunk/{session.upload_id}/{i}"
            for i in range(session.total_chunks)
        ]

        logger.info("Upload initiated successfully",
                    upload_id=session.upload_id,
                    filename=request.filename,
                    total_chunks=session.total_chunks)

        return InitiateUploadResponse(
            upload_id=session.upload_id,
            audio_id=session.audio_id,
            total_chunks=session.total_chunks,
            chunk_size=session.chunk_size,
            expires_at=session.expires_at.isoformat(),
            upload_urls=upload_urls
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to initiate upload",
                     filename=request.filename,
                     error_type=type(e).__name__,
                     error_message=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/chunk/{upload_id}/{chunk_index}")
@log_performance(logger)
async def upload_chunk(
        upload_id: str,
        chunk_index: int,
        file: UploadFile = File(...)
):
    """Upload a single chunk"""

    try:
        # Validate chunk index
        if chunk_index < 0:
            raise HTTPException(status_code=400, detail="Chunk index must be non-negative")

        # Read chunk data
        chunk_data = await file.read()

        # Validate chunk data
        session = await upload_service.get_upload_session(upload_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")

        if chunk_index >= len(session.chunks):
            raise HTTPException(status_code=400, detail="Chunk index out of range")

        expected_size = session.chunks[chunk_index].chunk_size
        is_valid, error_message = file_validator.validate_chunk_data(
            chunk_data=chunk_data,
            expected_size=expected_size,
            chunk_index=chunk_index
        )

        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)

        # Upload chunk
        success = await upload_service.upload_chunk(
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunk_data=chunk_data
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to upload chunk")

        logger.info("Chunk uploaded successfully",
                    upload_id=upload_id,
                    chunk_index=chunk_index,
                    chunk_size=len(chunk_data))

        return {
            "upload_id": upload_id,
            "chunk_index": chunk_index,
            "status": "uploaded",
            "chunk_size": len(chunk_data)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload chunk",
                     upload_id=upload_id,
                     chunk_index=chunk_index,
                     error_type=type(e).__name__,
                     error_message=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/complete/{upload_id}")
@log_performance(logger)
async def complete_upload(upload_id: str):
    """Complete the upload and start processing"""

    try:
        # Complete upload
        success = await upload_service.complete_upload(upload_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to complete upload")

        # Get session info for response
        session = await upload_service.get_upload_session(upload_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")

        logger.info("Upload completed successfully",
                    upload_id=upload_id,
                    audio_id=session.audio_id)

        return {
            "upload_id": upload_id,
            "audio_id": session.audio_id,
            "status": "completed",
            "total_chunks": session.total_chunks,
            "processing_started": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to complete upload",
                     upload_id=upload_id,
                     error_type=type(e).__name__,
                     error_message=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{upload_id}")
@log_performance(logger)
async def get_upload_status(upload_id: str):
    """Get upload session status"""

    try:
        session = await upload_service.get_upload_session(upload_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")

        # Calculate progress
        uploaded_chunks = len([chunk for chunk in session.chunks if chunk.status == "uploaded"])
        progress_percentage = int((uploaded_chunks / session.total_chunks) * 100) if session.total_chunks > 0 else 0

        return {
            "upload_id": upload_id,
            "audio_id": session.audio_id,
            "status": session.status.value,
            "progress_percentage": progress_percentage,
            "uploaded_chunks": uploaded_chunks,
            "total_chunks": session.total_chunks,
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "error_message": session.error_message
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get upload status",
                     upload_id=upload_id,
                     error_type=type(e).__name__,
                     error_message=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
```

## 4. Client-Side Integration Support
### 4.1 Upload Progress Tracking
#### 4.1.1 Progress Callback System
```python
# app/utils/upload_progress.py
from typing import Callable, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class UploadProgress:
    upload_id: str
    total_chunks: int
    uploaded_chunks: int
    current_chunk: Optional[int]
    bytes_uploaded: int
    total_bytes: int
    progress_percentage: float
    upload_speed_bps: float
    estimated_time_remaining: Optional[float]
    status: str
    timestamp: datetime

class UploadProgressTracker:
    """Track upload progress and provide real-time updates"""
    
    def __init__(self, upload_id: str, total_chunks: int, total_bytes: int):
        self.upload_id = upload_id
        self.total_chunks = total_chunks
        self.total_bytes = total_bytes
        self.uploaded_chunks = 0
        self.bytes_uploaded = 0
        self.start_time = datetime.utcnow()
        self.last_update = self.start_time
        self.callbacks: list[Callable[[UploadProgress], None]] = []
    
    def add_progress_callback(self, callback: Callable[[UploadProgress], None]):
        """Add a progress callback function"""
        self.callbacks.append(callback)
    
    def update_chunk_progress(self, chunk_index: int, chunk_size: int):
        """Update progress when a chunk is uploaded"""
        self.uploaded_chunks += 1
        self.bytes_uploaded += chunk_size
        
        now = datetime.utcnow()
        elapsed_seconds = (now - self.start_time).total_seconds()
        
        # Calculate progress percentage
        progress_percentage = (self.bytes_uploaded / self.total_bytes) * 100
        
        # Calculate upload speed
        upload_speed_bps = self.bytes_uploaded / elapsed_seconds if elapsed_seconds > 0 else 0
        
        # Estimate time remaining
        remaining_bytes = self.total_bytes - self.bytes_uploaded
        estimated_time_remaining = remaining_bytes / upload_speed_bps if upload_speed_bps > 0 else None
        
        # Create progress object
        progress = UploadProgress(
            upload_id=self.upload_id,
            total_chunks=self.total_chunks,
            uploaded_chunks=self.uploaded_chunks,
            current_chunk=chunk_index,
            bytes_uploaded=self.bytes_uploaded,
            total_bytes=self.total_bytes,
            progress_percentage=progress_percentage,
            upload_speed_bps=upload_speed_bps,
            estimated_time_remaining=estimated_time_remaining,
            status="uploading",
            timestamp=now
        )
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning("Progress callback failed",
                             upload_id=self.upload_id,
                             error_message=str(e))
        
        self.last_update = now
```

### 4.2 Error Recovery and Retry Logic
#### 4.2.1 Chunk Upload Retry System

```python
# app/utils/upload_retry.py
import asyncio
from typing import Optional, Callable
from app.utils.logging_utils import get_logger

logger = get_logger("upload_retry")


class UploadRetryHandler:
    """Handle chunk upload retries with exponential backoff"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def retry_chunk_upload(
            self,
            upload_func: Callable,
            upload_id: str,
            chunk_index: int,
            chunk_data: bytes,
            **kwargs
    ) -> bool:
        """Retry chunk upload with exponential backoff"""

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Attempt upload
                success = await upload_func(
                    upload_id=upload_id,
                    chunk_index=chunk_index,
                    chunk_data=chunk_data,
                    **kwargs
                )

                if success:
                    if attempt > 0:
                        logger.info("Chunk upload succeeded after retry",
                                    upload_id=upload_id,
                                    chunk_index=chunk_index,
                                    attempt=attempt)
                    return True

            except Exception as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning("Chunk upload failed, retrying",
                                   upload_id=upload_id,
                                   chunk_index=chunk_index,
                                   attempt=attempt,
                                   delay_seconds=delay,
                                   error_message=str(e))
                    await asyncio.sleep(delay)
                else:
                    logger.error("Chunk upload failed after all retries",
                                 upload_id=upload_id,
                                 chunk_index=chunk_index,
                                 total_attempts=attempt + 1,
                                 error_message=str(e))

        # All attempts failed
        raise last_error if last_error else Exception("Upload failed for unknown reason")
```

## 5. Performance Optimization
### 5.1 Concurrent Upload Support
#### 5.1.1 Parallel Chunk Processing

```python
# app/utils/concurrent_upload.py
import asyncio
from typing import List, Dict, Callable
from app.utils.logging_utils import get_logger

logger = get_logger("concurrent_upload")


class ConcurrentUploadManager:
    """Manage concurrent chunk uploads with rate limiting"""

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def upload_chunks_concurrently(
            self,
            upload_func: Callable,
            upload_id: str,
            chunks_data: List[tuple],  # [(chunk_index, chunk_data), ...]
            progress_callback: Optional[Callable] = None
    ) -> Dict[int, bool]:
        """Upload multiple chunks concurrently"""

        async def upload_single_chunk(chunk_index: int, chunk_data: bytes):
            """Upload a single chunk with semaphore protection"""
            async with self.semaphore:
                try:
                    success = await upload_func(
                        upload_id=upload_id,
                        chunk_index=chunk_index,
                        chunk_data=chunk_data
                    )

                    if progress_callback:
                        await progress_callback(chunk_index, len(chunk_data), success)

                    return chunk_index, success

                except Exception as e:
                    logger.error("Concurrent chunk upload failed",
                                 upload_id=upload_id,
                                 chunk_index=chunk_index,
                                 error_message=str(e))

                    if progress_callback:
                        await progress_callback(chunk_index, len(chunk_data), False)

                    return chunk_index, False

        # Create tasks for all chunks
        tasks = [
            upload_single_chunk(chunk_index, chunk_data)
            for chunk_index, chunk_data in chunks_data
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        upload_results = {}
        for result in results:
            if isinstance(result, tuple):
                chunk_index, success = result
                upload_results[chunk_index] = success
            else:
                logger.error("Unexpected result from concurrent upload",
                             result=str(result))

        logger.info("Concurrent upload completed",
                    upload_id=upload_id,
                    total_chunks=len(chunks_data),
                    successful_chunks=sum(upload_results.values()),
                    failed_chunks=len(chunks_data) - sum(upload_results.values()))

        return upload_results
```

### 5.2 Memory Optimization
#### 5.2.1 Streaming Chunk Processing

```python
# app/utils/streaming_upload.py
import tempfile
import os
from typing import AsyncIterator
from app.utils.logging_utils import get_logger

logger = get_logger("streaming_upload")


class StreamingChunkProcessor:
    """Process large files in chunks without loading entire file into memory"""

    def __init__(self, chunk_size: int = 5 * 1024 * 1024):
        self.chunk_size = chunk_size

    async def process_file_stream(
            self,
            file_stream: AsyncIterator[bytes],
            total_size: int
    ) -> AsyncIterator[tuple[int, bytes]]:
        """Process file stream and yield chunks"""

        chunk_index = 0
        current_chunk = b""
        bytes_processed = 0

        async for data in file_stream:
            current_chunk += data
            bytes_processed += len(data)

            # Check if we have a complete chunk
            while len(current_chunk) >= self.chunk_size:
                # Extract chunk
                chunk_data = current_chunk[:self.chunk_size]
                current_chunk = current_chunk[self.chunk_size:]

                yield chunk_index, chunk_data
                chunk_index += 1

                logger.debug("Chunk processed",
                             chunk_index=chunk_index - 1,
                             chunk_size=len(chunk_data),
                             bytes_processed=bytes_processed,
                             total_size=total_size)

        # Process remaining data as final chunk
        if current_chunk:
            yield chunk_index, current_chunk

            logger.debug("Final chunk processed",
                         chunk_index=chunk_index,
                         chunk_size=len(current_chunk),
                         bytes_processed=bytes_processed,
                         total_size=total_size)

    async def create_temp_chunks(
            self,
            file_data: bytes,
            chunk_size: int
    ) -> List[str]:
        """Create temporary files for large chunks to reduce memory usage"""

        temp_files = []

        try:
            for i in range(0, len(file_data), chunk_size):
                chunk_data = file_data[i:i + chunk_size]

                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(chunk_data)
                    temp_files.append(temp_file.name)

                    logger.debug("Temporary chunk file created",
                                 chunk_index=len(temp_files) - 1,
                                 temp_file=temp_file.name,
                                 chunk_size=len(chunk_data))

            return temp_files

        except Exception as e:
            # Cleanup temp files on error
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            raise

    def cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files"""

        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
                logger.debug("Temporary file cleaned up", temp_file=temp_file)
            except Exception as e:
                logger.warning("Failed to cleanup temporary file",
                               temp_file=temp_file,
                               error_message=str(e))
```