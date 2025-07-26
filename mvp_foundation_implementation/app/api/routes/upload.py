from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from app.services.upload_service import upload_service
from app.utils.upload_validation import file_validator
from app.utils.logging import get_logger, log_performance

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