import uuid
from datetime import datetime, timedelta
from typing import Optional, List
from app.services.firestore import FirestoreService
from app.models.upload_session import UploadSession, ChunkInfo, UploadStatus
from app.utils.logging import get_logger, log_performance
from app.config.settings import get_settings
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
            
            logger.info("Upload completed successfully",
                       upload_id=upload_id,
                       audio_id=session.audio_id,
                       total_chunks=session.total_chunks)
            
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