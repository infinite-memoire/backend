from dataclasses import dataclass, field
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
    metadata: Dict = field(default_factory=dict)