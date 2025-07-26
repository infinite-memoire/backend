# FastAPI Project Structure Draft

Based on MVP tech stack requirements for audio-to-book processing backend.

## Project Root Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py         # Configuration management
│   │   └── logging.py          # Logging configuration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── audio.py            # Audio file models
│   │   ├── transcript.py       # Transcript models
│   │   └── graph.py            # Graph node models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── firestore.py        # Firestore database service
│   │   ├── neo4j.py            # Neo4j graph database service
│   │   ├── audio_processing.py # Audio upload and processing
│   │   └── background_tasks.py # FastAPI background task handlers
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── audio.py        # Audio upload/management endpoints
│   │   │   ├── transcripts.py  # Transcript management endpoints
│   │   │   └── health.py       # Health check endpoints
│   │   └── dependencies.py     # FastAPI dependencies
│   └── utils/
│       ├── __init__.py
│       ├── chunked_upload.py   # Chunked file upload utilities
│       └── validators.py       # Input validation utilities
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Local development setup
├── config.yaml                 # Application configuration
└── README.md                   # Setup and usage instructions
```

## Core Application Components

### 1. FastAPI Application (main.py)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import get_settings
from app.config.app_logging import setup_logging
from app.api.routes import audio, transcripts, health


# Application factory pattern
def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging()

    app = FastAPI(
        title="Memoire Backend API",
        description="Audio-to-book processing backend",
        version="1.0.0"
    )

    # CORS configuration (permissive for MVP)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(audio.router, prefix="/api/v1/audio")
    app.include_router(transcripts.router, prefix="/api/v1/transcripts")
    app.include_router(health.router, prefix="/api/v1/health")

    return app


app = create_app()
```

### 2. Configuration Management (config/settings.py)
```python
from pydantic import BaseSettings
from typing import Optional
import yaml

class Settings(BaseSettings):
    # Application settings
    debug: bool = True
    log_level: str = "INFO"
    
    # Database settings
    firestore_project_id: str
    firestore_credentials_path: Optional[str] = None
    
    # Neo4j settings
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    
    # File upload settings
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    chunk_size: int = 1024 * 1024  # 1MB chunks
    
    class Config:
        env_file = ".env"
        case_sensitive = False

def get_settings() -> Settings:
    return Settings()
```

### 3. Database Services

#### Firestore Service (services/firestore.py)
```python
from google.cloud import firestore
from app.config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

class FirestoreService:
    def __init__(self):
        settings = get_settings()
        self.db = firestore.Client(project=settings.firestore_project_id)
        
    async def create_audio_record(self, audio_data: dict) -> str:
        doc_ref = self.db.collection('audio_files').document()
        doc_ref.set(audio_data)
        logger.info(f"Created audio record: {doc_ref.id}")
        return doc_ref.id
        
    async def create_transcript_record(self, transcript_data: dict) -> str:
        doc_ref = self.db.collection('transcripts').document()
        doc_ref.set(transcript_data)
        logger.info(f"Created transcript record: {doc_ref.id}")
        return doc_ref.id
        
    async def get_audio_record(self, audio_id: str) -> dict:
        doc_ref = self.db.collection('audio_files').document(audio_id)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None
```

#### Neo4j Service (services/neo4j.py)
```python
from neo4j import GraphDatabase
from app.config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

class Neo4jService:
    def __init__(self):
        settings = get_settings()
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
    def close(self):
        self.driver.close()
        
    async def create_story_node(self, node_data: dict) -> str:
        with self.driver.session() as session:
            result = session.write_transaction(self._create_node, node_data)
            logger.info(f"Created story node: {result}")
            return result
            
    @staticmethod
    def _create_node(tx, node_data):
        query = """
        CREATE (n:StoryNode {
            id: $id,
            summary: $summary,
            temporal_marker: $temporal_marker,
            chunk_ids: $chunk_ids
        })
        RETURN n.id as node_id
        """
        result = tx.run(query, **node_data)
        return result.single()["node_id"]
```

### 4. API Endpoints

#### Audio Routes (api/routes/audio.py)
```python
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.services.firestore import FirestoreService
from app.services.background_tasks import process_audio_upload
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    # Validate file type
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be audio format")
    
    # Create audio record in Firestore
    firestore_service = FirestoreService()
    audio_data = {
        "filename": file.filename,
        "content_type": file.content_type,
        "upload_status": "uploading",
        "created_at": datetime.utcnow()
    }
    
    audio_id = await firestore_service.create_audio_record(audio_data)
    
    # Queue background processing
    background_tasks.add_task(process_audio_upload, audio_id, file)
    
    return {
        "audio_id": audio_id,
        "status": "upload_initiated",
        "message": "Audio file upload started"
    }

@router.get("/{audio_id}/status")
async def get_audio_status(audio_id: str):
    firestore_service = FirestoreService()
    audio_record = await firestore_service.get_audio_record(audio_id)
    
    if not audio_record:
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    return {
        "audio_id": audio_id,
        "status": audio_record.get("upload_status", "unknown"),
        "processing_stage": audio_record.get("processing_stage", "none")
    }
```

### 5. Background Task Processing (services/background_tasks.py)
```python
from app.services.firestore import FirestoreService
import logging

logger = logging.getLogger(__name__)

async def process_audio_upload(audio_id: str, audio_file):
    """
    Background task to handle audio file processing
    """
    try:
        firestore_service = FirestoreService()
        
        # Update status to processing
        await firestore_service.update_audio_status(audio_id, "processing")
        
        # Store file content in Firestore
        file_content = await audio_file.read()
        await firestore_service.store_audio_content(audio_id, file_content)
        
        # Update status to completed
        await firestore_service.update_audio_status(audio_id, "completed")
        
        logger.info(f"Successfully processed audio file: {audio_id}")
        
    except Exception as e:
        logger.error(f"Error processing audio file {audio_id}: {str(e)}")
        await firestore_service.update_audio_status(audio_id, "failed")
```

### 6. Docker Configuration (Dockerfile)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config.yaml .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7. Dependencies (requirements.txt)
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
google-cloud-firestore==2.13.1
neo4j==5.14.1
pydantic==2.5.0
python-multipart==0.0.6
pyyaml==6.0.1
python-json-logger==2.0.7
```

## Key Design Principles

1. **Separation of Concerns**: Clear separation between API, services, and data models
2. **Dependency Injection**: Services injected into routes for testability
3. **Configuration Management**: Centralized settings with environment variable support
4. **Error Handling**: Comprehensive logging and error reporting
5. **Scalability**: Background task processing for long-running operations
6. **Container Ready**: Docker configuration for cloud deployment
7. **API Documentation**: FastAPI automatic OpenAPI documentation generation