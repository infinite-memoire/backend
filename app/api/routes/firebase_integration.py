# Firebase Integration Endpoint for AI Processing
# Integrates Firebase Storage audio with existing AI agentic system

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import requests
import tempfile
import uuid
import logging
from pathlib import Path
from datetime import datetime

# Import existing services from the backend architecture
from app.services.stt.whisper_service import WhisperSTTService
from app.services.orchestrator import AgentOrchestrator

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["firebase-integration"])

# Request/Response models
class ProcessFromFirebaseRequest(BaseModel):
    audioUrls: List[str] = Field(..., min_items=1, max_items=10)
    chapterTitle: str = Field(..., min_length=1, max_length=200)
    chapterDescription: str = Field(..., min_length=1, max_length=1000)
    userPreferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('audioUrls')
    def validate_firebase_urls(cls, v):
        """Validate that URLs are from Firebase Storage"""
        for url in v:
            if not url.startswith('https://firebasestorage.googleapis.com/'):
                raise ValueError(f'Invalid Firebase Storage URL: {url}')
        return v

class ProcessFromFirebaseResponse(BaseModel):
    sessionId: str
    status: str = "processing"
    message: str
    estimatedDuration: Optional[str] = None

# Audio processing service
class FirebaseAudioProcessor:
    def __init__(self):
        self.stt_service = WhisperSTTService()
        self.orchestrator = AgentOrchestrator()
        self.temp_dir = Path(tempfile.gettempdir()) / "infinite_memoire_audio"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def download_audio_files(self, audio_urls: List[str]) -> List[Path]:
        """Download audio files from Firebase URLs with error handling"""
        temp_files = []
        
        try:
            for i, url in enumerate(audio_urls):
                logger.info(f"Downloading audio file {i+1}/{len(audio_urls)} from Firebase")
                
                # Download with timeout and proper headers
                response = requests.get(
                    url, 
                    timeout=60,
                    headers={'User-Agent': 'InfiniteMemoire/1.0'}
                )
                response.raise_for_status()
                
                # Save to temporary file with unique name
                temp_file = self.temp_dir / f"audio_{uuid.uuid4().hex}_{i}.wav"
                temp_file.write_bytes(response.content)
                temp_files.append(temp_file)
                
                logger.info(f"Downloaded {len(response.content)} bytes to {temp_file}")
        
        except requests.RequestException as e:
            # Cleanup partial downloads
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download audio from Firebase: {str(e)}"
            )
        
        return temp_files
    
    async def transcribe_audio_files(self, audio_files: List[Path]) -> str:
        """Transcribe audio files using existing STT service"""
        combined_transcript = ""
        
        try:
            for i, audio_file in enumerate(audio_files):
                logger.info(f"Transcribing audio file {i+1}/{len(audio_files)}: {audio_file}")
                
                # Use existing STT service interface
                transcription_result = await self.stt_service.transcribe_audio_file(str(audio_file))
                
                if transcription_result and "text" in transcription_result:
                    transcript_text = transcription_result["text"].strip()
                    if transcript_text:
                        combined_transcript += f"[Recording {i+1}]\n{transcript_text}\n\n"
                        logger.info(f"Transcribed {len(transcript_text)} characters")
                else:
                    logger.warning(f"No transcription result for file {audio_file}")
        
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Audio transcription failed: {str(e)}"
            )
        
        if not combined_transcript.strip():
            raise HTTPException(
                status_code=400,
                detail="No transcribable audio content found in the provided files"
            )
        
        return combined_transcript.strip()
    
    def cleanup_temp_files(self, temp_files: List[Path]):
        """Clean up temporary audio files"""
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {str(e)}")

# Background processing task
async def process_audio_background(
    processor: FirebaseAudioProcessor,
    session_id: str,
    audio_urls: List[str],
    chapter_title: str,
    chapter_description: str,
    user_preferences: Dict[str, Any]
):
    """Background task for processing audio through AI pipeline"""
    temp_files = []
    
    try:
        # Step 1: Download audio files from Firebase
        temp_files = await processor.download_audio_files(audio_urls)
        
        # Step 2: Transcribe using WhisperSTTService
        combined_transcript = await processor.transcribe_audio_files(temp_files)
        
        # Step 3: Process through AgentOrchestrator (existing AI pipeline)
        await processor.orchestrator.start_processing(
            session_id=session_id,
            transcript=combined_transcript,
            metadata={
                "chapter_title": chapter_title,
                "chapter_description": chapter_description,
                "audio_urls": audio_urls,
                "user_preferences": user_preferences,
                "processing_type": "firebase_audio_to_book",
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Successfully started AI processing for session {session_id}")
    
    except Exception as e:
        logger.error(f"Background processing failed for session {session_id}: {str(e)}")
        # Update session status to failed (assuming orchestrator handles this)
        await processor.orchestrator.mark_session_failed(session_id, str(e))
    
    finally:
        # Always cleanup temporary files
        processor.cleanup_temp_files(temp_files)

# Main endpoint
@router.post("/process-from-firebase", response_model=ProcessFromFirebaseResponse)
async def process_from_firebase(
    request: ProcessFromFirebaseRequest,
    background_tasks: BackgroundTasks
):
    """
    Process audio files from Firebase Storage URLs through the AI agentic system.
    
    This endpoint implements the minimal polling approach:
    1. Downloads audio files from Firebase URLs
    2. Transcribes using existing WhisperSTTService  
    3. Processes through existing AgentOrchestrator
    4. Returns session ID for polling via existing /ai/status/{session_id}
    """
    try:
        # Initialize processor
        processor = FirebaseAudioProcessor()
        
        # Create processing session using existing infrastructure
        session_id = str(uuid.uuid4())
        
        # Create session with orchestrator
        await processor.orchestrator.create_processing_session(
            session_id=session_id,
            initial_status="initializing",
            metadata={
                "chapter_title": request.chapterTitle,
                "chapter_description": request.chapterDescription,
                "audio_urls": request.audioUrls,
                "user_preferences": request.userPreferences,
                "processing_type": "firebase_audio_to_book",
                "estimated_duration": "5-10 minutes"
            }
        )
        
        # Start background processing
        background_tasks.add_task(
            process_audio_background,
            processor,
            session_id,
            request.audioUrls,
            request.chapterTitle,
            request.chapterDescription,
            request.userPreferences
        )
        
        logger.info(f"Created processing session {session_id} for {len(request.audioUrls)} audio files")
        
        return ProcessFromFirebaseResponse(
            sessionId=session_id,
            status="processing",
            message="Audio processing started. Use /ai/status/{session_id} to monitor progress.",
            estimatedDuration="5-10 minutes"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in process_from_firebase: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during audio processing setup: {str(e)}"
        )

# Health check endpoint
@router.get("/firebase-integration/health")
async def firebase_integration_health():
    """Health check for Firebase integration functionality"""
    try:
        # Basic service checks
        processor = FirebaseAudioProcessor()
        
        # Check if temp directory is accessible
        if not processor.temp_dir.exists():
            raise Exception("Temporary directory not accessible")
        
        # Check if services can be instantiated
        if not processor.stt_service:
            raise Exception("STT service not available")
            
        if not processor.orchestrator:
            raise Exception("Agent orchestrator not available")
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "stt_service": "available",
                "agent_orchestrator": "available",
                "temp_storage": "accessible"
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }