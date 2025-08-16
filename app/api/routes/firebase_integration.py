# Firebase Integration Endpoint for AI Processing
# Integrates Firebase Storage audio with existing AI agentic system

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage
from urllib.parse import urlparse, parse_qs

# Import existing services from the backend architecture
from app.services.stt.whisper_service import WhisperSTTService
from app.services.orchestrator import orchestrator
from app.services.firestore import firestore_service
from app.utils.logging_utils import get_logger
from app.config.settings_config import get_settings

# Configure logging
logger = get_logger(__name__)

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
        self.orchestrator = orchestrator  # Use global orchestrator instance
        self.firestore_service = firestore_service  # Use existing Firestore service
        self.temp_dir = Path(tempfile.gettempdir()) / "infinite_memoire_audio"
        self.temp_dir.mkdir(exist_ok=True)
        self.settings = get_settings()
        
        # Initialize Firebase Admin SDK if not already initialized
        self._init_firebase_admin()
    
    def _init_firebase_admin(self):
        """Initialize Firebase Admin SDK with proper credentials"""
        try:
            # Check if Firebase Admin is already initialized
            firebase_admin.get_app()
            logger.info("Firebase Admin SDK already initialized")
        except ValueError:
            # Initialize Firebase Admin SDK
            if self.settings.database.firestore_credentials_path:
                cred = credentials.Certificate(self.settings.database.firestore_credentials_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': f'{self.settings.database.firestore_project_id}.appspot.com'
                })
                logger.info("Firebase Admin SDK initialized with service account")
            else:
                logger.warning("No Firebase credentials found, using default credentials")
                firebase_admin.initialize_app()
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Firebase authentication not configured properly"
            )
    
    def _extract_storage_path(self, firebase_url: str) -> str:
        """Extract the storage path from a Firebase Storage URL"""
        try:
            parsed_url = urlparse(firebase_url)
            # Firebase Storage URLs have format: /v0/b/{bucket}/o/{path}?{params}
            path_parts = parsed_url.path.split('/o/')
            if len(path_parts) != 2:
                raise ValueError("Invalid Firebase Storage URL format")
            
            # Decode the path (Firebase encodes paths)
            from urllib.parse import unquote
            storage_path = unquote(path_parts[1].split('?')[0])
            return storage_path
        except Exception as e:
            logger.error(f"Failed to extract storage path from URL {firebase_url}: {str(e)}")
            raise ValueError(f"Invalid Firebase Storage URL: {firebase_url}")
    
    async def download_audio_files(self, audio_urls: List[str]) -> List[Path]:
        """Download audio files from Firebase Storage with proper authentication"""
        temp_files = []
        
        try:
            # Get Firebase Storage bucket
            bucket = storage.bucket()
            
            for i, url in enumerate(audio_urls):
                logger.info(f"Downloading audio file {i+1}/{len(audio_urls)} from Firebase Storage")
                
                # Extract storage path from Firebase URL
                storage_path = self._extract_storage_path(url)
                logger.info(f"Storage path: {storage_path}")
                
                # Get blob reference with proper authentication
                blob = bucket.blob(storage_path)
                
                # Check if blob exists
                if not blob.exists():
                    raise FileNotFoundError(f"Audio file not found in Firebase Storage: {storage_path}")
                
                # Download blob content with authentication
                audio_content = blob.download_as_bytes()
                
                # Save to temporary file with unique name
                temp_file = self.temp_dir / f"audio_{uuid.uuid4().hex}_{i}.wav"
                temp_file.write_bytes(audio_content)
                temp_files.append(temp_file)
                
                logger.info(f"Downloaded {len(audio_content)} bytes to {temp_file}")
                
                # Record download in Firestore for audit trail
                await self.firestore_service.create_audio_record({
                    'storage_path': storage_path,
                    'download_timestamp': datetime.utcnow().isoformat(),
                    'file_size_bytes': len(audio_content),
                    'temp_file_path': str(temp_file),
                    'status': 'downloaded'
                })
        
        except Exception as e:
            # Cleanup partial downloads
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            
            logger.error(f"Failed to download audio from Firebase Storage: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download audio from Firebase Storage: {str(e)}"
            )
        
        return temp_files
    
    async def transcribe_audio_files(self, audio_files: List[Path]) -> str:
        """Transcribe audio files using existing STT service and record in Firestore"""
        combined_transcript = ""
        transcript_records = []
        
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
                        
                        # Record transcription in Firestore
                        transcript_record = {
                            'audio_file_path': str(audio_file),
                            'transcript_text': transcript_text,
                            'transcription_timestamp': datetime.utcnow().isoformat(),
                            'character_count': len(transcript_text),
                            'confidence_score': transcription_result.get('confidence', None),
                            'language': transcription_result.get('language', 'unknown'),
                            'processing_time_seconds': transcription_result.get('processing_time', None)
                        }
                        
                        transcript_id = await self.firestore_service.create_transcript_record(transcript_record)
                        transcript_records.append(transcript_id)
                        logger.info(f"Saved transcript record: {transcript_id}")
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
        
        # Store combined transcript metadata
        if transcript_records:
            await self.firestore_service.create_document(
                collection="combined_transcripts",
                document_id=str(uuid.uuid4()),
                data={
                    'individual_transcripts': transcript_records,
                    'combined_transcript': combined_transcript.strip(),
                    'total_character_count': len(combined_transcript.strip()),
                    'file_count': len(audio_files),
                    'creation_timestamp': datetime.utcnow().isoformat()
                }
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
        success = await processor.orchestrator.start_processing_from_session(
            session_id=session_id,
            transcript=combined_transcript,
            user_id="firebase_user",  # Default user ID for Firebase integration
            user_preferences=user_preferences
        )
        
        if not success:
            raise Exception(f"Failed to start processing for session {session_id}")
        
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
        
        # Test Firebase Storage connection
        try:
            bucket = storage.bucket()
            # Try to list blobs (this tests connectivity without downloading)
            blobs = list(bucket.list_blobs(max_results=1))
            firebase_storage_status = "available"
        except Exception as e:
            logger.warning(f"Firebase Storage test failed: {str(e)}")
            firebase_storage_status = f"unavailable: {str(e)}"
        
        # Test Firestore connection
        try:
            await processor.firestore_service.test_connection()
            firestore_status = "available"
        except Exception as e:
            logger.warning(f"Firestore test failed: {str(e)}")
            firestore_status = f"unavailable: {str(e)}"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "stt_service": "available",
                "agent_orchestrator": "available",
                "temp_storage": "accessible",
                "firebase_storage": firebase_storage_status,
                "firestore": firestore_status
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }