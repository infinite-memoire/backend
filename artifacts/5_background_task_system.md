# Background Task System Implementation

Based on API endpoint design and FastAPI background task requirements for audio processing pipeline.

## System Overview
FastAPI Background Tasks for MVP implementation:
- No Celery/Redis complexity for MVP
- Async processing for long-running operations
- Simple task coordination and status tracking
- Background processing for audio upload and AI pipeline

## 1. Background Task Architecture

### 1.1 Task Categories
```python
from enum import Enum

class TaskType(Enum):
    AUDIO_UPLOAD_PROCESSING = "audio_upload_processing"
    TRANSCRIPTION = "transcription"
    SEMANTIC_CHUNKING = "semantic_chunking"
    GRAPH_GENERATION = "graph_generation"
    CHAPTER_ANALYSIS = "chapter_analysis"
    QUESTION_GENERATION = "question_generation"
```

### 1.2 Task Status Tracking
```python
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskInfo:
    task_id: str
    task_type: TaskType
    status: TaskStatus
    progress_percentage: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## 2. Task Management Service

### 2.1 Core Task Manager
```python
import asyncio
import uuid
from typing import Dict, Callable, Any
from app.services.firestore import FirestoreService
import logging

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    def __init__(self):
        self.firestore_service = FirestoreService()
        self.active_tasks: Dict[str, TaskInfo] = {}
        
    async def create_task(
        self, 
        task_type: TaskType, 
        task_function: Callable,
        task_args: tuple = (),
        task_kwargs: dict = None,
        metadata: dict = None
    ) -> str:
        """Create and start a background task"""
        task_id = str(uuid.uuid4())
        task_kwargs = task_kwargs or {}
        metadata = metadata or {}
        
        task_info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            metadata=metadata
        )
        
        # Store task info in memory and Firestore
        self.active_tasks[task_id] = task_info
        await self._persist_task_status(task_info)
        
        # Start the background task
        asyncio.create_task(
            self._execute_task(task_id, task_function, task_args, task_kwargs)
        )
        
        logger.info(f"Created background task {task_id} of type {task_type}")
        return task_id
        
    async def _execute_task(
        self, 
        task_id: str, 
        task_function: Callable,
        task_args: tuple,
        task_kwargs: dict
    ):
        """Execute the background task with error handling"""
        task_info = self.active_tasks[task_id]
        
        try:
            # Update status to running
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = datetime.utcnow()
            await self._persist_task_status(task_info)
            
            # Execute the task function
            logger.info(f"Starting execution of task {task_id}")
            result = await task_function(task_id, *task_args, **task_kwargs)
            
            # Update status to completed
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = datetime.utcnow()
            task_info.progress_percentage = 100
            task_info.result_data = result
            
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            # Update status to failed
            task_info.status = TaskStatus.FAILED
            task_info.completed_at = datetime.utcnow()
            task_info.error_message = str(e)
            
            logger.error(f"Task {task_id} failed: {str(e)}")
            
        finally:
            await self._persist_task_status(task_info)
            
    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get current task status"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
            
        # Check Firestore for completed/old tasks
        return await self._load_task_status(task_id)
        
    async def update_task_progress(self, task_id: str, progress: int, message: str = None):
        """Update task progress from within the task"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].progress_percentage = progress
            if message:
                self.active_tasks[task_id].metadata['current_step'] = message
            await self._persist_task_status(self.active_tasks[task_id])
            
    async def _persist_task_status(self, task_info: TaskInfo):
        """Save task status to Firestore"""
        task_data = {
            "task_id": task_info.task_id,
            "task_type": task_info.task_type.value,
            "status": task_info.status.value,
            "progress_percentage": task_info.progress_percentage,
            "started_at": task_info.started_at,
            "completed_at": task_info.completed_at,
            "error_message": task_info.error_message,
            "result_data": task_info.result_data,
            "metadata": task_info.metadata,
            "updated_at": datetime.utcnow()
        }
        
        await self.firestore_service.upsert_document(
            collection="processing_tasks",
            document_id=task_info.task_id,
            data=task_data
        )
        
    async def _load_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Load task status from Firestore"""
        task_doc = await self.firestore_service.get_document(
            collection="processing_tasks",
            document_id=task_id
        )
        
        if not task_doc:
            return None
            
        return TaskInfo(
            task_id=task_doc["task_id"],
            task_type=TaskType(task_doc["task_type"]),
            status=TaskStatus(task_doc["status"]),
            progress_percentage=task_doc["progress_percentage"],
            started_at=task_doc.get("started_at"),
            completed_at=task_doc.get("completed_at"),
            error_message=task_doc.get("error_message"),
            result_data=task_doc.get("result_data"),
            metadata=task_doc.get("metadata", {})
        )

# Global task manager instance
task_manager = BackgroundTaskManager()
```

## 3. Audio Processing Tasks

### 3.1 Audio Upload Processing Task
```python
async def process_audio_upload(
    task_id: str,
    audio_id: str,
    uploaded_chunks: list,
    audio_metadata: dict
) -> dict:
    """Process uploaded audio file chunks"""
    
    try:
        await task_manager.update_task_progress(task_id, 10, "Validating upload")
        
        # Validate all chunks are present
        total_chunks = audio_metadata.get("chunk_count", 1)
        if len(uploaded_chunks) != total_chunks:
            raise ValueError(f"Missing chunks: expected {total_chunks}, got {len(uploaded_chunks)}")
            
        await task_manager.update_task_progress(task_id, 30, "Reassembling file")
        
        # Reassemble chunks into complete file
        complete_audio_data = b""
        for chunk_index in range(total_chunks):
            chunk_data = await firestore_service.get_chunk_data(audio_id, chunk_index)
            complete_audio_data += chunk_data
            
        await task_manager.update_task_progress(task_id, 60, "Storing complete file")
        
        # Store complete file in Firestore
        await firestore_service.store_complete_audio(audio_id, complete_audio_data)
        
        await task_manager.update_task_progress(task_id, 80, "Updating metadata")
        
        # Update audio record status
        await firestore_service.update_audio_record(audio_id, {
            "upload_status": "completed",
            "file_size_bytes": len(complete_audio_data),
            "processing_stage": "upload_completed",
            "completed_at": datetime.utcnow()
        })
        
        await task_manager.update_task_progress(task_id, 100, "Upload completed")
        
        return {
            "audio_id": audio_id,
            "file_size_bytes": len(complete_audio_data),
            "status": "completed"
        }
        
    except Exception as e:
        # Update audio record with error status
        await firestore_service.update_audio_record(audio_id, {
            "upload_status": "failed",
            "error_message": str(e),
            "failed_at": datetime.utcnow()
        })
        raise
```

### 3.2 Transcription Task
```python
async def process_transcription(
    task_id: str,
    audio_id: str,
    transcription_options: dict = None
) -> dict:
    """Process audio transcription using STT model"""
    
    transcription_options = transcription_options or {}
    
    try:
        await task_manager.update_task_progress(task_id, 5, "Loading audio file")
        
        # Load audio file from Firestore
        audio_data = await firestore_service.get_complete_audio(audio_id)
        if not audio_data:
            raise ValueError(f"Audio file {audio_id} not found")
            
        await task_manager.update_task_progress(task_id, 15, "Initializing STT model")
        
        # Initialize STT service (placeholder for actual STT implementation)
        from app.services.stt_service import STTService
        stt_service = STTService()
        
        await task_manager.update_task_progress(task_id, 25, "Starting transcription")
        
        # Perform transcription
        transcription_result = await stt_service.transcribe_audio(
            audio_data,
            progress_callback=lambda p: asyncio.create_task(
                task_manager.update_task_progress(task_id, 25 + int(p * 0.6), f"Transcribing... {int(p)}%")
            )
        )
        
        await task_manager.update_task_progress(task_id, 90, "Saving transcript")
        
        # Create transcript record
        transcript_data = {
            "audio_file_id": audio_id,
            "raw_text": transcription_result["text"],
            "language_detected": transcription_result.get("language", "en-US"),
            "confidence_score": transcription_result.get("confidence", 0.0),
            "word_timestamps": transcription_result.get("word_timestamps", []),
            "model_used": transcription_result.get("model", "unknown"),
            "processing_time_seconds": transcription_result.get("processing_time", 0),
            "created_at": datetime.utcnow()
        }
        
        transcript_id = await firestore_service.create_transcript_record(transcript_data)
        
        # Update audio record
        await firestore_service.update_audio_record(audio_id, {
            "processing_stage": "transcription_completed",
            "transcript_id": transcript_id,
            "transcription_completed_at": datetime.utcnow()
        })
        
        await task_manager.update_task_progress(task_id, 100, "Transcription completed")
        
        return {
            "audio_id": audio_id,
            "transcript_id": transcript_id,
            "word_count": len(transcription_result["text"].split()),
            "confidence_score": transcription_result.get("confidence", 0.0)
        }
        
    except Exception as e:
        await firestore_service.update_audio_record(audio_id, {
            "processing_stage": "transcription_failed",
            "error_message": str(e),
            "failed_at": datetime.utcnow()
        })
        raise
```

### 3.3 Semantic Chunking Task
```python
async def process_semantic_chunking(
    task_id: str,
    transcript_id: str,
    chunking_options: dict = None
) -> dict:
    """Process transcript into semantic chunks"""
    
    chunking_options = chunking_options or {}
    
    try:
        await task_manager.update_task_progress(task_id, 10, "Loading transcript")
        
        # Load transcript
        transcript = await firestore_service.get_transcript_record(transcript_id)
        if not transcript:
            raise ValueError(f"Transcript {transcript_id} not found")
            
        await task_manager.update_task_progress(task_id, 25, "Analyzing text structure")
        
        # Perform semantic chunking (placeholder for actual NLP implementation)
        from app.services.nlp_service import NLPService
        nlp_service = NLPService()
        
        chunks = await nlp_service.create_semantic_chunks(
            text=transcript["raw_text"],
            word_timestamps=transcript.get("word_timestamps", []),
            chunk_size=chunking_options.get("chunk_size", 150),
            overlap=chunking_options.get("overlap", 25)
        )
        
        await task_manager.update_task_progress(task_id, 60, "Creating chunk records")
        
        # Create chunk records in Firestore
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "transcript_id": transcript_id,
                "audio_file_id": transcript["audio_file_id"],
                "chunk_index": i,
                "text_content": chunk["text"],
                "word_count": len(chunk["text"].split()),
                "character_count": len(chunk["text"]),
                "start_word_index": chunk.get("start_word_index", 0),
                "end_word_index": chunk.get("end_word_index", 0),
                "temporal_info": {
                    "temporal_marker": None,
                    "temporal_confidence": 0.0,
                    "requires_user_clarification": True
                },
                "created_at": datetime.utcnow()
            }
            
            chunk_id = await firestore_service.create_text_chunk(chunk_data)
            chunk_ids.append(chunk_id)
            
            # Update progress
            progress = 60 + int((i / len(chunks)) * 30)
            await task_manager.update_task_progress(task_id, progress, f"Created chunk {i+1}/{len(chunks)}")
            
        await task_manager.update_task_progress(task_id, 95, "Updating transcript record")
        
        # Update transcript with chunk references
        await firestore_service.update_transcript_record(transcript_id, {
            "semantic_chunks": [{"chunk_id": cid} for cid in chunk_ids],
            "chunking_completed_at": datetime.utcnow(),
            "processing_stage": "semantic_chunking_completed"
        })
        
        await task_manager.update_task_progress(task_id, 100, "Semantic chunking completed")
        
        return {
            "transcript_id": transcript_id,
            "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids
        }
        
    except Exception as e:
        await firestore_service.update_transcript_record(transcript_id, {
            "processing_stage": "semantic_chunking_failed",
            "error_message": str(e),
            "failed_at": datetime.utcnow()
        })
        raise
```

## 4. Task Coordination

### 4.1 Pipeline Orchestration
```python
async def process_audio_pipeline(
    audio_id: str,
    include_stages: list = None
) -> dict:
    """Orchestrate the complete audio processing pipeline"""
    
    include_stages = include_stages or [
        "upload_processing",
        "transcription", 
        "semantic_chunking",
        "graph_generation"
    ]
    
    results = {}
    
    try:
        # Stage 1: Upload Processing
        if "upload_processing" in include_stages:
            upload_task_id = await task_manager.create_task(
                TaskType.AUDIO_UPLOAD_PROCESSING,
                process_audio_upload,
                task_args=(audio_id,),
                metadata={"audio_id": audio_id, "stage": "upload_processing"}
            )
            
            # Wait for upload completion
            upload_result = await _wait_for_task_completion(upload_task_id)
            results["upload"] = upload_result
            
        # Stage 2: Transcription
        if "transcription" in include_stages:
            transcription_task_id = await task_manager.create_task(
                TaskType.TRANSCRIPTION,
                process_transcription,
                task_args=(audio_id,),
                metadata={"audio_id": audio_id, "stage": "transcription"}
            )
            
            transcription_result = await _wait_for_task_completion(transcription_task_id)
            results["transcription"] = transcription_result
            
        # Stage 3: Semantic Chunking
        if "semantic_chunking" in include_stages and "transcription" in results:
            chunking_task_id = await task_manager.create_task(
                TaskType.SEMANTIC_CHUNKING,
                process_semantic_chunking,
                task_args=(results["transcription"]["transcript_id"],),
                metadata={"audio_id": audio_id, "stage": "semantic_chunking"}
            )
            
            chunking_result = await _wait_for_task_completion(chunking_task_id)
            results["semantic_chunking"] = chunking_result
            
        return results
        
    except Exception as e:
        logger.error(f"Pipeline processing failed for audio {audio_id}: {str(e)}")
        raise

async def _wait_for_task_completion(task_id: str, timeout_seconds: int = 3600) -> dict:
    """Wait for task completion with timeout"""
    start_time = datetime.utcnow()
    
    while True:
        task_info = await task_manager.get_task_status(task_id)
        
        if task_info.status == TaskStatus.COMPLETED:
            return task_info.result_data
        elif task_info.status == TaskStatus.FAILED:
            raise Exception(f"Task {task_id} failed: {task_info.error_message}")
        elif task_info.status == TaskStatus.CANCELLED:
            raise Exception(f"Task {task_id} was cancelled")
            
        # Check timeout
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Task {task_id} timed out after {timeout_seconds} seconds")
            
        # Wait before checking again
        await asyncio.sleep(5)
```

## 5. Integration with FastAPI Routes

### 5.1 Route Integration Example
```python
from fastapi import BackgroundTasks, HTTPException
from app.services.background_tasks import task_manager, process_audio_pipeline

@router.post("/audio/process")
async def start_audio_processing(
    audio_id: str,
    background_tasks: BackgroundTasks,
    stages: list = None
):
    """Start audio processing pipeline"""
    
    # Validate audio exists
    audio_record = await firestore_service.get_audio_record(audio_id)
    if not audio_record:
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    # Create pipeline task
    pipeline_task_id = await task_manager.create_task(
        TaskType.AUDIO_UPLOAD_PROCESSING,  # Will be expanded to full pipeline
        process_audio_pipeline,
        task_args=(audio_id,),
        task_kwargs={"include_stages": stages},
        metadata={"audio_id": audio_id, "pipeline": True}
    )
    
    return {
        "task_id": pipeline_task_id,
        "audio_id": audio_id,
        "status": "processing_started",
        "estimated_duration_minutes": 15
    }

@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get task processing status"""
    
    task_info = await task_manager.get_task_status(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return {
        "task_id": task_id,
        "status": task_info.status.value,
        "progress_percentage": task_info.progress_percentage,
        "started_at": task_info.started_at,
        "completed_at": task_info.completed_at,
        "error_message": task_info.error_message,
        "current_step": task_info.metadata.get("current_step"),
        "metadata": task_info.metadata
    }
```

## 6. Error Handling and Recovery

### 6.1 Task Retry Logic
```python
async def retry_failed_task(task_id: str, max_retries: int = 3) -> str:
    """Retry a failed task with exponential backoff"""
    
    original_task = await task_manager.get_task_status(task_id)
    if not original_task or original_task.status != TaskStatus.FAILED:
        raise ValueError("Task not found or not in failed state")
        
    retry_count = original_task.metadata.get("retry_count", 0)
    if retry_count >= max_retries:
        raise Exception(f"Task has exceeded maximum retries ({max_retries})")
        
    # Create new task with retry metadata
    new_metadata = original_task.metadata.copy()
    new_metadata["retry_count"] = retry_count + 1
    new_metadata["original_task_id"] = task_id
    
    # Add exponential backoff delay
    delay_seconds = 2 ** retry_count
    await asyncio.sleep(delay_seconds)
    
    # Start retry task
    # Implementation depends on recreating the original task function and args
    # This would need to be stored in the task metadata for proper retry support
    
    return "retry_task_id"
```

## 7. Monitoring and Logging

### 7.1 Task Metrics Collection
```python
class TaskMetrics:
    def __init__(self):
        self.task_counters = {
            "total_started": 0,
            "total_completed": 0,
            "total_failed": 0,
            "by_type": {}
        }
        
    async def record_task_start(self, task_type: TaskType):
        self.task_counters["total_started"] += 1
        type_name = task_type.value
        if type_name not in self.task_counters["by_type"]:
            self.task_counters["by_type"][type_name] = {"started": 0, "completed": 0, "failed": 0}
        self.task_counters["by_type"][type_name]["started"] += 1
        
    async def record_task_completion(self, task_type: TaskType, success: bool):
        if success:
            self.task_counters["total_completed"] += 1
            self.task_counters["by_type"][task_type.value]["completed"] += 1
        else:
            self.task_counters["total_failed"] += 1
            self.task_counters["by_type"][task_type.value]["failed"] += 1

# Global metrics instance
task_metrics = TaskMetrics()
```