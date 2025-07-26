"""
AI Processing API Routes
Handles AI-powered transcript processing, chapter generation, and user interaction
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime

from app.services.orchestrator import AgentOrchestrator
from app.services.firestore import firestore_service
from app.utils.logging import get_logger, log_performance

logger = get_logger("ai_processing_api")
router = APIRouter(tags=["AI Processing"])

# Global orchestrator instance
orchestrator = AgentOrchestrator()

# Request/Response Models
class ProcessTranscriptRequest(BaseModel):
    """Request model for transcript processing"""
    transcript: str = Field(..., description="Raw transcript text to process")
    user_id: str = Field(..., description="User identifier")
    audio_session_id: Optional[str] = Field(None, description="Associated audio session ID")
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing preferences")

class ProcessingStatusResponse(BaseModel):
    """Response model for processing status"""
    session_id: str
    user_id: str
    current_stage: str
    progress_percentage: float
    current_task: str
    start_time: str
    estimated_completion: Optional[str]
    is_completed: bool
    results_preview: Dict[str, Any]
    errors: List[Dict[str, Any]]

class ChapterResponse(BaseModel):
    """Response model for a generated chapter"""
    storyline_id: str
    title: str
    content: str
    word_count: int
    quality_score: float
    participants: List[str]
    themes: List[str]
    confidence: float

class FollowupQuestionResponse(BaseModel):
    """Response model for follow-up questions"""
    id: str
    category: str
    question: str
    context: str
    priority_score: float
    reasoning: str

class ProcessingResultsResponse(BaseModel):
    """Response model for complete processing results"""
    session_id: str
    processing_summary: Dict[str, Any]
    chapters: List[ChapterResponse]
    followup_questions: List[FollowupQuestionResponse]
    question_categories: Dict[str, List[FollowupQuestionResponse]]
    storylines: List[Dict[str, Any]]
    graph_summary: Dict[str, Any]

class AnswerQuestionRequest(BaseModel):
    """Request model for answering follow-up questions"""
    question_id: str
    answer: str
    confidence: Optional[float] = Field(1.0, description="User confidence in answer")

@router.post("/process-transcript", response_model=Dict[str, str])
@log_performance(logger)
async def process_transcript(
    request: ProcessTranscriptRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """
    Start AI processing of a transcript
    
    This endpoint initiates the complete AI processing pipeline:
    - Semantic chunking and analysis
    - Storyline graph construction
    - Chapter generation with multi-agent system
    - Follow-up question generation
    """
    try:
        logger.info("Starting transcript processing",
                   user_id=request.user_id,
                   transcript_length=len(request.transcript),
                   audio_session_id=request.audio_session_id)
        
        # Validate transcript
        if not request.transcript or len(request.transcript.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Transcript must be at least 100 characters long"
            )
        
        # Start processing
        session_id = await orchestrator.process_transcript(
            transcript=request.transcript,
            user_id=request.user_id,
            user_preferences=request.user_preferences
        )
        
        # Store session metadata in Firestore
        session_metadata = {
            "session_id": session_id,
            "user_id": request.user_id,
            "audio_session_id": request.audio_session_id,
            "transcript_length": len(request.transcript),
            "user_preferences": request.user_preferences,
            "created_at": datetime.now(),
            "status": "processing"
        }
        
        # Store in background to avoid blocking response
        background_tasks.add_task(
            firestore_service.create_document,
            "ai_sessions",
            session_id,
            session_metadata
        )
        
        return {
            "session_id": session_id,
            "status": "processing_started",
            "message": "AI processing has been initiated. Use the session_id to check progress."
        }
        
    except Exception as e:
        logger.error("Failed to start transcript processing",
                    error=str(e),
                    user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/status/{session_id}", response_model=ProcessingStatusResponse)
@log_performance(logger)
async def get_processing_status(session_id: str) -> ProcessingStatusResponse:
    """
    Get current processing status for a session
    
    Returns real-time status including:
    - Current processing stage
    - Progress percentage
    - Estimated completion time
    - Preview of results
    """
    try:
        status = await orchestrator.get_session_status(session_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        return ProcessingStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get processing status",
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/results/{session_id}", response_model=ProcessingResultsResponse)
@log_performance(logger)
async def get_processing_results(session_id: str) -> ProcessingResultsResponse:
    """
    Get complete processing results for a completed session
    
    Returns:
    - Generated chapters with metadata
    - Follow-up questions organized by category
    - Processing summary and statistics
    - Storyline analysis results
    """
    try:
        results = await orchestrator.get_session_results(session_id)
        
        if not results:
            # Check if session exists but isn't completed
            status = await orchestrator.get_session_status(session_id)
            if status:
                raise HTTPException(
                    status_code=202,
                    detail=f"Session {session_id} is still processing. Current progress: {status['progress_percentage']}%"
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found"
                )
        
        # Convert to response format
        chapters = [ChapterResponse(**chapter) for chapter in results.get("chapters", [])]
        
        followup_questions = [
            FollowupQuestionResponse(**question) 
            for question in results.get("followup_questions", [])
        ]
        
        question_categories = {}
        for category, questions in results.get("question_categories", {}).items():
            question_categories[category] = [
                FollowupQuestionResponse(**q) for q in questions
            ]
        
        return ProcessingResultsResponse(
            session_id=session_id,
            processing_summary=results.get("processing_summary", {}),
            chapters=chapters,
            followup_questions=followup_questions,
            question_categories=question_categories,
            storylines=results.get("storylines", []),
            graph_summary=results.get("graph_summary", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get processing results",
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

@router.post("/answer-question/{session_id}")
@log_performance(logger)
async def answer_followup_question(
    session_id: str,
    request: AnswerQuestionRequest
) -> Dict[str, str]:
    """
    Submit answer to a follow-up question
    
    This endpoint allows users to provide answers to generated follow-up questions,
    which can trigger reprocessing to improve content quality.
    """
    try:
        logger.info("Processing question answer",
                   session_id=session_id,
                   question_id=request.question_id)
        
        # Validate session exists
        status = await orchestrator.get_session_status(session_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Store answer in Firestore
        answer_data = {
            "session_id": session_id,
            "question_id": request.question_id,
            "answer": request.answer,
            "confidence": request.confidence,
            "answered_at": datetime.now()
        }
        
        await firestore_service.create_document(
            "question_answers",
            f"{session_id}_{request.question_id}",
            answer_data
        )
        
        # TODO: Implement answer processing and potential reprocessing
        # This would involve updating the graph with new information
        # and potentially regenerating affected chapters
        
        return {
            "status": "answer_recorded",
            "message": f"Answer to question {request.question_id} has been recorded"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process question answer",
                    session_id=session_id,
                    question_id=request.question_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Answer processing failed: {str(e)}")

@router.post("/cancel/{session_id}")
@log_performance(logger)
async def cancel_processing(session_id: str) -> Dict[str, str]:
    """
    Cancel an active processing session
    """
    try:
        success = await orchestrator.cancel_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or already completed"
            )
        
        logger.info("Processing session cancelled", session_id=session_id)
        
        return {
            "status": "cancelled",
            "message": f"Session {session_id} has been cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel processing session",
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")

@router.get("/health")
@log_performance(logger)
async def ai_health_check() -> Dict[str, Any]:
    """
    Health check for AI processing system
    
    Returns status of all AI components:
    - Orchestrator
    - Individual agents
    - Supporting services
    """
    try:
        health_status = await orchestrator.health_check()
        
        return {
            "status": "healthy" if health_status["orchestrator"] == "healthy" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": health_status
        }
        
    except Exception as e:
        logger.error("AI health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/sessions/{user_id}")
@log_performance(logger)
async def get_user_sessions(user_id: str) -> Dict[str, Any]:
    """
    Get all AI processing sessions for a user
    """
    try:
        # Query Firestore for user's sessions
        sessions = await firestore_service.query_documents(
            "ai_sessions",
            [("user_id", "==", user_id)],
            order_by=[("created_at", "desc")],
            limit=50
        )
        
        return {
            "user_id": user_id,
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        logger.error("Failed to get user sessions",
                    user_id=user_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Session retrieval failed: {str(e)}")

@router.get("/metrics")
@log_performance(logger)
async def get_processing_metrics() -> Dict[str, Any]:
    """
    Get AI processing system metrics
    """
    try:
        health_status = await orchestrator.health_check()
        
        metrics = {
            "active_sessions": health_status.get("active_sessions", 0),
            "completed_sessions": health_status.get("completed_sessions", 0),
            "system_status": health_status.get("orchestrator", "unknown"),
            "agent_status": {
                agent_id: agent_health.get("status", "unknown")
                for agent_id, agent_health in health_status.get("agents", {}).items()
            },
            "service_status": {
                service_id: service_health.get("status", "unknown")
                for service_id, service_health in health_status.get("services", {}).items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get processing metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# Startup and shutdown events
@router.on_event("startup")
async def startup_ai_processing():
    """Initialize AI processing system"""
    logger.info("AI processing system starting up")
    
    # Perform health check
    try:
        health = await orchestrator.health_check()
        logger.info("AI processing system health check completed", status=health["orchestrator"])
    except Exception as e:
        logger.error("AI processing system health check failed", error=str(e))

@router.on_event("shutdown") 
async def shutdown_ai_processing():
    """Clean shutdown of AI processing system"""
    logger.info("AI processing system shutting down")
    
    try:
        orchestrator.close()
        logger.info("AI processing system shutdown completed")
    except Exception as e:
        logger.error("AI processing system shutdown failed", error=str(e))