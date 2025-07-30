"""
Agent Orchestrator for coordinating the multi-agent AI processing pipeline
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import json
from enum import Enum
import traceback

import numpy as np

from app.utils.logging_utils import get_logger, log_performance
from .semantic_chunker import SemanticChunker
from .graph_builder import GraphBuilder, StorylineNode
from .agents import (
    BaseAgent, 
    ChapterWriterAgent, 
    ChapterHarmonizerAgent,
    FollowupQuestionsAgent,
    TaskRequest,
    TaskResult,
    TaskStatus,
    AgentMessage
)

logger = get_logger("orchestrator")

class ProcessingStage(Enum):
    INITIALIZATION = "initialization"
    SEMANTIC_PROCESSING = "semantic_processing"
    GRAPH_CONSTRUCTION = "graph_construction"
    AGENT_COORDINATION = "agent_coordination"
    CONTENT_GENERATION = "content_generation"
    QUALITY_ASSURANCE = "quality_assurance"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingSession:
    """Container for processing session state"""
    session_id: str
    user_id: str
    transcript: str
    current_stage: ProcessingStage
    start_time: datetime
    last_checkpoint: datetime
    progress_percentage: float
    results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]

@dataclass
class ProgressUpdate:
    """Progress update for user interface"""
    session_id: str
    stage: ProcessingStage
    progress_percentage: float
    current_task: str
    estimated_completion: Optional[datetime]
    results_preview: Dict[str, Any]
    timestamp: datetime

class AgentOrchestrator:
    """
    Central orchestrator that coordinates the entire AI processing pipeline,
    managing agents, tracking progress, and handling user interactions.
    """
    
    def __init__(self, 
                 anthropic_api_key: Optional[str] = None,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password"):
        """
        Initialize orchestrator with all required services
        
        Args:
            anthropic_api_key: API key for Claude LLM
            neo4j_uri: Neo4j database connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        # Initialize core services
        self.semantic_chunker = SemanticChunker()
        self.graph_builder = GraphBuilder()
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {}
        self._register_agents(anthropic_api_key)
        
        # Session management
        self.active_sessions: Dict[str, ProcessingSession] = {}
        self.completed_sessions: Dict[str, ProcessingSession] = {}
        
        # Progress tracking
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        
        # Configuration
        self.checkpoint_interval = timedelta(minutes=5)
        self.max_concurrent_sessions = 5
        self.session_timeout = timedelta(hours=4)
        
        logger.info("Agent orchestrator initialized",
                   agents_count=len(self.agents),
                   max_concurrent_sessions=self.max_concurrent_sessions)
    
    def _register_agents(self, anthropic_api_key: Optional[str]):
        """Register all agents with the orchestrator"""
        self.agents["chapter_writer"] = ChapterWriterAgent(
            anthropic_api_key=anthropic_api_key
        )
        self.agents["chapter_harmonizer"] = ChapterHarmonizerAgent()
        self.agents["followup_questions"] = FollowupQuestionsAgent()
        
        logger.info("Agents registered", agent_ids=list(self.agents.keys()))
    
    @log_performance(logger)
    async def process_transcript(self,
                               transcript: str,
                               user_id: str,
                               user_preferences: Optional[Dict[str, Any]] = None,
                               progress_callback: Optional[Callable] = None) -> str:
        """
        Main entry point for processing a transcript through the entire pipeline
        
        Args:
            transcript: Raw transcript text to process
            user_id: Unique identifier for the user
            user_preferences: User customization preferences
            progress_callback: Callback function for progress updates
            
        Returns:
            Session ID for tracking processing progress
        """
        # Create new processing session
        session_id = str(uuid.uuid4())
        session = ProcessingSession(
            session_id=session_id,
            user_id=user_id,
            transcript=transcript,
            current_stage=ProcessingStage.INITIALIZATION,
            start_time=datetime.now(),
            last_checkpoint=datetime.now(),
            progress_percentage=0.0,
            results={},
            errors=[],
            user_preferences=user_preferences or {}
        )
        
        self.active_sessions[session_id] = session
        
        if progress_callback:
            if session_id not in self.progress_callbacks:
                self.progress_callbacks[session_id] = []
            self.progress_callbacks[session_id].append(progress_callback)
        
        logger.info("Starting transcript processing",
                   session_id=session_id,
                   user_id=user_id,
                   transcript_length=len(transcript))
        
        # Start processing in background
        asyncio.create_task(self._process_session(session))
        
        return session_id
    
    async def _process_session(self, session: ProcessingSession):
        """Process a complete session through all pipeline stages"""
        try:
            await self._update_progress(session, ProcessingStage.INITIALIZATION, 5.0, 
                                      "Initializing processing pipeline")
            
            # Stage 1: Semantic Processing
            await self._stage_semantic_processing(session)
            await self._checkpoint_session(session)
            
            # Stage 2: Graph Construction
            await self._stage_graph_construction(session)
            await self._checkpoint_session(session)
            
            # Stage 3: Agent Coordination and Content Generation
            await self._stage_content_generation(session)
            await self._checkpoint_session(session)
            
            # Stage 4: Quality Assurance
            await self._stage_quality_assurance(session)
            await self._checkpoint_session(session)
            
            # Stage 5: Finalization
            await self._stage_finalization(session)
            
            # Mark as completed
            session.current_stage = ProcessingStage.COMPLETED
            session.progress_percentage = 100.0
            
            self.completed_sessions[session.session_id] = session
            del self.active_sessions[session.session_id]
            
            await self._update_progress(session, ProcessingStage.COMPLETED, 100.0,
                                      "Processing completed successfully")
            
            logger.info("Session processing completed",
                       session_id=session.session_id,
                       duration_seconds=(datetime.now() - session.start_time).total_seconds())
            
        except Exception as e:
            logger.error("Session processing failed",
                        session_id=session.session_id,
                        error=str(e),
                        traceback=traceback.format_exc())
            
            session.current_stage = ProcessingStage.FAILED
            session.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            await self._update_progress(session, ProcessingStage.FAILED, session.progress_percentage,
                                      f"Processing failed: {str(e)}")
    
    async def _stage_semantic_processing(self, session: ProcessingSession):
        """Stage 1: Semantic chunking and text analysis"""
        await self._update_progress(session, ProcessingStage.SEMANTIC_PROCESSING, 10.0,
                                  "Processing transcript into semantic chunks")
        
        # Process transcript with semantic chunker
        processed_transcript = await self.semantic_chunker.process_transcript(
            session.transcript,
            chunk_size=session.user_preferences.get("chunk_size", 20),
            preserve_speaker_boundaries=session.user_preferences.get("preserve_speakers", True),
            merge_similar_chunks=session.user_preferences.get("merge_similar", True)
        )
        
        session.results["processed_transcript"] = processed_transcript
        session.results["chunks_created"] = len(processed_transcript.chunks)
        
        await self._update_progress(session, ProcessingStage.SEMANTIC_PROCESSING, 25.0,
                                  f"Created {len(processed_transcript.chunks)} semantic chunks")
        
        logger.info("Semantic processing completed",
                   session_id=session.session_id,
                   chunks_created=len(processed_transcript.chunks))
    
    async def _stage_graph_construction(self, session: ProcessingSession):
        """Stage 2: Graph construction and storyline identification"""
        await self._update_progress(session, ProcessingStage.GRAPH_CONSTRUCTION, 30.0,
                                  "Building storyline graph from semantic chunks")
        
        processed_transcript = session.results["processed_transcript"]
        
        # Build storyline graph
        graph = await self.graph_builder.build_storyline_graph(processed_transcript.chunks)
        
        await self._update_progress(session, ProcessingStage.GRAPH_CONSTRUCTION, 40.0,
                                  "Analyzing graph structure and identifying storylines")
        
        # Identify main storylines
        storylines = await self.graph_builder.identify_main_storylines(graph)
        
        # Save graph to Neo4j
        await self.graph_builder.save_graph_to_neo4j(graph, session.session_id)
        
        session.results["graph"] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "communities": len(set([graph.nodes[node].get("community", 0) for node in graph.nodes()]))
        }
        session.results["storylines"] = [asdict(storyline) for storyline in storylines]
        session.results["main_storylines"] = [
            asdict(s) for s in storylines if s.main_storyline
        ]
        
        await self._update_progress(session, ProcessingStage.GRAPH_CONSTRUCTION, 50.0,
                                  f"Identified {len(storylines)} storylines ({len([s for s in storylines if s.main_storyline])} main)")
        
        logger.info("Graph construction completed",
                   session_id=session.session_id,
                   storylines_count=len(storylines),
                   main_storylines=len([s for s in storylines if s.main_storyline]))
    
    async def _stage_content_generation(self, session: ProcessingSession):
        """Stage 3: Multi-agent content generation"""
        await self._update_progress(session, ProcessingStage.CONTENT_GENERATION, 55.0,
                                  "Generating chapters with AI agents")
        
        storylines = [StorylineNode(**data) for data in session.results["storylines"]]
        main_storylines = [s for s in storylines if s.main_storyline]
        processed_transcript = session.results["processed_transcript"]
        
        # Create chunk lookup
        chunk_dict = {chunk.id: chunk for chunk in processed_transcript.chunks}
        
        # Generate chapters for main storylines
        chapters = []
        chapter_tasks = []
        
        for i, storyline in enumerate(main_storylines):
            # Prepare context chunks for this storyline
            context_chunks = []
            for chunk_id in storyline.chunk_ids:
                if chunk_id in chunk_dict:
                    chunk = chunk_dict[chunk_id]
                    context_chunks.append({
                        "id": chunk.id,
                        "content": chunk.content,
                        "start_position": chunk.start_position,
                        "speaker": chunk.speaker,
                        "entities": chunk.entities
                    })
            
            # Create chapter writing task
            task = TaskRequest(
                task_id=f"chapter_{i}_{session.session_id}",
                task_type="write_chapter",
                agent_id="chapter_writer",
                parameters={
                    "storyline": asdict(storyline),
                    "context_chunks": context_chunks,
                    "user_preferences": session.user_preferences
                },
                priority="high",
                created_at=datetime.now()
            )
            
            chapter_tasks.append(task)
        
        # Execute chapter writing tasks
        for i, task in enumerate(chapter_tasks):
            progress = 55.0 + (i / len(chapter_tasks)) * 25.0
            await self._update_progress(session, ProcessingStage.CONTENT_GENERATION, progress,
                                      f"Generating chapter {i+1} of {len(chapter_tasks)}")
            
            result = await self.agents["chapter_writer"].process_task(task)
            
            if result.status == "completed":
                chapters.append(result.result_data)
            else:
                logger.error("Chapter generation failed",
                           task_id=task.task_id,
                           error=result.error_message)
                session.errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task.task_id,
                    "error": result.error_message
                })
        
        session.results["chapters"] = chapters
        
        await self._update_progress(session, ProcessingStage.CONTENT_GENERATION, 80.0,
                                  f"Generated {len(chapters)} chapters")
        
        logger.info("Content generation completed",
                   session_id=session.session_id,
                   chapters_generated=len(chapters))
    
    async def _stage_quality_assurance(self, session: ProcessingSession):
        """Stage 4: Chapter harmonization and quality control"""
        await self._update_progress(session, ProcessingStage.QUALITY_ASSURANCE, 85.0,
                                  "Harmonizing chapters for consistency")
        
        chapters = session.results.get("chapters", [])
        
        if len(chapters) > 1:
            # Create harmonization task
            harmonization_task = TaskRequest(
                task_id=f"harmonize_{session.session_id}",
                task_type="harmonize_chapters",
                agent_id="chapter_harmonizer",
                parameters={
                    "chapters": chapters
                },
                priority="high",
                created_at=datetime.now()
            )
            
            # Execute harmonization
            result = await self.agents["chapter_harmonizer"].process_task(harmonization_task)
            
            if result.status == "completed":
                session.results["harmonized_chapters"] = result.result_data["harmonized_chapters"]
                session.results["harmonization_summary"] = result.result_data["harmonization_summary"]
            else:
                logger.warning("Chapter harmonization failed, using original chapters",
                             error=result.error_message)
                session.results["harmonized_chapters"] = chapters
        else:
            session.results["harmonized_chapters"] = chapters
        
        # Generate follow-up questions
        await self._update_progress(session, ProcessingStage.QUALITY_ASSURANCE, 90.0,
                                  "Generating follow-up questions")
        
        questions_task = TaskRequest(
            task_id=f"questions_{session.session_id}",
            task_type="generate_questions",
            agent_id="followup_questions",
            parameters={
                "storylines": session.results["storylines"],
                "graph_data": session.results["graph"]
            },
            priority="medium",
            created_at=datetime.now()
        )
        
        result = await self.agents["followup_questions"].process_task(questions_task)
        
        if result.status == "completed":
            session.results["followup_questions"] = result.result_data["questions"]
            session.results["question_categories"] = result.result_data["categorized_questions"]
        else:
            logger.warning("Question generation failed", error=result.error_message)
            session.results["followup_questions"] = []
            session.results["question_categories"] = {}
        
        await self._update_progress(session, ProcessingStage.QUALITY_ASSURANCE, 95.0,
                                  f"Generated {len(session.results.get('followup_questions', []))} follow-up questions")
        
        logger.info("Quality assurance completed",
                   session_id=session.session_id,
                   questions_generated=len(session.results.get("followup_questions", [])))
    
    async def _stage_finalization(self, session: ProcessingSession):
        """Stage 5: Final assembly and formatting"""
        await self._update_progress(session, ProcessingStage.FINALIZATION, 98.0,
                                  "Finalizing output and generating summary")
        
        # Create final processing summary
        harmonized_chapters = session.results.get("harmonized_chapters", [])
        followup_questions = session.results.get("followup_questions", [])
        
        processing_summary = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "processing_time_seconds": (datetime.now() - session.start_time).total_seconds(),
            "transcript_stats": {
                "original_length": len(session.transcript),
                "word_count": len(session.transcript.split()),
                "chunks_created": session.results.get("chunks_created", 0)
            },
            "graph_stats": session.results.get("graph", {}),
            "content_stats": {
                "chapters_generated": len(harmonized_chapters),
                "total_words": sum(chapter.get("word_count", 0) for chapter in harmonized_chapters),
                "avg_chapter_length": np.mean([chapter.get("word_count", 0) for chapter in harmonized_chapters]) if harmonized_chapters else 0,
                "quality_scores": [chapter.get("quality_score", 0) for chapter in harmonized_chapters]
            },
            "interaction_stats": {
                "questions_generated": len(followup_questions),
                "question_categories": list(session.results.get("question_categories", {}).keys())
            },
            "harmonization_applied": bool(session.results.get("harmonization_summary")),
            "errors_encountered": len(session.errors)
        }
        
        session.results["processing_summary"] = processing_summary
        
        logger.info("Finalization completed",
                   session_id=session.session_id,
                   **processing_summary["content_stats"])
    
    async def _update_progress(self, 
                             session: ProcessingSession,
                             stage: ProcessingStage,
                             progress: float,
                             task_description: str):
        """Update session progress and notify callbacks"""
        session.current_stage = stage
        session.progress_percentage = progress
        session.last_checkpoint = datetime.now()
        
        # Create progress update
        progress_update = ProgressUpdate(
            session_id=session.session_id,
            stage=stage,
            progress_percentage=progress,
            current_task=task_description,
            estimated_completion=self._estimate_completion_time(session),
            results_preview=self._create_results_preview(session),
            timestamp=datetime.now()
        )
        
        # Notify progress callbacks
        callbacks = self.progress_callbacks.get(session.session_id, [])
        for callback in callbacks:
            try:
                await callback(progress_update)
            except Exception as e:
                logger.error("Progress callback failed", 
                           session_id=session.session_id,
                           error=str(e))
        
        logger.info("Progress updated",
                   session_id=session.session_id,
                   stage=stage.value,
                   progress=progress,
                   task=task_description)
    
    def _estimate_completion_time(self, session: ProcessingSession) -> Optional[datetime]:
        """Estimate when processing will complete"""
        if session.progress_percentage <= 0:
            return None
        
        elapsed_time = datetime.now() - session.start_time
        estimated_total_time = elapsed_time / (session.progress_percentage / 100.0)
        estimated_completion = session.start_time + estimated_total_time
        
        return estimated_completion
    
    def _create_results_preview(self, session: ProcessingSession) -> Dict[str, Any]:
        """Create preview of current results"""
        preview = {
            "stage": session.current_stage.value,
            "progress": session.progress_percentage
        }
        
        if "chunks_created" in session.results:
            preview["chunks_created"] = session.results["chunks_created"]
        
        if "storylines" in session.results:
            preview["storylines_identified"] = len(session.results["storylines"])
            preview["main_storylines"] = len([s for s in session.results["storylines"] if s.get("main_storyline")])
        
        if "chapters" in session.results:
            preview["chapters_generated"] = len(session.results["chapters"])
        
        if "followup_questions" in session.results:
            preview["questions_generated"] = len(session.results["followup_questions"])
        
        return preview
    
    async def _checkpoint_session(self, session: ProcessingSession):
        """Save session checkpoint"""
        session.last_checkpoint = datetime.now()
        
        # In real implementation, would save to persistent storage
        logger.debug("Session checkpoint saved",
                    session_id=session.session_id,
                    stage=session.current_stage.value,
                    progress=session.progress_percentage)
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a processing session"""
        session = self.active_sessions.get(session_id) or self.completed_sessions.get(session_id)
        
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "current_stage": session.current_stage.value,
            "progress_percentage": session.progress_percentage,
            "start_time": session.start_time.isoformat(),
            "last_checkpoint": session.last_checkpoint.isoformat(),
            "estimated_completion": self._estimate_completion_time(session).isoformat() if self._estimate_completion_time(session) else None,
            "results_preview": self._create_results_preview(session),
            "errors": session.errors,
            "is_completed": session.current_stage in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]
        }
    
    async def get_session_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get complete results for a session"""
        session = self.completed_sessions.get(session_id)
        
        if not session or session.current_stage != ProcessingStage.COMPLETED:
            return None
        
        return {
            "session_id": session.session_id,
            "processing_summary": session.results.get("processing_summary", {}),
            "chapters": session.results.get("harmonized_chapters", []),
            "followup_questions": session.results.get("followup_questions", []),
            "question_categories": session.results.get("question_categories", {}),
            "storylines": session.results.get("storylines", []),
            "graph_summary": session.results.get("graph", {}),
            "harmonization_summary": session.results.get("harmonization_summary", {}),
            "errors": session.errors
        }
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an active processing session"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.current_stage = ProcessingStage.FAILED
        session.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": "Session cancelled by user request"
        })
        
        del self.active_sessions[session_id]
        
        logger.info("Session cancelled", session_id=session_id)
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for orchestrator and all components"""
        health_status = {
            "orchestrator": "healthy",
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.completed_sessions),
            "agents": {},
            "services": {}
        }
        
        # Check agent health
        for agent_id, agent in self.agents.items():
            try:
                agent_health = await agent.health_check()
                health_status["agents"][agent_id] = agent_health
            except Exception as e:
                health_status["agents"][agent_id] = {"status": "unhealthy", "error": str(e)}
        
        # Check service health
        try:
            chunker_health = await self.semantic_chunker.health_check()
            health_status["services"]["semantic_chunker"] = chunker_health
        except Exception as e:
            health_status["services"]["semantic_chunker"] = {"status": "unhealthy", "error": str(e)}
        
        try:
            graph_health = await self.graph_builder.health_check()
            health_status["services"]["graph_builder"] = graph_health
        except Exception as e:
            health_status["services"]["graph_builder"] = {"status": "unhealthy", "error": str(e)}
        
        # Overall status
        unhealthy_components = []
        for agent_id, agent_health in health_status["agents"].items():
            if agent_health.get("status") == "unhealthy":
                unhealthy_components.append(f"agent:{agent_id}")
        
        for service_id, service_health in health_status["services"].items():
            if service_health.get("status") == "unhealthy":
                unhealthy_components.append(f"service:{service_id}")
        
        if unhealthy_components:
            health_status["orchestrator"] = "degraded"
            health_status["unhealthy_components"] = unhealthy_components
        
        return health_status
    
    def close(self):
        """Clean shutdown of orchestrator and all services"""
        logger.info("Shutting down orchestrator")
        
        # Close graph builder
        self.graph_builder.close()
        
        # Clear active sessions
        self.active_sessions.clear()
        self.progress_callbacks.clear()
        
        logger.info("Orchestrator shutdown completed")

# Global orchestrator instance
orchestrator = AgentOrchestrator()