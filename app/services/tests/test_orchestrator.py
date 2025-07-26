"""
Tests for agent orchestrator
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from ai_implementation.orchestrator import AgentOrchestrator, ProcessingStage

@pytest.fixture
def sample_transcript():
    """Sample transcript for testing"""
    return """
    My name is Alice and I grew up in Boston in the 1990s.
    I have many memories of playing in the park with my sister Emma.
    Our parents worked at the local hospital as doctors.
    Every summer, we would visit our grandparents in Maine.
    """

@pytest.fixture
def orchestrator():
    """Create orchestrator instance for testing"""
    # Use mock credentials to avoid requiring real services
    return AgentOrchestrator(
        anthropic_api_key=None,  # Will fall back to template generation
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j", 
        neo4j_password="test"
    )

@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test that orchestrator initializes properly"""
    assert orchestrator is not None
    assert len(orchestrator.agents) > 0
    assert "chapter_writer" in orchestrator.agents
    assert "chapter_harmonizer" in orchestrator.agents
    assert "followup_questions" in orchestrator.agents

@pytest.mark.asyncio
async def test_process_transcript_session_creation(orchestrator, sample_transcript):
    """Test that process_transcript creates a session"""
    session_id = await orchestrator.process_transcript(
        transcript=sample_transcript,
        user_id="test_user",
        user_preferences={"writing_style": "memoir"}
    )
    
    assert session_id is not None
    assert isinstance(session_id, str)
    assert len(session_id) > 0
    
    # Session should be in active sessions
    assert session_id in orchestrator.active_sessions
    
    session = orchestrator.active_sessions[session_id]
    assert session.user_id == "test_user"
    assert session.transcript == sample_transcript

@pytest.mark.asyncio
async def test_get_session_status(orchestrator, sample_transcript):
    """Test getting session status"""
    session_id = await orchestrator.process_transcript(
        transcript=sample_transcript,
        user_id="test_user"
    )
    
    # Wait a moment for processing to start
    await asyncio.sleep(0.1)
    
    status = await orchestrator.get_session_status(session_id)
    
    assert status is not None
    assert status["session_id"] == session_id
    assert status["user_id"] == "test_user"
    assert "current_stage" in status
    assert "progress_percentage" in status
    assert "start_time" in status

@pytest.mark.asyncio
async def test_get_session_status_nonexistent(orchestrator):
    """Test getting status for nonexistent session"""
    status = await orchestrator.get_session_status("nonexistent_session")
    assert status is None

@pytest.mark.asyncio
async def test_progress_callback(orchestrator, sample_transcript):
    """Test progress callback functionality"""
    progress_updates = []
    
    async def progress_callback(update):
        progress_updates.append(update)
    
    session_id = await orchestrator.process_transcript(
        transcript=sample_transcript,
        user_id="test_user",
        progress_callback=progress_callback
    )
    
    # Wait for some processing
    await asyncio.sleep(1.0)
    
    # Should have received some progress updates
    assert len(progress_updates) > 0
    
    for update in progress_updates:
        assert update.session_id == session_id
        assert hasattr(update, 'stage')
        assert hasattr(update, 'progress_percentage')
        assert hasattr(update, 'current_task')

@pytest.mark.asyncio
async def test_cancel_session(orchestrator, sample_transcript):
    """Test session cancellation"""
    session_id = await orchestrator.process_transcript(
        transcript=sample_transcript,
        user_id="test_user"
    )
    
    # Session should be active
    assert session_id in orchestrator.active_sessions
    
    # Cancel session
    success = await orchestrator.cancel_session(session_id)
    assert success is True
    
    # Session should no longer be active
    assert session_id not in orchestrator.active_sessions
    
    # Cancelling again should return False
    success2 = await orchestrator.cancel_session(session_id)
    assert success2 is False

@pytest.mark.asyncio
async def test_health_check(orchestrator):
    """Test orchestrator health check"""
    health = await orchestrator.health_check()
    
    assert health is not None
    assert "orchestrator" in health
    assert "agents" in health
    assert "services" in health
    
    # Should have health info for all agents
    expected_agents = ["chapter_writer", "chapter_harmonizer", "followup_questions"]
    for agent_id in expected_agents:
        assert agent_id in health["agents"]

@pytest.mark.asyncio 
async def test_estimate_completion_time(orchestrator, sample_transcript):
    """Test completion time estimation"""
    session_id = await orchestrator.process_transcript(
        transcript=sample_transcript,
        user_id="test_user"
    )
    
    # Wait for some progress
    await asyncio.sleep(0.5)
    
    session = orchestrator.active_sessions[session_id]
    completion_time = orchestrator._estimate_completion_time(session)
    
    if session.progress_percentage > 0:
        assert completion_time is not None
        assert completion_time > datetime.now()

def test_create_results_preview(orchestrator):
    """Test results preview creation"""
    from ai_implementation.orchestrator import ProcessingSession
    
    session = ProcessingSession(
        session_id="test_session",
        user_id="test_user", 
        transcript="test transcript",
        current_stage=ProcessingStage.SEMANTIC_PROCESSING,
        start_time=datetime.now(),
        last_checkpoint=datetime.now(),
        progress_percentage=25.0,
        results={
            "chunks_created": 10,
            "storylines": [{"id": "story1"}, {"id": "story2"}]
        },
        errors=[],
        user_preferences={}
    )
    
    preview = orchestrator._create_results_preview(session)
    
    assert preview["stage"] == ProcessingStage.SEMANTIC_PROCESSING.value
    assert preview["progress"] == 25.0
    assert preview["chunks_created"] == 10
    assert preview["storylines_identified"] == 2

@pytest.mark.asyncio
async def test_session_timeout_handling(orchestrator):
    """Test that sessions are properly cleaned up"""
    # This test verifies the orchestrator can handle session management
    # In a real implementation, would test timeout cleanup
    
    initial_session_count = len(orchestrator.active_sessions)
    
    # Create a session
    session_id = await orchestrator.process_transcript(
        transcript="Short transcript for timeout test.",
        user_id="timeout_test_user"
    )
    
    assert len(orchestrator.active_sessions) == initial_session_count + 1
    
    # Cancel the session to clean up
    await orchestrator.cancel_session(session_id)
    
    assert len(orchestrator.active_sessions) == initial_session_count

@pytest.mark.asyncio
async def test_concurrent_sessions(orchestrator, sample_transcript):
    """Test handling multiple concurrent sessions"""
    session_ids = []
    
    # Start multiple sessions
    for i in range(3):
        session_id = await orchestrator.process_transcript(
            transcript=f"{sample_transcript} Session {i}",
            user_id=f"user_{i}"
        )
        session_ids.append(session_id)
    
    # All sessions should be active
    for session_id in session_ids:
        assert session_id in orchestrator.active_sessions
    
    # Clean up
    for session_id in session_ids:
        await orchestrator.cancel_session(session_id)

def test_orchestrator_configuration(orchestrator):
    """Test orchestrator configuration parameters"""
    assert orchestrator.max_concurrent_sessions > 0
    assert orchestrator.checkpoint_interval.total_seconds() > 0
    assert orchestrator.session_timeout.total_seconds() > 0

def test_orchestrator_close(orchestrator):
    """Test orchestrator cleanup"""
    initial_sessions = len(orchestrator.active_sessions)
    
    # Close orchestrator
    orchestrator.close()
    
    # Should clear active sessions
    assert len(orchestrator.active_sessions) == 0
    assert len(orchestrator.progress_callbacks) == 0