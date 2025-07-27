# Sequential Chapter Workflow Implementation

## Workflow Architecture

### Chapter Generation State Machine
```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

class ChapterStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    GENERATED = "generated"
    HARMONIZED = "harmonized"
    COMPLETED = "completed"
    FAILED = "failed"

class BookGenerationStatus(Enum):
    INITIALIZED = "initialized"
    PROCESSING = "processing"
    HARMONIZING = "harmonizing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ChapterWorkflowItem:
    chapter_id: str
    book_id: str
    book_version: str
    chapter_number: int
    storyline_node_id: str
    title: str
    status: ChapterStatus
    dependencies: List[str]  # Other chapter IDs that must complete first
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    agent_session_id: Optional[str] = None
```

## Sequential Processing Engine

### Workflow Coordinator
```python
class SequentialChapterWorkflow:
    def __init__(self, orchestrator: AgentOrchestrator, storage: ContentStorageService):
        self.orchestrator = orchestrator
        self.storage = storage
        self.active_sessions = {}  # book_id -> processing session
        
    async def start_book_generation(self, book_id: str, storyline_nodes: List[dict]) -> str:
        """Initialize sequential chapter generation for a book"""
        
        # Create book version
        version = await self._create_book_version(book_id)
        
        # Generate chapter workflow items
        workflow_items = await self._create_chapter_workflow(book_id, version, storyline_nodes)
        
        # Store workflow in processing session
        session_id = await self._create_processing_session(book_id, version, workflow_items)
        
        # Start processing first chapter
        await self._process_next_chapter(session_id)
        
        return session_id
    
    async def _create_chapter_workflow(self, book_id: str, version: str, storyline_nodes: List[dict]) -> List[ChapterWorkflowItem]:
        """Create ordered workflow items from storyline nodes"""
        
        # Sort nodes by centrality score (main storylines first)
        sorted_nodes = sorted(storyline_nodes, 
                            key=lambda x: x.get('centrality_score', 0), 
                            reverse=True)
        
        workflow_items = []
        for i, node in enumerate(sorted_nodes):
            chapter_item = ChapterWorkflowItem(
                chapter_id=f"{book_id}_{version}_chapter_{i+1}",
                book_id=book_id,
                book_version=version,
                chapter_number=i + 1,
                storyline_node_id=node['id'],
                title=node.get('summary', f"Chapter {i+1}"),
                status=ChapterStatus.PENDING,
                dependencies=self._calculate_dependencies(i, sorted_nodes),
                created_at=datetime.now()
            )
            workflow_items.append(chapter_item)
            
        return workflow_items
    
    async def _process_next_chapter(self, session_id: str) -> None:
        """Process the next available chapter in sequence"""
        
        session = await self._get_processing_session(session_id)
        
        # Find next chapter ready for processing
        next_chapter = self._find_next_ready_chapter(session.workflow_items)
        
        if not next_chapter:
            # No more chapters or waiting for dependencies
            if self._all_chapters_completed(session.workflow_items):
                await self._start_harmonization_phase(session_id)
            return
            
        # Update chapter status
        next_chapter.status = ChapterStatus.IN_PROGRESS
        next_chapter.started_at = datetime.now()
        
        # Start chapter generation
        try:
            await self._generate_chapter(session_id, next_chapter)
        except Exception as e:
            await self._handle_chapter_error(session_id, next_chapter, e)
```

### Chapter Generation Process
```python
    async def _generate_chapter(self, session_id: str, chapter_item: ChapterWorkflowItem) -> None:
        """Generate a single chapter using the Chapter Writer Agent"""
        
        # Load storyline node and associated chunks
        storyline_node = await self._load_storyline_node(chapter_item.storyline_node_id)
        text_chunks = await self._load_associated_chunks(storyline_node)
        
        # Prepare context from previous chapters
        previous_chapters = await self._get_previous_chapters(chapter_item)
        context = self._build_chapter_context(previous_chapters, storyline_node)
        
        # Create Chapter Writer Agent task
        task_request = {
            "task_type": "write_chapter",
            "chapter_number": chapter_item.chapter_number,
            "title": chapter_item.title,
            "storyline_node": storyline_node,
            "text_chunks": text_chunks,
            "previous_context": context,
            "book_metadata": await self._get_book_metadata(chapter_item.book_id),
            "user_preferences": await self._get_user_preferences(chapter_item.book_id)
        }
        
        # Execute chapter writing
        agent_result = await self.orchestrator.execute_chapter_writer_task(task_request)
        
        # Process result
        if agent_result.success:
            await self._store_generated_chapter(chapter_item, agent_result)
            chapter_item.status = ChapterStatus.GENERATED
            chapter_item.completed_at = datetime.now()
            
            # Continue with next chapter
            await self._process_next_chapter(session_id)
        else:
            raise Exception(f"Chapter generation failed: {agent_result.error}")
    
    async def _store_generated_chapter(self, chapter_item: ChapterWorkflowItem, agent_result) -> None:
        """Store the generated chapter content and metadata"""
        
        chapter_document = {
            "metadata": {
                "book_id": chapter_item.book_id,
                "book_version": chapter_item.book_version,
                "chapter_number": chapter_item.chapter_number,
                "title": agent_result.title,
                "word_count": len(agent_result.content.split()),
                "status": "generated",
                "quality_score": agent_result.quality_score,
                "created_at": datetime.now(),
                "generation_agent": "chapter_writer"
            },
            "content": {
                "markdown_text": agent_result.content,
                "themes": agent_result.themes,
                "participants": agent_result.participants,
                "tags": agent_result.tags
            },
            "source_references": {
                "storyline_node_ids": [chapter_item.storyline_node_id],
                "transcript_chunk_ids": agent_result.source_chunk_ids,
                "source_confidence": agent_result.confidence
            },
            "generation_metadata": {
                "agent_type": "chapter_writer",
                "processing_session_id": chapter_item.agent_session_id,
                "generation_timestamp": datetime.now(),
                "harmonization_applied": False,
                "quality_metrics": agent_result.quality_metrics
            }
        }
        
        await self.storage.store_chapter(chapter_item.chapter_id, chapter_document)
```

### Harmonization Phase
```python
    async def _start_harmonization_phase(self, session_id: str) -> None:
        """Start inter-chapter harmonization after all chapters are generated"""
        
        session = await self._get_processing_session(session_id)
        session.status = BookGenerationStatus.HARMONIZING
        
        # Load all generated chapters
        chapters = await self._load_all_chapters(session.book_id, session.book_version)
        
        # Create harmonization task
        harmonization_task = {
            "task_type": "harmonize_chapters",
            "book_id": session.book_id,
            "book_version": session.book_version,
            "chapters": chapters,
            "consistency_rules": await self._get_consistency_rules()
        }
        
        # Execute harmonization
        harmonizer_result = await self.orchestrator.execute_harmonizer_task(harmonization_task)
        
        if harmonizer_result.success:
            # Apply harmonization changes
            await self._apply_harmonization_changes(harmonizer_result.changes)
            
            # Generate follow-up questions
            await self._generate_followup_questions(session_id)
            
            # Mark book as completed
            session.status = BookGenerationStatus.COMPLETED
            await self._update_processing_session(session)
        else:
            session.status = BookGenerationStatus.FAILED
            session.error_message = harmonizer_result.error
```

### Error Handling and Recovery
```python
    async def _handle_chapter_error(self, session_id: str, chapter_item: ChapterWorkflowItem, error: Exception) -> None:
        """Handle chapter generation errors with retry logic"""
        
        chapter_item.retry_count += 1
        chapter_item.error_message = str(error)
        
        if chapter_item.retry_count < 3:  # Max 3 retries
            # Reset status for retry
            chapter_item.status = ChapterStatus.PENDING
            chapter_item.started_at = None
            
            # Wait before retry (exponential backoff)
            await asyncio.sleep(2 ** chapter_item.retry_count)
            
            # Retry generation
            await self._process_next_chapter(session_id)
        else:
            # Mark as failed and continue with other chapters
            chapter_item.status = ChapterStatus.FAILED
            
            # Skip dependencies that depend on this failed chapter
            await self._handle_dependency_failure(session_id, chapter_item)
            
            # Continue with other available chapters
            await self._process_next_chapter(session_id)
    
    def _find_next_ready_chapter(self, workflow_items: List[ChapterWorkflowItem]) -> Optional[ChapterWorkflowItem]:
        """Find the next chapter ready for processing based on dependencies"""
        
        for item in workflow_items:
            if item.status != ChapterStatus.PENDING:
                continue
                
            # Check if all dependencies are completed
            dependencies_met = all(
                self._find_chapter_by_id(dep_id, workflow_items).status == ChapterStatus.COMPLETED
                for dep_id in item.dependencies
            )
            
            if dependencies_met:
                return item
                
        return None
```

### Processing Session Management
```python
@dataclass
class ProcessingSession:
    session_id: str
    book_id: str
    book_version: str
    status: BookGenerationStatus
    workflow_items: List[ChapterWorkflowItem]
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    
class ProcessingSessionManager:
    def __init__(self, storage: ContentStorageService):
        self.storage = storage
    
    async def create_session(self, book_id: str, version: str, workflow_items: List[ChapterWorkflowItem]) -> str:
        """Create a new processing session"""
        
        session_id = f"{book_id}_{version}_{int(datetime.now().timestamp())}"
        
        session = ProcessingSession(
            session_id=session_id,
            book_id=book_id,
            book_version=version,
            status=BookGenerationStatus.INITIALIZED,
            workflow_items=workflow_items,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        await self.storage.store_processing_session(session)
        return session_id
    
    async def get_session_status(self, session_id: str) -> dict:
        """Get current processing status"""
        
        session = await self.storage.get_processing_session(session_id)
        
        completed_chapters = sum(1 for item in session.workflow_items 
                               if item.status == ChapterStatus.COMPLETED)
        total_chapters = len(session.workflow_items)
        
        current_chapter = next((item for item in session.workflow_items 
                              if item.status == ChapterStatus.IN_PROGRESS), None)
        
        return {
            "session_id": session_id,
            "book_id": session.book_id,
            "book_version": session.book_version,
            "status": session.status.value,
            "progress_percentage": (completed_chapters / total_chapters) * 100,
            "completed_chapters": completed_chapters,
            "total_chapters": total_chapters,
            "current_chapter": current_chapter.title if current_chapter else None,
            "estimated_completion": self._estimate_completion_time(session),
            "updated_at": session.updated_at.isoformat()
        }
```

### Dependency Management
```python
    def _calculate_dependencies(self, chapter_index: int, storyline_nodes: List[dict]) -> List[str]:
        """Calculate chapter dependencies based on storyline relationships"""
        
        # For MVP: Simple sequential dependency (each chapter depends on previous)
        if chapter_index == 0:
            return []  # First chapter has no dependencies
        
        # Each chapter depends on the previous chapter for narrative flow
        previous_chapter_id = f"chapter_{chapter_index}"
        return [previous_chapter_id]
        
        # Future enhancement: Complex dependency based on storyline graph relationships
        # current_node = storyline_nodes[chapter_index]
        # dependencies = self._analyze_storyline_dependencies(current_node, storyline_nodes)
        # return dependencies
```

## API Integration

### Workflow API Endpoints
```python
@router.post("/start-book-generation/{book_id}")
async def start_book_generation(book_id: str, storyline_nodes: List[dict]) -> dict:
    """Start sequential chapter generation workflow"""
    
    session_id = await workflow.start_book_generation(book_id, storyline_nodes)
    
    return {
        "session_id": session_id,
        "status": "started",
        "message": "Sequential chapter generation initiated"
    }

@router.get("/workflow-status/{session_id}")
async def get_workflow_status(session_id: str) -> dict:
    """Get current workflow processing status"""
    
    return await session_manager.get_session_status(session_id)

@router.post("/retry-failed-chapter/{session_id}/{chapter_id}")
async def retry_failed_chapter(session_id: str, chapter_id: str) -> dict:
    """Retry generation of a failed chapter"""
    
    success = await workflow.retry_chapter(session_id, chapter_id)
    
    return {
        "success": success,
        "message": "Chapter retry initiated" if success else "Retry failed"
    }
```

This implementation provides a robust sequential chapter generation workflow that maintains narrative consistency while handling errors gracefully and providing real-time progress tracking.