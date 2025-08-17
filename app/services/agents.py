"""
Multi-Agent System for Content Generation and Harmonization
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import re
import numpy as np
from abc import ABC, abstractmethod
from anthropic import Anthropic
from mistralai import Mistral
import google.generativeai as genai
from app.utils.logging_utils import get_logger, log_performance
from app.config.settings_config import get_settings
from .graph_builder import StorylineNode

logger = get_logger("agents")

# Message and Task Management
@dataclass
class AgentMessage:
    """Structured inter-agent communication"""
    sender: str
    receiver: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: str
    priority: str = "normal"

@dataclass
class TaskRequest:
    """Work assignment with parameters and priorities"""
    task_id: str
    task_type: str
    agent_id: str
    parameters: Dict[str, Any]
    priority: str
    created_at: datetime
    deadline: Optional[datetime] = None

@dataclass
class TaskResult:
    """Completed work with status and output artifacts"""
    task_id: str
    agent_id: str
    status: str
    result_data: Dict[str, Any]
    execution_time: float
    completed_at: datetime
    error_message: Optional[str] = None

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR_NOTIFICATION = "error_notification"
    COORDINATION = "coordination"

# Base Agent Class
class BaseAgent(ABC):
    """Abstract base class for all agents in the system"""
    
    def __init__(self, agent_id: str, name: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.status = TaskStatus.PENDING
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.current_task: Optional[TaskRequest] = None
        self.task_history: List[TaskResult] = []
        self.health_status = "healthy"
        
    @abstractmethod
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process a specific task and return results"""
        pass
    
    async def send_message(self, receiver: str, message_type: str, payload: Dict) -> None:
        """Send message to another agent or coordinator"""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=f"{self.agent_id}_{datetime.now().timestamp()}"
        )
        
        logger.info("Agent sending message",
                   sender=self.agent_id,
                   receiver=receiver,
                   message_type=message_type)
        
        # In real implementation, this would go through message broker
        await self.message_queue.put(message)
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive message from queue"""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Return agent health status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.health_status,
            "capabilities": self.capabilities,
            "current_task": self.current_task.task_id if self.current_task else None,
            "completed_tasks": len(self.task_history),
            "queue_size": self.message_queue.qsize()
        }

# Chapter Writer Agent
class ChapterWriterAgent(BaseAgent):
    """Agent responsible for generating chapter content from storylines"""
    
    def __init__(self, agent_id: str = "chapter_writer", anthropic_api_key: Optional[str] = None):
        super().__init__(agent_id, "Chapter Writer", ["content_generation", "narrative_writing"])
        
        # Get settings
        self.settings = get_settings()
        
        # Initialize LLM clients based on provider setting
        self.ai_provider = self.settings.ai.provider.lower()
        self.anthropic_client = None
        self.mistral_client = None
        self.gemini_model = None
        
        if self.ai_provider == "anthropic":
            api_key = anthropic_api_key or self.settings.ai.anthropic_api_key
            if api_key and api_key != "your-anthropic-api-key-here":
                self.anthropic_client = Anthropic(api_key=api_key)
                logger.info("Anthropic client initialized successfully")
            else:
                logger.warning("No valid Anthropic API key provided")
                
        elif self.ai_provider == "mistral":
            api_key = self.settings.ai.mistral_api_key
            if api_key and api_key != "your-mistral-api-key-here":
                self.mistral_client = Mistral(api_key=api_key)
                logger.info("Mistral client initialized successfully")
            else:
                logger.warning("No valid Mistral API key provided")
                
        elif self.ai_provider == "gemini":
            api_key = self.settings.ai.gemini_api_key
            if api_key and api_key != "your-gemini-api-key-here":
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel(self.settings.ai.gemini_model)
                logger.info("Google Gemini client initialized successfully")
            else:
                logger.warning("No valid Google Gemini API key provided")
                
        else:
            logger.warning(f"Unknown AI provider: {self.ai_provider}")
        
        if not self.anthropic_client and not self.mistral_client and not self.gemini_model:
            logger.warning("No AI client available, chapter generation will use templates")
        
        # Writing configuration
        self.target_word_count = 1200
        self.min_word_count = 800
        self.max_word_count = 1800
        self.writing_style = "first_person_memoir"
        
    @log_performance(logger)
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process chapter writing task"""
        start_time = datetime.now()
        self.current_task = task
        self.status = TaskStatus.IN_PROGRESS
        
        try:
            if task.task_type == "write_chapter":
                result = await self._write_chapter(task.parameters)
                status = "completed"
                error_message = None
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
        except Exception as e:
            logger.error("Chapter writing task failed",
                        task_id=task.task_id,
                        error=str(e))
            result = {}
            status = "failed"
            error_message = str(e)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        task_result = TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=status,
            result_data=result,
            execution_time=execution_time,
            completed_at=datetime.now(),
            error_message=error_message
        )
        
        self.task_history.append(task_result)
        self.current_task = None
        self.status = TaskStatus.PENDING
        
        return task_result
    
    async def _write_chapter(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Write a chapter based on storyline and context"""
        storyline_data = parameters.get("storyline")
        context_chunks = parameters.get("context_chunks", [])
        user_preferences = parameters.get("user_preferences", {})
        
        if not storyline_data:
            raise ValueError("No storyline data provided")
        
        # Reconstruct storyline object
        storyline = StorylineNode(**storyline_data)
        
        logger.info("Starting chapter writing",
                   storyline_id=storyline.id,
                   context_chunks_count=len(context_chunks),
                   target_word_count=self.target_word_count)
        
        # Prepare context
        context_text = self._prepare_context_text(context_chunks)
        
        # Generate chapter content
        if self.anthropic_client or self.mistral_client or self.gemini_model:
            chapter_content = await self._generate_with_llm(storyline, context_text, user_preferences)
        else:
            chapter_content = self._generate_with_template(storyline, context_text)
        
        # Post-process content
        processed_content = self._post_process_content(chapter_content)
        
        # Calculate metrics
        word_count = len(processed_content.split())
        quality_score = self._assess_content_quality(processed_content, storyline)
        
        return {
            "storyline_id": storyline.id,
            "title": self._generate_chapter_title(storyline),
            "content": processed_content,
            "word_count": word_count,
            "quality_score": quality_score,
            "source_chunks": [chunk.get("id") for chunk in context_chunks],
            "participants": storyline.participants,
            "themes": storyline.themes,
            "confidence": storyline.confidence,
            "generation_metadata": {
                "method": "llm" if (self.anthropic_client or self.mistral_client or self.gemini_model) else "template",
                "ai_provider": self.ai_provider if (self.anthropic_client or self.mistral_client or self.gemini_model) else "none",
                "target_word_count": self.target_word_count,
                "actual_word_count": word_count,
                "writing_style": self.writing_style,
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _prepare_context_text(self, context_chunks: List[Dict]) -> str:
        """Prepare context text from chunks"""
        if not context_chunks:
            return ""
        
        # Sort chunks by position in transcript
        sorted_chunks = sorted(
            context_chunks,
            key=lambda x: x.get("start_position", 0)
        )
        
        context_parts = []
        for chunk in sorted_chunks:
            content = chunk.get("content", "")
            speaker = chunk.get("speaker")
            
            if speaker:
                context_parts.append(f"[{speaker}]: {content}")
            else:
                context_parts.append(content)
        
        return "\n\n".join(context_parts)
    
    async def _generate_with_llm(self, 
                                storyline: StorylineNode,
                                context_text: str,
                                user_preferences: Dict) -> str:
        """Generate chapter content using configured LLM provider"""
        prompt = self._build_chapter_prompt(storyline, context_text, user_preferences)
        
        try:
            if self.anthropic_client:
                content = await self._generate_with_anthropic(prompt)
            elif self.mistral_client:
                content = await self._generate_with_mistral(prompt)
            elif self.gemini_model:
                content = await self._generate_with_gemini(prompt)
            else:
                raise ValueError("No AI client available")
            
            logger.info("LLM chapter generation completed",
                       provider=self.ai_provider,
                       input_tokens=len(prompt.split()),
                       output_tokens=len(content.split()))
            
            return content
            
        except Exception as e:
            logger.error("LLM generation failed, falling back to template", 
                        provider=self.ai_provider, error=str(e))
            return self._generate_with_template(storyline, context_text)
    
    async def _generate_with_anthropic(self, prompt: str) -> str:
        """Generate content using Anthropic Claude"""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.anthropic_client.messages.create(
                model=self.settings.ai.anthropic_model,
                max_tokens=self.settings.ai.anthropic_max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return response.content[0].text
    
    async def _generate_with_mistral(self, prompt: str) -> str:
        """Generate content using Mistral AI"""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.mistral_client.chat.complete(
                model=self.settings.ai.mistral_model,
                max_tokens=self.settings.ai.mistral_max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return response.choices[0].message.content
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate content using Google Gemini"""
        
        # Configure generation settings
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.settings.ai.gemini_max_tokens,
            temperature=self.settings.ai.gemini_temperature,
        )
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
        )
        
        if response.parts:
            return response.text
        else:
            # Handle safety filters or blocked content
            if hasattr(response, 'prompt_feedback'):
                feedback = response.prompt_feedback
                if feedback.block_reason:
                    raise ValueError(f"Content blocked by Gemini safety filters: {feedback.block_reason}")
            raise ValueError("Gemini did not generate any content")
    
    def _build_chapter_prompt(self, 
                             storyline: StorylineNode,
                             context_text: str,
                             user_preferences: Dict) -> str:
        """Build prompt for LLM chapter generation"""
        
        writing_style = user_preferences.get("writing_style", "memoir")
        tone = user_preferences.get("tone", "reflective")
        perspective = user_preferences.get("perspective", "first_person")
        
        prompt = f"""
Write a chapter for a memoir book based on the following storyline and source material.

STORYLINE OVERVIEW:
{storyline.summary}

MAIN PARTICIPANTS: {', '.join(storyline.participants) if storyline.participants else 'Various people'}

THEMES: {', '.join(storyline.themes) if storyline.themes else 'Personal experiences'}

SOURCE MATERIAL:
{context_text}

WRITING REQUIREMENTS:
- Style: {writing_style} with {tone} tone
- Perspective: {perspective}
- Target length: {self.target_word_count} words
- Maintain authenticity to the source material
- Create natural narrative flow
- Include specific details and dialogue where appropriate
- Use vivid, engaging prose
- Preserve the emotional tone of the original recordings
- Write in complete, well-structured paragraphs
- Ensure the content is appropriate and family-friendly

STRUCTURE:
- Start with a compelling opening that sets the scene
- Develop the story chronologically where possible
- Include character development and relationships
- End with reflection or transition to next chapter

Write the chapter now:
"""
        return prompt
    
    def _generate_with_template(self, storyline: StorylineNode, context_text: str) -> str:
        """Generate chapter content using template-based approach"""
        logger.info("Using template-based generation")
        
        title = self._generate_chapter_title(storyline)
        
        # Simple template-based generation
        template = f"""
        # {title}
        
        {self._create_opening_paragraph(storyline)}
        
        {self._process_context_into_narrative(context_text, storyline)}
        
        {self._create_closing_paragraph(storyline)}
        """
        
        return template.strip()
    
    def _generate_chapter_title(self, storyline: StorylineNode) -> str:
        """Generate appropriate chapter title"""
        if storyline.participants:
            if len(storyline.participants) == 1:
                return f"Memories of {storyline.participants[0]}"
            else:
                return f"Together with {storyline.participants[0]} and Others"
        elif storyline.themes:
            theme = storyline.themes[0].replace('_', ' ').title()
            return f"Chapter on {theme}"
        else:
            return f"Memories and Reflections"
    
    def _create_opening_paragraph(self, storyline: StorylineNode) -> str:
        """Create engaging opening paragraph"""
        if storyline.participants:
            return f"When I think about {storyline.participants[0]}, so many memories come flooding back. The stories we shared, the experiences we had together - they all form an important part of my life's tapestry."
        else:
            return "Looking back on this period of my life, I'm struck by how certain moments and experiences shaped who I became."
    
    def _process_context_into_narrative(self, context_text: str, storyline: StorylineNode) -> str:
        """Convert context chunks into narrative prose"""
        if not context_text:
            return "The details of these experiences remain vivid in my memory, each one contributing to the larger story of my life."
        
        # Simple processing - in real implementation would be more sophisticated
        paragraphs = context_text.split('\n\n')
        narrative_parts = []
        
        for paragraph in paragraphs[:5]:  # Limit to prevent overly long chapters
            if paragraph.strip():
                # Remove speaker indicators
                cleaned = re.sub(r'^\[.*?\]:\s*', '', paragraph.strip())
                if cleaned:
                    narrative_parts.append(f"I remember {cleaned.lower()}")
        
        return '\n\n'.join(narrative_parts)
    
    def _create_closing_paragraph(self, storyline: StorylineNode) -> str:
        """Create reflective closing paragraph"""
        return "These memories, preserved now in these words, remind me of the richness of human experience and the importance of sharing our stories with those we love."
    
    def _post_process_content(self, content: str) -> str:
        """Post-process generated content for quality"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Ensure proper paragraph separation
        content = re.sub(r'\n(?=[A-Z])', '\n\n', content)
        
        # Fix common formatting issues
        content = content.replace('  ', ' ')  # Double spaces
        content = content.strip()
        
        return content
    
    def _assess_content_quality(self, content: str, storyline: StorylineNode) -> float:
        """Assess quality of generated content"""
        quality_factors = []
        
        # Word count appropriateness
        word_count = len(content.split())
        if self.min_word_count <= word_count <= self.max_word_count:
            quality_factors.append(1.0)
        else:
            deviation = abs(word_count - self.target_word_count) / self.target_word_count
            quality_factors.append(max(0.0, 1.0 - deviation))
        
        # Presence of participants
        participant_mentions = 0
        for participant in storyline.participants:
            if participant.lower() in content.lower():
                participant_mentions += 1
        
        if storyline.participants:
            participant_score = participant_mentions / len(storyline.participants)
        else:
            participant_score = 0.5  # Neutral score if no participants
        
        quality_factors.append(min(participant_score, 1.0))
        
        # Structural completeness (has beginning, middle, end)
        paragraphs = content.split('\n\n')
        structure_score = min(len(paragraphs) / 3.0, 1.0)  # Expect at least 3 paragraphs
        quality_factors.append(structure_score)
        
        # Overall confidence
        quality_factors.append(storyline.confidence)
        
        return sum(quality_factors) / len(quality_factors)

# Chapter Harmonizer Agent
class ChapterHarmonizerAgent(BaseAgent):
    """Agent responsible for ensuring consistency across chapters"""
    
    def __init__(self, agent_id: str = "chapter_harmonizer"):
        super().__init__(agent_id, "Chapter Harmonizer", 
                        ["consistency_checking", "conflict_resolution", "style_harmonization"])
        
        # Harmonization configuration
        self.consistency_threshold = 0.7
        self.conflict_detection_enabled = True
        self.style_normalization_enabled = True
        
    @log_performance(logger)
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process harmonization task"""
        start_time = datetime.now()
        self.current_task = task
        self.status = TaskStatus.IN_PROGRESS
        
        try:
            if task.task_type == "harmonize_chapters":
                result = await self._harmonize_chapters(task.parameters)
                status = "completed"
                error_message = None
            elif task.task_type == "detect_conflicts":
                result = await self._detect_conflicts(task.parameters)
                status = "completed"
                error_message = None
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
        except Exception as e:
            logger.error("Harmonization task failed",
                        task_id=task.task_id,
                        error=str(e))
            result = {}
            status = "failed"
            error_message = str(e)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        task_result = TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=status,
            result_data=result,
            execution_time=execution_time,
            completed_at=datetime.now(),
            error_message=error_message
        )
        
        self.task_history.append(task_result)
        self.current_task = None
        self.status = TaskStatus.PENDING
        
        return task_result
    
    async def _harmonize_chapters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Harmonize multiple chapters for consistency"""
        chapters = parameters.get("chapters", [])
        if not chapters:
            raise ValueError("No chapters provided for harmonization")
        
        logger.info("Starting chapter harmonization", chapter_count=len(chapters))
        
        # Detect conflicts
        conflicts = await self._analyze_all_conflicts(chapters)
        
        # Apply harmonization
        harmonized_chapters = []
        total_changes = 0
        
        for chapter in chapters:
            chapter_conflicts = [c for c in conflicts if chapter["storyline_id"] in c.get("affected_chapters", [])]
            
            if chapter_conflicts:
                harmonized_chapter = await self._apply_harmonization(chapter, chapter_conflicts)
                total_changes += harmonized_chapter.get("changes_applied", 0)
            else:
                harmonized_chapter = chapter
                harmonized_chapter["harmonization_applied"] = False
                harmonized_chapter["changes_applied"] = 0
            
            harmonized_chapters.append(harmonized_chapter)
        
        return {
            "harmonized_chapters": harmonized_chapters,
            "conflicts_detected": len(conflicts),
            "total_changes_applied": total_changes,
            "harmonization_summary": {
                "character_consistency_issues": len([c for c in conflicts if c["type"] == "character_inconsistency"]),
                "timeline_conflicts": len([c for c in conflicts if c["type"] == "timeline_conflict"]),
                "style_inconsistencies": len([c for c in conflicts if c["type"] == "style_inconsistency"]),
                "factual_contradictions": len([c for c in conflicts if c["type"] == "factual_contradiction"])
            }
        }
    
    async def _detect_conflicts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect conflicts between chapters without applying fixes"""
        chapters = parameters.get("chapters", [])
        conflicts = await self._analyze_all_conflicts(chapters)
        
        return {
            "conflicts": conflicts,
            "conflict_summary": {
                "total_conflicts": len(conflicts),
                "severity_distribution": self._analyze_conflict_severity(conflicts)
            }
        }
    
    async def _analyze_all_conflicts(self, chapters: List[Dict]) -> List[Dict]:
        """Analyze all types of conflicts between chapters"""
        conflicts = []
        
        # Character consistency conflicts
        conflicts.extend(await self._detect_character_conflicts(chapters))
        
        # Timeline conflicts
        conflicts.extend(await self._detect_timeline_conflicts(chapters))
        
        # Style inconsistencies
        conflicts.extend(await self._detect_style_conflicts(chapters))
        
        # Factual contradictions
        conflicts.extend(await self._detect_factual_conflicts(chapters))
        
        return conflicts
    
    async def _detect_character_conflicts(self, chapters: List[Dict]) -> List[Dict]:
        """Detect character portrayal inconsistencies"""
        conflicts = []
        character_portrayals = {}
        
        # Collect character information from each chapter
        for chapter in chapters:
            participants = chapter.get("participants", [])
            content = chapter.get("content", "")
            
            for participant in participants:
                if participant not in character_portrayals:
                    character_portrayals[participant] = []
                
                # Extract character-related sentences
                sentences = content.split('.')
                character_sentences = [
                    s.strip() for s in sentences 
                    if participant.lower() in s.lower()
                ]
                
                character_portrayals[participant].extend([{
                    "chapter_id": chapter["storyline_id"],
                    "sentences": character_sentences
                }])
        
        # Analyze for conflicts (simple heuristic)
        for character, portrayals in character_portrayals.items():
            if len(portrayals) > 1:
                # Check for contradictory descriptions
                all_sentences = []
                for portrayal in portrayals:
                    all_sentences.extend(portrayal["sentences"])
                
                # Simple conflict detection based on contradictory words
                contradictory_pairs = [
                    ("kind", "cruel"), ("young", "old"), ("tall", "short"),
                    ("happy", "sad"), ("friendly", "hostile")
                ]
                
                for pos_word, neg_word in contradictory_pairs:
                    pos_count = sum(1 for s in all_sentences if pos_word in s.lower())
                    neg_count = sum(1 for s in all_sentences if neg_word in s.lower())
                    
                    if pos_count > 0 and neg_count > 0:
                        conflicts.append({
                            "type": "character_inconsistency",
                            "character": character,
                            "description": f"Contradictory descriptions: {pos_word} vs {neg_word}",
                            "severity": "medium",
                            "affected_chapters": [p["chapter_id"] for p in portrayals],
                            "evidence": {
                                "positive": [s for s in all_sentences if pos_word in s.lower()],
                                "negative": [s for s in all_sentences if neg_word in s.lower()]
                            }
                        })
        
        return conflicts
    
    async def _detect_timeline_conflicts(self, chapters: List[Dict]) -> List[Dict]:
        """Detect chronological inconsistencies"""
        conflicts = []
        
        # Simple timeline conflict detection
        temporal_references = []
        for chapter in chapters:
            content = chapter.get("content", "")
            storyline_id = chapter["storyline_id"]
            
            # Look for temporal markers
            temporal_patterns = [
                r'\b(before|after|during)\s+\w+',
                r'\b(first|then|later|finally)\b',
                r'\b\d{4}\b',  # Years
                r'\b(yesterday|today|tomorrow)\b'
            ]
            
            for pattern in temporal_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    temporal_references.append({
                        "chapter_id": storyline_id,
                        "text": match.group(),
                        "position": match.start(),
                        "pattern": pattern
                    })
        
        # Analyze for conflicts (simplified)
        if len(temporal_references) > 1:
            # Look for obvious contradictions
            year_references = [ref for ref in temporal_references if re.match(r'\b\d{4}\b', ref["text"])]
            if len(set([ref["text"] for ref in year_references])) > 1:
                conflicts.append({
                    "type": "timeline_conflict",
                    "description": "Multiple different years mentioned",
                    "severity": "high",
                    "affected_chapters": list(set([ref["chapter_id"] for ref in year_references])),
                    "evidence": year_references
                })
        
        return conflicts
    
    async def _detect_style_conflicts(self, chapters: List[Dict]) -> List[Dict]:
        """Detect writing style inconsistencies"""
        conflicts = []
        
        # Analyze style metrics for each chapter
        style_metrics = []
        for chapter in chapters:
            content = chapter.get("content", "")
            metrics = self._calculate_style_metrics(content)
            metrics["chapter_id"] = chapter["storyline_id"]
            style_metrics.append(metrics)
        
        # Check for significant variations
        if len(style_metrics) > 1:
            avg_sentence_length = np.mean([m["avg_sentence_length"] for m in style_metrics])
            sentence_length_std = np.std([m["avg_sentence_length"] for m in style_metrics])
            
            # Flag chapters with significantly different sentence lengths
            threshold = avg_sentence_length * 0.5  # 50% deviation
            outliers = [
                m for m in style_metrics 
                if abs(m["avg_sentence_length"] - avg_sentence_length) > threshold
            ]
            
            if outliers:
                conflicts.append({
                    "type": "style_inconsistency",
                    "description": "Significant variation in sentence length",
                    "severity": "low",
                    "affected_chapters": [m["chapter_id"] for m in outliers],
                    "evidence": {
                        "average_length": avg_sentence_length,
                        "outliers": outliers
                    }
                })
        
        return conflicts
    
    def _calculate_style_metrics(self, content: str) -> Dict[str, float]:
        """Calculate style metrics for text"""
        sentences = content.split('.')
        words = content.split()
        
        return {
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0
        }
    
    async def _detect_factual_conflicts(self, chapters: List[Dict]) -> List[Dict]:
        """Detect factual contradictions between chapters"""
        conflicts = []
        
        # Simple factual conflict detection
        # In a real implementation, this would be much more sophisticated
        fact_patterns = [
            r'\b(was|is|were|are)\s+\w+',  # Simple statements
            r'\b(born|died|married|divorced)\s+in\s+\d{4}',  # Life events
            r'\b(lived|worked|studied)\s+in\s+\w+'  # Location statements
        ]
        
        extracted_facts = []
        for chapter in chapters:
            content = chapter.get("content", "")
            chapter_id = chapter["storyline_id"]
            
            for pattern in fact_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    extracted_facts.append({
                        "chapter_id": chapter_id,
                        "fact": match.group(),
                        "context": content[max(0, match.start()-50):match.end()+50]
                    })
        
        # Group facts by similarity and check for contradictions
        # This is a simplified implementation
        fact_groups = {}
        for fact in extracted_facts:
            key = fact["fact"].split()[0]  # Group by first word
            if key not in fact_groups:
                fact_groups[key] = []
            fact_groups[key].append(fact)
        
        for key, facts in fact_groups.items():
            if len(facts) > 1:
                unique_facts = set([f["fact"] for f in facts])
                if len(unique_facts) > 1:
                    conflicts.append({
                        "type": "factual_contradiction",
                        "description": f"Conflicting statements about {key}",
                        "severity": "medium",
                        "affected_chapters": list(set([f["chapter_id"] for f in facts])),
                        "evidence": facts
                    })
        
        return conflicts
    
    async def _apply_harmonization(self, chapter: Dict, conflicts: List[Dict]) -> Dict:
        """Apply harmonization fixes to a chapter"""
        harmonized_chapter = chapter.copy()
        changes_applied = 0
        
        content = harmonized_chapter.get("content", "")
        
        for conflict in conflicts:
            if conflict["severity"] in ["low", "medium"]:
                # Apply automatic fixes for low/medium severity conflicts
                if conflict["type"] == "style_inconsistency":
                    # Simple style normalization
                    content = self._normalize_style(content)
                    changes_applied += 1
                elif conflict["type"] == "character_inconsistency":
                    # Simple character name normalization
                    content = self._normalize_character_references(content, conflict)
                    changes_applied += 1
        
        harmonized_chapter["content"] = content
        harmonized_chapter["harmonization_applied"] = changes_applied > 0
        harmonized_chapter["changes_applied"] = changes_applied
        harmonized_chapter["conflicts_addressed"] = [c["type"] for c in conflicts]
        
        return harmonized_chapter
    
    def _normalize_style(self, content: str) -> str:
        """Apply basic style normalization"""
        # Ensure consistent sentence spacing
        content = re.sub(r'\.\s*', '. ', content)
        
        # Normalize paragraph breaks
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Remove excessive punctuation
        content = re.sub(r'[!]{2,}', '!', content)
        content = re.sub(r'[?]{2,}', '?', content)
        
        return content.strip()
    
    def _normalize_character_references(self, content: str, conflict: Dict) -> str:
        """Normalize character name references"""
        character = conflict.get("character", "")
        if not character:
            return content
        
        # Simple normalization - ensure consistent capitalization
        # In real implementation, would be more sophisticated
        content = re.sub(f'\\b{character.lower()}\\b', character, content, flags=re.IGNORECASE)
        
        return content
    
    def _analyze_conflict_severity(self, conflicts: List[Dict]) -> Dict[str, int]:
        """Analyze severity distribution of conflicts"""
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        
        for conflict in conflicts:
            severity = conflict.get("severity", "medium")
            severity_counts[severity] += 1
        
        return severity_counts

# Follow-up Questions Agent
class FollowupQuestionsAgent(BaseAgent):
    """Agent responsible for generating follow-up questions to fill information gaps"""
    
    def __init__(self, agent_id: str = "followup_questions"):
        super().__init__(agent_id, "Follow-up Questions", 
                        ["gap_analysis", "question_generation", "information_prioritization"])
        
        # Question generation configuration
        self.max_questions_per_storyline = 3
        self.priority_threshold = 0.6
        self.question_categories = ["temporal", "factual", "emotional", "contextual", "relationship"]
        
    @log_performance(logger)
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process question generation task"""
        start_time = datetime.now()
        self.current_task = task
        self.status = TaskStatus.IN_PROGRESS
        
        try:
            if task.task_type == "generate_questions":
                result = await self._generate_questions(task.parameters)
                status = "completed"
                error_message = None
            elif task.task_type == "analyze_gaps":
                result = await self._analyze_information_gaps(task.parameters)
                status = "completed"
                error_message = None
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
        except Exception as e:
            logger.error("Question generation task failed",
                        task_id=task.task_id,
                        error=str(e))
            result = {}
            status = "failed"
            error_message = str(e)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        task_result = TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=status,
            result_data=result,
            execution_time=execution_time,
            completed_at=datetime.now(),
            error_message=error_message
        )
        
        self.task_history.append(task_result)
        self.current_task = None
        self.status = TaskStatus.PENDING
        
        return task_result
    
    async def _generate_questions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate follow-up questions based on storylines and graph analysis"""
        storylines = parameters.get("storylines", [])
        graph_data = parameters.get("graph_data", {})
        
        logger.info("Starting question generation", storyline_count=len(storylines))
        
        all_questions = []
        
        # Generate questions for each storyline
        for storyline_data in storylines:
            storyline = StorylineNode(**storyline_data)
            storyline_questions = await self._generate_storyline_questions(storyline)
            all_questions.extend(storyline_questions)
        
        # Generate graph-based questions
        if graph_data:
            graph_questions = await self._generate_graph_questions(graph_data)
            all_questions.extend(graph_questions)
        
        # Prioritize and filter questions
        prioritized_questions = self._prioritize_questions(all_questions)
        
        # Categorize questions
        categorized_questions = self._categorize_questions(prioritized_questions)
        
        return {
            "questions": prioritized_questions,
            "categorized_questions": categorized_questions,
            "generation_summary": {
                "total_questions": len(all_questions),
                "prioritized_questions": len(prioritized_questions),
                "categories": list(categorized_questions.keys()),
                "avg_priority_score": np.mean([q["priority_score"] for q in prioritized_questions])
            }
        }
    
    async def _generate_storyline_questions(self, storyline: StorylineNode) -> List[Dict[str, Any]]:
        """Generate questions specific to a storyline"""
        questions = []
        
        # Temporal questions
        if not storyline.temporal_info or not storyline.temporal_info.get("explicit_dates"):
            questions.append({
                "id": f"temporal_{storyline.id}",
                "storyline_id": storyline.id,
                "category": "temporal",
                "question": f"When did the events involving {', '.join(storyline.participants[:2]) if storyline.participants else 'these experiences'} take place?",
                "context": storyline.summary,
                "priority_score": 0.8,
                "reasoning": "Missing temporal information affects chronological organization"
            })
        
        # Participant questions
        if not storyline.participants:
            questions.append({
                "id": f"participants_{storyline.id}",
                "storyline_id": storyline.id,
                "category": "factual",
                "question": f"Who else was involved in these events? Can you provide more details about the people mentioned?",
                "context": storyline.summary,
                "priority_score": 0.7,
                "reasoning": "Missing participant information affects relationship mapping"
            })
        
        # Detail expansion questions
        if storyline.confidence < 0.7:
            questions.append({
                "id": f"details_{storyline.id}",
                "storyline_id": storyline.id,
                "category": "contextual",
                "question": f"Can you provide more details about '{storyline.summary}'? What specific events or conversations do you remember?",
                "context": storyline.summary,
                "priority_score": 0.6,
                "reasoning": "Low confidence indicates need for additional detail"
            })
        
        # Emotional context questions
        if storyline.themes and "emotional" not in [theme.lower() for theme in storyline.themes]:
            questions.append({
                "id": f"emotional_{storyline.id}",
                "storyline_id": storyline.id,
                "category": "emotional",
                "question": f"How did you feel during these events? What emotions do you associate with this time?",
                "context": storyline.summary,
                "priority_score": 0.5,
                "reasoning": "Emotional context enriches narrative depth"
            })
        
        return questions[:self.max_questions_per_storyline]
    
    async def _generate_graph_questions(self, graph_data: Dict) -> List[Dict[str, Any]]:
        """Generate questions based on graph structure analysis"""
        questions = []
        
        # Questions about weak connections
        weak_connections = graph_data.get("weak_connections", [])
        for i, connection in enumerate(weak_connections[:3]):  # Limit to top 3
            questions.append({
                "id": f"connection_{i}",
                "storyline_id": None,
                "category": "relationship",
                "question": f"How are these different parts of your story connected? Are there relationships or events that link them together?",
                "context": f"Connection between storylines: {connection}",
                "priority_score": 0.6,
                "reasoning": "Weak connections indicate potential missing relationships"
            })
        
        # Questions about isolated storylines
        isolated_storylines = graph_data.get("isolated_storylines", [])
        for storyline_id in isolated_storylines[:2]:  # Limit to top 2
            questions.append({
                "id": f"isolation_{storyline_id}",
                "storyline_id": storyline_id,
                "category": "contextual",
                "question": f"How does this experience relate to other parts of your life story? Are there connections we might be missing?",
                "context": f"Isolated storyline: {storyline_id}",
                "priority_score": 0.5,
                "reasoning": "Isolated storylines may have missing connections"
            })
        
        return questions
    
    def _prioritize_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize questions based on importance and impact"""
        # Sort by priority score
        sorted_questions = sorted(questions, key=lambda q: q["priority_score"], reverse=True)
        
        # Filter by threshold
        high_priority = [q for q in sorted_questions if q["priority_score"] >= self.priority_threshold]
        
        # Ensure diversity across categories
        category_counts = {}
        balanced_questions = []
        
        for question in high_priority:
            category = question["category"]
            if category_counts.get(category, 0) < 3:  # Max 3 per category
                balanced_questions.append(question)
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return balanced_questions
    
    def _categorize_questions(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Organize questions by category"""
        categorized = {}
        
        for question in questions:
            category = question["category"]
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(question)
        
        return categorized
    
    async def _analyze_information_gaps(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze information gaps without generating questions"""
        storylines = parameters.get("storylines", [])
        
        gap_analysis = {
            "temporal_gaps": 0,
            "participant_gaps": 0,
            "detail_gaps": 0,
            "emotional_gaps": 0,
            "connection_gaps": 0,
            "total_storylines": len(storylines),
            "completeness_score": 0.0
        }
        
        for storyline_data in storylines:
            storyline = StorylineNode(**storyline_data)
            
            # Check for various types of gaps
            if not storyline.temporal_info or not storyline.temporal_info.get("explicit_dates"):
                gap_analysis["temporal_gaps"] += 1
            
            if not storyline.participants:
                gap_analysis["participant_gaps"] += 1
            
            if storyline.confidence < 0.7:
                gap_analysis["detail_gaps"] += 1
            
            if not storyline.themes or "emotional" not in [t.lower() for t in storyline.themes]:
                gap_analysis["emotional_gaps"] += 1
        
        # Calculate overall completeness score
        total_possible_gaps = len(storylines) * 4  # 4 types of gaps per storyline
        total_actual_gaps = (gap_analysis["temporal_gaps"] + 
                           gap_analysis["participant_gaps"] + 
                           gap_analysis["detail_gaps"] + 
                           gap_analysis["emotional_gaps"])
        
        gap_analysis["completeness_score"] = 1.0 - (total_actual_gaps / max(total_possible_gaps, 1))
        
        return gap_analysis

# Global agent instances with settings integration
def get_chapter_writer_agent() -> ChapterWriterAgent:
    """Get a properly configured chapter writer agent"""
    return ChapterWriterAgent()

def get_chapter_harmonizer_agent() -> ChapterHarmonizerAgent:
    """Get a properly configured chapter harmonizer agent"""
    return ChapterHarmonizerAgent()

def get_followup_questions_agent() -> FollowupQuestionsAgent:
    """Get a properly configured followup questions agent"""
    return FollowupQuestionsAgent()

# Legacy global instances for backward compatibility
chapter_writer_agent = get_chapter_writer_agent()
chapter_harmonizer_agent = get_chapter_harmonizer_agent()
followup_questions_agent = get_followup_questions_agent()