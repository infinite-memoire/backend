# Chat Interface Design for Follow-up Questions

## Interface Architecture

### Chat Component Structure
```typescript
interface ChatMessage {
  id: string;
  type: 'question' | 'answer' | 'system' | 'validation';
  content: string;
  timestamp: Date;
  sender: 'ai' | 'user' | 'system';
  metadata?: {
    question_id?: string;
    category?: string;
    priority_score?: number;
    context?: string;
    validation_status?: 'valid' | 'invalid' | 'warning';
  };
}

interface ChatSession {
  session_id: string;
  book_id: string;
  user_id: string;
  messages: ChatMessage[];
  active_questions: FollowupQuestion[];
  answered_questions: AnsweredQuestion[];
  created_at: Date;
  updated_at: Date;
  status: 'active' | 'completed' | 'paused';
}

interface FollowupQuestion {
  id: string;
  category: 'temporal' | 'character' | 'setting' | 'clarification' | 'detail';
  question: string;
  context: string;
  priority_score: number;
  reasoning: string;
  storyline_id?: string;
  chapter_references: string[];
  suggested_answers?: string[];
  is_required: boolean;
}
```

### Chat Interface Components

#### Main Chat Container
```tsx
import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage, ChatSession, FollowupQuestion } from './types';

const ChatInterface: React.FC<{
  bookId: string;
  sessionId: string;
  onAnswerSubmit: (questionId: string, answer: string, confidence: number) => void;
}> = ({ bookId, sessionId, onAnswerSubmit }) => {
  const [chatSession, setChatSession] = useState<ChatSession | null>(null);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [activeQuestion, setActiveQuestion] = useState<FollowupQuestion | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatSession?.messages]);

  return (
    <div className="chat-interface">
      <ChatHeader session={chatSession} />
      <MessagesList messages={chatSession?.messages || []} />
      <QuestionPrompt 
        question={activeQuestion}
        onAnswer={handleAnswerSubmit}
      />
      <ChatInput 
        onSendMessage={handleSendMessage}
        disabled={!activeQuestion}
      />
      <div ref={messagesEndRef} />
    </div>
  );
};
```

#### Question Presentation Component
```tsx
const QuestionPrompt: React.FC<{
  question: FollowupQuestion | null;
  onAnswer: (answer: string, confidence: number) => void;
}> = ({ question, onAnswer }) => {
  const [answer, setAnswer] = useState('');
  const [confidence, setConfidence] = useState(0.8);
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!question) return null;

  const handleSubmit = async () => {
    setIsSubmitting(true);
    await onAnswer(answer, confidence);
    setAnswer('');
    setIsSubmitting(false);
  };

  return (
    <div className="question-prompt">
      <div className="question-header">
        <span className="category-badge">{question.category}</span>
        <span className="priority-indicator">
          {getPriorityLevel(question.priority_score)}
        </span>
      </div>
      
      <div className="question-content">
        <h3>{question.question}</h3>
        {question.context && (
          <div className="context-section">
            <h4>Context:</h4>
            <p>{question.context}</p>
          </div>
        )}
        {question.reasoning && (
          <div className="reasoning-section">
            <details>
              <summary>Why we're asking this</summary>
              <p>{question.reasoning}</p>
            </details>
          </div>
        )}
      </div>

      <div className="answer-section">
        {question.suggested_answers && (
          <div className="suggested-answers">
            <h4>Quick options:</h4>
            {question.suggested_answers.map((suggestion, index) => (
              <button 
                key={index}
                className="suggestion-button"
                onClick={() => setAnswer(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}
        
        <textarea
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          placeholder="Your answer..."
          className="answer-input"
          rows={4}
        />
        
        <div className="confidence-slider">
          <label>How confident are you in this answer?</label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
          />
          <span>{Math.round(confidence * 100)}%</span>
        </div>

        <div className="action-buttons">
          <button 
            onClick={handleSubmit}
            disabled={!answer.trim() || isSubmitting}
            className="submit-button primary"
          >
            {isSubmitting ? 'Submitting...' : 'Submit Answer'}
          </button>
          <button 
            onClick={() => onAnswer('', 0)}
            className="skip-button secondary"
          >
            Skip for now
          </button>
        </div>
      </div>
    </div>
  );
};
```

#### Message Display Component
```tsx
const MessagesList: React.FC<{ messages: ChatMessage[] }> = ({ messages }) => {
  return (
    <div className="messages-list">
      {messages.map((message) => (
        <MessageItem key={message.id} message={message} />
      ))}
    </div>
  );
};

const MessageItem: React.FC<{ message: ChatMessage }> = ({ message }) => {
  const getMessageClass = () => {
    const baseClass = 'message-item';
    return `${baseClass} ${message.type} ${message.sender}`;
  };

  const renderMessageContent = () => {
    switch (message.type) {
      case 'question':
        return <QuestionMessage message={message} />;
      case 'answer':
        return <AnswerMessage message={message} />;
      case 'system':
        return <SystemMessage message={message} />;
      case 'validation':
        return <ValidationMessage message={message} />;
      default:
        return <span>{message.content}</span>;
    }
  };

  return (
    <div className={getMessageClass()}>
      <div className="message-header">
        <span className="sender">{message.sender}</span>
        <span className="timestamp">
          {formatTimestamp(message.timestamp)}
        </span>
      </div>
      <div className="message-content">
        {renderMessageContent()}
      </div>
    </div>
  );
};
```

## Backend API Integration

### Chat Session Management
```python
from fastapi import APIRouter, WebSocket, HTTPException
from typing import List, Dict, Any
import asyncio
import json

router = APIRouter(tags=["Chat Interface"])

class ChatSessionManager:
    def __init__(self, storage_service, question_service):
        self.storage = storage_service
        self.questions = question_service
        self.active_sessions = {}  # session_id -> WebSocket connection
        
    async def create_chat_session(self, book_id: str, user_id: str) -> str:
        """Create a new chat session for follow-up questions"""
        
        # Get pending questions for this book
        pending_questions = await self.questions.get_pending_questions(book_id)
        
        # Create chat session
        session = ChatSession(
            session_id=f"chat_{book_id}_{int(time.time())}",
            book_id=book_id,
            user_id=user_id,
            messages=[],
            active_questions=pending_questions,
            answered_questions=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="active"
        )
        
        await self.storage.store_chat_session(session)
        
        # Send welcome message and first question
        await self._send_welcome_message(session)
        await self._send_next_question(session)
        
        return session.session_id
    
    async def _send_welcome_message(self, session: ChatSession):
        """Send initial welcome message"""
        
        welcome_msg = ChatMessage(
            id=f"msg_{int(time.time())}_welcome",
            type="system",
            content=f"Hi! I have some questions about your book to help improve the content. "
                   f"There are {len(session.active_questions)} questions total. "
                   f"Let's start with the most important ones first.",
            timestamp=datetime.now(),
            sender="ai"
        )
        
        session.messages.append(welcome_msg)
        await self.storage.update_chat_session(session)
    
    async def _send_next_question(self, session: ChatSession):
        """Send the next highest priority question"""
        
        if not session.active_questions:
            await self._send_completion_message(session)
            return
            
        # Sort by priority and get next question
        next_question = max(session.active_questions, 
                          key=lambda q: q.priority_score)
        
        question_msg = ChatMessage(
            id=f"msg_{int(time.time())}_question",
            type="question",
            content=next_question.question,
            timestamp=datetime.now(),
            sender="ai",
            metadata={
                "question_id": next_question.id,
                "category": next_question.category,
                "priority_score": next_question.priority_score,
                "context": next_question.context
            }
        )
        
        session.messages.append(question_msg)
        await self.storage.update_chat_session(session)

@router.websocket("/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """WebSocket connection for real-time chat"""
    
    await websocket.accept()
    chat_manager.active_sessions[session_id] = websocket
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process the message
            await process_chat_message(session_id, message_data)
            
    except Exception as e:
        logger.error(f"Chat WebSocket error: {e}")
    finally:
        # Clean up connection
        if session_id in chat_manager.active_sessions:
            del chat_manager.active_sessions[session_id]

async def process_chat_message(session_id: str, message_data: dict):
    """Process incoming chat message"""
    
    if message_data["type"] == "answer":
        await process_answer_submission(session_id, message_data)
    elif message_data["type"] == "skip":
        await process_question_skip(session_id, message_data)
```

### Question Processing Service
```python
class QuestionProcessingService:
    def __init__(self, storage_service, orchestrator):
        self.storage = storage_service
        self.orchestrator = orchestrator
    
    async def process_answer(self, session_id: str, question_id: str, 
                           answer: str, confidence: float) -> dict:
        """Process user answer and determine next steps"""
        
        # Store the answer
        answer_record = {
            "session_id": session_id,
            "question_id": question_id,
            "answer": answer,
            "confidence": confidence,
            "answered_at": datetime.now(),
            "processing_status": "pending"
        }
        
        await self.storage.store_question_answer(answer_record)
        
        # Get question context to determine impact
        question = await self.storage.get_question(question_id)
        impact_scope = await self._analyze_answer_impact(question, answer)
        
        # Update chat session
        chat_session = await self.storage.get_chat_session(session_id)
        
        # Move question from active to answered
        chat_session.active_questions = [
            q for q in chat_session.active_questions if q.id != question_id
        ]
        chat_session.answered_questions.append(question)
        
        # Send confirmation message
        confirmation_msg = ChatMessage(
            id=f"msg_{int(time.time())}_confirmation",
            type="answer",
            content=f"Thanks for your answer! {self._get_confirmation_message(impact_scope)}",
            timestamp=datetime.now(),
            sender="ai",
            metadata={"question_id": question_id}
        )
        
        chat_session.messages.append(confirmation_msg)
        await self.storage.update_chat_session(chat_session)
        
        # Send next question or completion message
        await self._send_next_question(chat_session)
        
        return {
            "status": "processed",
            "impact_scope": impact_scope,
            "next_question_available": len(chat_session.active_questions) > 0
        }
    
    def _get_confirmation_message(self, impact_scope: dict) -> str:
        """Generate appropriate confirmation message based on impact"""
        
        if impact_scope["reprocessing_required"]:
            return "This will help improve the relevant chapters. I'll update the content based on your answer."
        elif impact_scope["affects_multiple_chapters"]:
            return "This information will help maintain consistency across multiple chapters."
        else:
            return "This helps clarify the details in your story."
```

## User Experience Features

### Progressive Question Presentation
```typescript
const useQuestionProgression = (sessionId: string) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [questionsCompleted, setQuestionsCompleted] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  
  const progressPercentage = totalQuestions > 0 
    ? (questionsCompleted / totalQuestions) * 100 
    : 0;
    
  return {
    currentQuestionIndex,
    questionsCompleted,
    totalQuestions,
    progressPercentage,
    moveToNext: () => setCurrentQuestionIndex(prev => prev + 1),
    markCompleted: () => setQuestionsCompleted(prev => prev + 1)
  };
};

const ProgressIndicator: React.FC<{ 
  completed: number; 
  total: number; 
  category?: string 
}> = ({ completed, total, category }) => {
  return (
    <div className="progress-indicator">
      <div className="progress-header">
        <span>Questions Progress</span>
        {category && <span className="category">{category}</span>}
      </div>
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${(completed / total) * 100}%` }}
        />
      </div>
      <div className="progress-text">
        {completed} of {total} completed
      </div>
    </div>
  );
};
```

### Smart Answer Suggestions
```typescript
const useAnswerSuggestions = (question: FollowupQuestion) => {
  const [suggestions, setSuggestions] = useState<string[]>([]);
  
  useEffect(() => {
    if (question) {
      generateSuggestions(question).then(setSuggestions);
    }
  }, [question]);
  
  const generateSuggestions = async (q: FollowupQuestion): Promise<string[]> => {
    // Generate contextual answer suggestions based on question type
    switch (q.category) {
      case 'temporal':
        return generateTemporalSuggestions(q);
      case 'character':
        return generateCharacterSuggestions(q);
      case 'setting':
        return generateSettingSuggestions(q);
      default:
        return [];
    }
  };
  
  return suggestions;
};
```

This chat interface design provides an intuitive, progressive way for users to answer follow-up questions while maintaining context and providing helpful guidance throughout the process.