# Markdown Storage Design Implementation

## Firestore Collections Structure

### 1. Users Collection (`users`)
```
users/{user_id}
├── profile: {
│   name: string,
│   email: string,
│   created_at: timestamp,
│   subscription_tier: string,
│   preferences: {
│     writing_style: string,
│     default_tone: string,
│     target_word_count: number
│   }
│ }
├── books: array of book_ids
└── active_sessions: array of session_ids
```

### 2. Books Collection (`books`)
```
books/{book_id}
├── metadata: {
│   title: string,
│   description: string,
│   owner_user_id: string,
│   created_at: timestamp,
│   updated_at: timestamp,
│   status: "draft" | "in_progress" | "completed" | "published",
│   current_version: string,
│   publication_settings: {
│     is_public: boolean,
│     marketplace_visible: boolean,
│     html_template: string,
│     conversion_options: object
│   }
│ }
├── versions: {
│   "v1.0": {
│     created_at: timestamp,
│     chapter_count: number,
│     total_word_count: number,
│     status: string
│   }
│ }
└── processing_config: {
│   source_audio_sessions: array,
│   user_preferences: object,
│   ai_agent_settings: object
│ }
```

### 3. Chapters Collection (`chapters`)
```
chapters/{chapter_id}
├── metadata: {
│   book_id: string,
│   book_version: string,
│   chapter_number: number,
│   title: string,
│   word_count: number,
│   created_at: timestamp,
│   updated_at: timestamp,
│   status: "draft" | "generated" | "reviewed" | "published",
│   quality_score: number,
│   generation_agent: string
│ }
├── content: {
│   markdown_text: string,
│   table_of_contents: array,
│   tags: array,
│   themes: array,
│   participants: array
│ }
├── source_references: {
│   audio_session_ids: array,
│   transcript_chunk_ids: array,
│   storyline_node_ids: array,
│   source_confidence: number
│ }
└── generation_metadata: {
│   agent_type: string,
│   processing_session_id: string,
│   generation_timestamp: timestamp,
│   harmonization_applied: boolean,
│   changes_count: number,
│   quality_metrics: object
│ }
```

### 4. Processing Sessions Collection (`ai_sessions`)
```
ai_sessions/{session_id}
├── session_info: {
│   user_id: string,
│   book_id: string,
│   audio_session_id: string,
│   transcript_length: number,
│   created_at: timestamp,
│   updated_at: timestamp,
│   status: "processing" | "completed" | "failed",
│   current_stage: string,
│   progress_percentage: number
│ }
├── processing_results: {
│   semantic_chunks_count: number,
│   storyline_nodes_count: number,
│   chapters_generated: number,
│   followup_questions_count: number,
│   processing_time_seconds: number
│ }
└── error_log: array
```

### 5. Follow-up Questions Collection (`followup_questions`)
```
followup_questions/{question_id}
├── question_data: {
│   session_id: string,
│   book_id: string,
│   storyline_id: string,
│   category: string,
│   question_text: string,
│   context: string,
│   priority_score: number,
│   reasoning: string,
│   created_at: timestamp
│ }
├── user_response: {
│   answer_text: string,
│   confidence: number,
│   answered_at: timestamp,
│   processed: boolean
│ }
└── impact_scope: {
│   affected_chapters: array,
│   reprocessing_required: boolean,
│   update_priority: string
│ }
```

### 6. Question Answers Collection (`question_answers`)
```
question_answers/{answer_id}
├── answer_info: {
│   session_id: string,
│   question_id: string,
│   user_id: string,
│   answer_text: string,
│   confidence: number,
│   answered_at: timestamp,
│   processing_status: "pending" | "processed" | "integrated"
│ }
└── integration_results: {
│   chapters_updated: array,
│   graph_nodes_modified: array,
│   content_changes_applied: number
│ }
```

## Storage Implementation Patterns

### Chapter Content Storage Strategy
- **Markdown as Primary Format**: Store chapter content as markdown strings in Firestore
- **Size Management**: Chapters stored as single documents (MVP - assume chapters < 1MB)
- **Version Tracking**: Version number in metadata, separate documents for each version
- **Atomic Updates**: Use Firestore transactions for chapter updates with metadata

### Cross-Reference Management
```python
# Example chapter document structure
chapter_document = {
    "metadata": {
        "book_id": "book_123",
        "book_version": "v1.0",
        "chapter_number": 3,
        "title": "The Early Years",
        "word_count": 1247,
        "status": "generated",
        "quality_score": 0.87
    },
    "content": {
        "markdown_text": "# The Early Years\n\nIn the summer of 1965...",
        "themes": ["childhood", "family", "education"],
        "participants": ["John", "Mary", "Teacher Williams"]
    },
    "source_references": {
        "audio_session_ids": ["audio_001", "audio_003"],
        "transcript_chunk_ids": ["chunk_15", "chunk_16", "chunk_22"],
        "storyline_node_ids": ["node_5", "node_7"],
        "source_confidence": 0.92
    }
}
```

### Search and Query Optimization
```python
# Common query patterns for indexing
queries = [
    # Get all chapters for a book version
    "WHERE book_id == 'book_123' AND book_version == 'v1.0' ORDER BY chapter_number",
    
    # Get chapters by status
    "WHERE book_id == 'book_123' AND status == 'draft'",
    
    # Get questions for a session
    "WHERE session_id == 'session_456' ORDER BY priority_score DESC",
    
    # Get user's books
    "WHERE owner_user_id == 'user_789' ORDER BY updated_at DESC"
]
```

### Data Consistency Rules
1. **Book-Chapter Consistency**: Chapters must reference valid book_id and version
2. **Reference Integrity**: All source references must be validated
3. **Status Transitions**: Defined state machines for book and chapter status
4. **Version Immutability**: Published versions cannot be modified

### Performance Considerations
- **Batch Operations**: Use batch writes for multi-chapter updates
- **Caching Strategy**: Cache frequently accessed book metadata
- **Lazy Loading**: Load chapter content on-demand for large books
- **Index Optimization**: Create composite indexes for common query patterns

### Security Rules
```javascript
// Firestore security rules example
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /books/{bookId} {
      allow read, write: if request.auth != null && 
        resource.data.owner_user_id == request.auth.uid;
    }
    
    // Chapters inherit book permissions
    match /chapters/{chapterId} {
      allow read, write: if request.auth != null &&
        exists(/databases/$(database)/documents/books/$(resource.data.book_id)) &&
        get(/databases/$(database)/documents/books/$(resource.data.book_id)).data.owner_user_id == request.auth.uid;
    }
  }
}
```

## Implementation Guidelines

### Database Service Layer
```python
class ContentStorageService:
    def __init__(self, firestore_client):
        self.db = firestore_client
    
    async def create_book(self, user_id: str, book_data: dict) -> str:
        """Create new book with initial metadata"""
        
    async def store_chapter(self, book_id: str, chapter_data: dict) -> str:
        """Store chapter with source references"""
        
    async def get_book_chapters(self, book_id: str, version: str) -> List[dict]:
        """Retrieve all chapters for book version"""
        
    async def update_chapter_content(self, chapter_id: str, content: str) -> None:
        """Update chapter markdown content"""
        
    async def store_followup_questions(self, session_id: str, questions: List[dict]) -> None:
        """Store generated follow-up questions"""
```

### Validation Layer
```python
class ContentValidator:
    def validate_markdown_syntax(self, content: str) -> bool:
        """Validate markdown syntax and structure"""
        
    def validate_chapter_metadata(self, metadata: dict) -> bool:
        """Validate chapter metadata schema"""
        
    def validate_source_references(self, references: dict) -> bool:
        """Validate all source IDs exist"""
```

This design provides a scalable foundation for storing AI-generated content while maintaining data integrity and supporting the publishing workflow.