# API Endpoint Design - Hierarchical Breakdown

Based on Neo4j graph schema and Firestore data models for audio-to-book processing.

## 1. Audio Management Endpoints
### 1.1 Audio Upload Operations
#### 1.1.1 Chunked Upload Endpoint
```
POST /api/v1/audio/upload/initiate
```
**Purpose**: Initialize chunked audio file upload
**Request Body**:
```json
{
  "filename": "my_story.mp3",
  "file_size": 15728640,
  "content_type": "audio/mpeg",
  "chunk_size": 1048576
}
```
**Response**:
```json
{
  "upload_id": "upload_uuid",
  "audio_id": "audio_file_uuid",
  "chunk_count": 15,
  "upload_urls": [
    "/api/v1/audio/upload/chunk/upload_uuid/0",
    "/api/v1/audio/upload/chunk/upload_uuid/1"
  ]
}
```

#### 1.1.2 Chunk Upload Endpoint
```
PUT /api/v1/audio/upload/chunk/{upload_id}/{chunk_index}
```
**Purpose**: Upload individual file chunk
**Request**: Binary chunk data
**Response**:
```json
{
  "chunk_index": 0,
  "status": "uploaded",
  "checksum": "md5_hash",
  "bytes_uploaded": 1048576
}
```

#### 1.1.3 Complete Upload Endpoint
```
POST /api/v1/audio/upload/complete/{upload_id}
```
**Purpose**: Finalize upload and trigger processing
**Response**:
```json
{
  "audio_id": "audio_file_uuid",
  "status": "upload_completed",
  "processing_started": true,
  "estimated_processing_time_minutes": 15
}
```

### 1.2 Audio Status and Retrieval
#### 1.2.1 Audio Status Endpoint
```
GET /api/v1/audio/{audio_id}/status
```
**Purpose**: Get current processing status
**Response**:
```json
{
  "audio_id": "audio_file_uuid",
  "filename": "my_story.mp3",
  "upload_status": "completed",
  "processing_stages": {
    "transcription": {
      "status": "completed",
      "progress_percentage": 100,
      "started_at": "2025-01-26T01:00:00Z",
      "completed_at": "2025-01-26T01:10:00Z"
    },
    "semantic_chunking": {
      "status": "in_progress",
      "progress_percentage": 65,
      "started_at": "2025-01-26T01:10:00Z"
    },
    "graph_generation": {
      "status": "pending"
    }
  }
}
```

#### 1.2.2 Audio List Endpoint
```
GET /api/v1/audio
```
**Purpose**: List all audio files
**Query Parameters**:
- `status` (optional): Filter by processing status
- `limit` (optional): Number of results (default: 20)
- `offset` (optional): Pagination offset

**Response**:
```json
{
  "audio_files": [
    {
      "audio_id": "uuid",
      "filename": "story1.mp3",
      "duration_seconds": 900,
      "upload_status": "completed",
      "created_at": "2025-01-26T01:00:00Z"
    }
  ],
  "total_count": 5,
  "has_more": false
}
```

## 2. Transcript Management Endpoints
### 2.1 Transcript Retrieval
#### 2.1.1 Get Transcript Endpoint
```
GET /api/v1/transcripts/{transcript_id}
```
**Purpose**: Retrieve complete transcript with metadata
**Response**:
```json
{
  "transcript_id": "transcript_uuid",
  "audio_file_id": "audio_uuid",
  "raw_text": "Complete transcribed text...",
  "language_detected": "en-US",
  "confidence_score": 0.92,
  "word_count": 1250,
  "processing_time_seconds": 45.2,
  "semantic_chunks": [
    {
      "chunk_id": "chunk_uuid_1",
      "text_content": "First chunk text...",
      "temporal_marker": "2020-05-15T00:00:00Z",
      "word_count": 150
    }
  ],
  "created_at": "2025-01-26T01:10:00Z"
}
```

#### 2.1.2 Transcript by Audio ID
```
GET /api/v1/audio/{audio_id}/transcript
```
**Purpose**: Get transcript for specific audio file
**Response**: Same as transcript retrieval endpoint

### 2.2 Transcript Search and Export
#### 2.2.1 Text Search Endpoint
```
GET /api/v1/transcripts/search
```
**Purpose**: Full-text search across transcripts
**Query Parameters**:
- `q` (required): Search query
- `audio_id` (optional): Limit search to specific audio file
- `limit` (optional): Number of results

**Response**:
```json
{
  "results": [
    {
      "transcript_id": "uuid",
      "audio_id": "uuid",
      "matched_chunks": [
        {
          "chunk_id": "chunk_uuid",
          "text_snippet": "...highlighted search terms...",
          "temporal_marker": "2020-05-15T00:00:00Z"
        }
      ],
      "relevance_score": 0.85
    }
  ],
  "total_matches": 12
}
```

#### 2.2.2 Export Transcript Endpoint
```
GET /api/v1/transcripts/{transcript_id}/export
```
**Purpose**: Export transcript in various formats
**Query Parameters**:
- `format`: `json`, `txt`, `md`, `srt`
- `include_timestamps`: `true`, `false`

**Response**: Content in requested format

## 3. Semantic Analysis Endpoints
### 3.1 Text Chunk Management
#### 3.1.1 Get Text Chunks
```
GET /api/v1/transcripts/{transcript_id}/chunks
```
**Purpose**: Retrieve semantic chunks for transcript
**Response**:
```json
{
  "transcript_id": "transcript_uuid",
  "chunks": [
    {
      "chunk_id": "chunk_uuid",
      "text_content": "Semantic chunk text...",
      "chunk_index": 0,
      "word_count": 45,
      "temporal_info": {
        "temporal_marker": "2020-05-15T00:00:00Z",
        "temporal_confidence": 0.85,
        "requires_user_clarification": false
      },
      "graph_relationships": {
        "neo4j_node_id": "story_node_uuid",
        "assigned_chapters": ["chapter_1"]
      }
    }
  ]
}
```

#### 3.1.2 Update Chunk Temporal Information
```
PUT /api/v1/chunks/{chunk_id}/temporal
```
**Purpose**: Update temporal marker based on user input
**Request Body**:
```json
{
  "temporal_marker": "2020-05-15T00:00:00Z",
  "confidence": 0.95,
  "source": "user_input"
}
```

### 3.2 Graph Analysis Endpoints
#### 3.2.1 Story Graph Overview
```
GET /api/v1/graph/overview
```
**Purpose**: Get high-level graph statistics
**Response**:
```json
{
  "total_nodes": 156,
  "main_nodes": 12,
  "total_relationships": 324,
  "identified_chapters": 8,
  "temporal_coverage": {
    "earliest_date": "1995-06-15T00:00:00Z",
    "latest_date": "2023-12-01T00:00:00Z",
    "coverage_percentage": 78
  },
  "processing_status": "graph_analysis_complete"
}
```

#### 3.2.2 Chapter Structure
```
GET /api/v1/graph/chapters
```
**Purpose**: Get identified chapters and their organization
**Response**:
```json
{
  "chapters": [
    {
      "chapter_id": "chapter_uuid",
      "chapter_number": 1,
      "title": "Early Childhood",
      "summary": "Memories from early childhood years",
      "temporal_range": {
        "start": "1995-01-01T00:00:00Z",
        "end": "2000-12-31T23:59:59Z"
      },
      "main_nodes": [
        {
          "node_id": "story_node_uuid",
          "summary": "First day of school",
          "importance_score": 0.92
        }
      ],
      "total_story_nodes": 25,
      "estimated_word_count": 3500
    }
  ]
}
```

## 4. Follow-up Questions Endpoints
### 4.1 Question Management
#### 4.1.1 Get Pending Questions
```
GET /api/v1/questions/pending
```
**Purpose**: Retrieve questions requiring user input
**Response**:
```json
{
  "questions": [
    {
      "question_id": "question_uuid",
      "question_text": "When did you start working at the company mentioned in your story?",
      "question_type": "temporal_clarification",
      "related_chunks": [
        {
          "chunk_id": "chunk_uuid",
          "text_snippet": "I remember my first day at the company..."
        }
      ],
      "priority": "high",
      "created_at": "2025-01-26T01:20:00Z"
    }
  ],
  "total_pending": 5
}
```

#### 4.1.2 Submit Question Answer
```
POST /api/v1/questions/{question_id}/answer
```
**Purpose**: Submit user answer to follow-up question
**Request Body**:
```json
{
  "answer_text": "I started working there on September 15, 2010",
  "extracted_date": "2010-09-15T00:00:00Z",
  "confidence": "high"
}
```

**Response**:
```json
{
  "question_id": "question_uuid",
  "answer_processed": true,
  "affected_chunks": ["chunk_uuid_1", "chunk_uuid_2"],
  "graph_updates": {
    "nodes_updated": 3,
    "relationships_created": 2
  }
}
```

## 5. Processing Job Endpoints
### 5.1 Job Management
#### 5.1.1 Trigger Processing
```
POST /api/v1/processing/start
```
**Purpose**: Start AI processing pipeline for audio files
**Request Body**:
```json
{
  "audio_ids": ["audio_uuid_1", "audio_uuid_2"],
  "processing_stages": ["transcription", "semantic_analysis", "graph_generation"],
  "priority": "normal"
}
```

#### 5.1.2 Job Status
```
GET /api/v1/processing/jobs/{job_id}
```
**Purpose**: Get processing job status
**Response**:
```json
{
  "job_id": "job_uuid",
  "job_type": "full_pipeline",
  "status": "in_progress",
  "progress_percentage": 45,
  "current_stage": "semantic_analysis",
  "estimated_completion": "2025-01-26T02:00:00Z",
  "processed_items": 3,
  "total_items": 7,
  "error_count": 0
}
```

## 6. Health and System Endpoints
### 6.1 Health Checks
#### 6.1.1 System Health
```
GET /api/v1/health
```
**Purpose**: Basic health check
**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-26T01:30:00Z",
  "version": "1.0.0"
}
```

#### 6.1.2 Detailed Health
```
GET /api/v1/health/detailed
```
**Purpose**: Detailed system status
**Response**:
```json
{
  "status": "healthy",
  "services": {
    "firestore": {
      "status": "connected",
      "response_time_ms": 45
    },
    "neo4j": {
      "status": "connected",
      "response_time_ms": 23,
      "node_count": 156
    },
    "background_tasks": {
      "status": "running",
      "active_tasks": 2,
      "queued_tasks": 0
    }
  },
  "system_metrics": {
    "memory_usage_mb": 512,
    "cpu_usage_percent": 15,
    "disk_usage_percent": 45
  }
}
```

## 7. Error Handling and Response Patterns
### 7.1 Standard Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid file format. Only audio files are supported.",
    "details": {
      "field": "file",
      "provided_type": "text/plain",
      "expected_types": ["audio/*"]
    },
    "request_id": "req_uuid",
    "timestamp": "2025-01-26T01:30:00Z"
  }
}
```

### 7.2 HTTP Status Codes
- `200 OK`: Successful operation
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict (e.g., duplicate upload)
- `422 Unprocessable Entity`: Valid request but processing failed
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## 8. API Versioning and Documentation
### 8.1 Versioning Strategy
- URL path versioning: `/api/v1/`
- Backward compatibility maintained within major versions
- Deprecation notices in response headers

### 8.2 Documentation
- FastAPI automatic OpenAPI documentation at `/docs`
- Interactive API testing at `/redoc`
- Postman collection export available

## 9. Rate Limiting and Security
### 9.1 Rate Limiting (Future Implementation)
- Per-IP limits for upload endpoints
- Per-session limits for processing jobs
- Exponential backoff for failed requests

### 9.2 Security Headers
- CORS configured for mobile app domains
- Content-Type validation for all endpoints
- Request size limits for upload endpoints