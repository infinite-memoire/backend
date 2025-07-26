# Firestore Data Models - Hierarchical Breakdown

Based on FastAPI project structure requirements for audio-to-book processing.

## 1. Audio Files Collection Structure
### 1.1 Primary Audio Document
- **Collection**: `audio_files`
- **Document ID**: Auto-generated UUID
- **Purpose**: Store audio file metadata and processing status

#### 1.1.1 Core Fields
```json
{
  "id": "auto_generated_uuid",
  "filename": "original_filename.mp3",
  "content_type": "audio/mpeg",
  "file_size_bytes": 15728640,
  "duration_seconds": 900.5,
  "upload_status": "completed",
  "processing_stage": "transcription_pending",
  "created_at": "2025-01-26T01:00:00Z",
  "updated_at": "2025-01-26T01:05:00Z"
}
```

#### 1.1.2 Processing Status Fields
```json
{
  "processing_stages": {
    "upload": {
      "status": "completed",
      "started_at": "2025-01-26T01:00:00Z",
      "completed_at": "2025-01-26T01:02:00Z"
    },
    "transcription": {
      "status": "in_progress",
      "started_at": "2025-01-26T01:03:00Z",
      "model_used": "whisper-large-v2"
    },
    "semantic_chunking": {
      "status": "pending"
    },
    "graph_generation": {
      "status": "pending"
    }
  }
}
```

#### 1.1.3 Audio Storage Reference
```json
{
  "storage": {
    "firestore_blob_ref": "audio_blobs/uuid_filename",
    "chunk_count": 1,
    "compression_applied": false,
    "encoding_format": "original"
  }
}
```

### 1.2 Audio Content Storage
- **Subcollection**: `audio_files/{audio_id}/chunks`
- **Purpose**: Store actual audio file content in manageable chunks

#### 1.2.1 Chunk Document Structure
```json
{
  "chunk_index": 0,
  "chunk_size_bytes": 1048576,
  "content_base64": "base64_encoded_audio_chunk",
  "checksum_md5": "hash_for_integrity_verification"
}
```

## 2. Transcripts Collection Structure
### 2.1 Primary Transcript Document
- **Collection**: `transcripts`
- **Document ID**: Auto-generated UUID
- **Purpose**: Store complete transcript with metadata

#### 2.1.1 Core Transcript Fields
```json
{
  "id": "auto_generated_uuid",
  "audio_file_id": "reference_to_audio_files_document",
  "raw_text": "Complete transcribed text content...",
  "language_detected": "en-US",
  "confidence_score": 0.92,
  "model_used": "whisper-large-v2",
  "processing_time_seconds": 45.2,
  "created_at": "2025-01-26T01:10:00Z"
}
```

#### 2.1.2 Word-Level Timestamps
```json
{
  "word_timestamps": [
    {
      "word": "Hello",
      "start_time": 0.5,
      "end_time": 0.8,
      "confidence": 0.98
    },
    {
      "word": "world",
      "start_time": 0.9,
      "end_time": 1.2,
      "confidence": 0.95
    }
  ]
}
```

#### 2.1.3 Semantic Chunks
```json
{
  "semantic_chunks": [
    {
      "chunk_id": "chunk_uuid_1",
      "start_word_index": 0,
      "end_word_index": 50,
      "text_content": "First semantic chunk text...",
      "temporal_marker": null,
      "processing_status": "pending_temporal_extraction"
    },
    {
      "chunk_id": "chunk_uuid_2",
      "start_word_index": 51,
      "end_word_index": 100,
      "text_content": "Second semantic chunk text...",
      "temporal_marker": "2020-05-15T00:00:00Z",
      "processing_status": "temporal_extracted"
    }
  ]
}
```

### 2.2 Transcript Processing Metadata
#### 2.2.1 Version Control
```json
{
  "version_info": {
    "version_number": 1,
    "previous_version_id": null,
    "changes_description": "Initial transcription",
    "modified_by": "system",
    "modification_type": "transcription"
  }
}
```

#### 2.2.2 Quality Metrics
```json
{
  "quality_metrics": {
    "overall_confidence": 0.92,
    "low_confidence_word_count": 15,
    "silence_duration_seconds": 12.5,
    "speech_rate_wpm": 145
  }
}
```

## 3. Text Chunks Collection Structure
### 3.1 Individual Chunk Documents
- **Collection**: `text_chunks`
- **Document ID**: Auto-generated UUID (referenced in transcripts)
- **Purpose**: Store individual semantic chunks for graph processing

#### 3.1.1 Chunk Content Fields
```json
{
  "id": "chunk_uuid",
  "transcript_id": "parent_transcript_uuid",
  "audio_file_id": "source_audio_file_uuid",
  "chunk_index": 0,
  "text_content": "This chunk contains a specific semantic unit...",
  "word_count": 45,
  "character_count": 289
}
```

#### 3.1.2 Temporal Information
```json
{
  "temporal_info": {
    "temporal_marker": "2020-05-15T00:00:00Z",
    "temporal_confidence": 0.85,
    "temporal_source": "explicit_mention",
    "temporal_extraction_method": "nlp_model",
    "requires_user_clarification": false
  }
}
```

#### 3.1.3 Graph Relationships
```json
{
  "graph_relationships": {
    "neo4j_node_id": "story_node_uuid",
    "assigned_chapters": ["chapter_1", "chapter_3"],
    "relationship_strength": 0.78,
    "processing_status": "graph_linked"
  }
}
```

### 3.2 Chunk Processing Status
#### 3.2.1 NLP Processing Results
```json
{
  "nlp_analysis": {
    "entities_extracted": [
      {"type": "PERSON", "text": "John", "confidence": 0.95},
      {"type": "LOCATION", "text": "Paris", "confidence": 0.88}
    ],
    "sentiment_score": 0.65,
    "topic_tags": ["family", "travel", "memories"],
    "semantic_similarity_vector": [0.1, 0.2, 0.3, "..."]
  }
}
```

## 4. User Data Collection Structure (Future)
### 4.1 User Documents
- **Collection**: `users`
- **Document ID**: User UUID
- **Purpose**: Store user-specific data and preferences

#### 4.1.1 Basic User Information
```json
{
  "id": "user_uuid",
  "created_at": "2025-01-26T01:00:00Z",
  "preferences": {
    "default_language": "en-US",
    "audio_quality_preference": "high",
    "processing_speed_preference": "accurate"
  }
}
```

### 4.2 User Audio Collections
- **Subcollection**: `users/{user_id}/audio_files`
- **Purpose**: User-specific audio file organization

## 5. Processing Jobs Collection Structure
### 5.1 Job Documents
- **Collection**: `processing_jobs`
- **Document ID**: Auto-generated UUID
- **Purpose**: Track long-running background processing tasks

#### 5.1.1 Job Status Fields
```json
{
  "id": "job_uuid",
  "job_type": "audio_transcription",
  "audio_file_id": "source_audio_uuid",
  "status": "in_progress",
  "progress_percentage": 65,
  "started_at": "2025-01-26T01:00:00Z",
  "estimated_completion": "2025-01-26T01:15:00Z",
  "error_message": null
}
```

#### 5.1.2 Job Configuration
```json
{
  "configuration": {
    "model_selection": "whisper-large-v2",
    "language_hint": "en-US",
    "quality_preference": "accurate",
    "background_task_id": "fastapi_task_uuid"
  }
}
```

## 6. Data Indexing Strategy
### 6.1 Composite Indexes
- **audio_files**: `(upload_status, created_at)`
- **transcripts**: `(audio_file_id, created_at)`
- **text_chunks**: `(transcript_id, chunk_index)`
- **processing_jobs**: `(status, started_at)`

### 6.2 Full-Text Search Indexes
- **transcripts.raw_text**: Full-text search capability
- **text_chunks.text_content**: Semantic search support

## 7. Data Relationships
### 7.1 Primary Relationships
```
audio_files (1) → transcripts (1)
transcripts (1) → text_chunks (many)
text_chunks (many) → neo4j_nodes (many-to-many via IDs)
audio_files (1) → processing_jobs (many)
```

### 7.2 Reference Integrity
- All foreign key references use UUID strings
- Orphaned document cleanup handled by background tasks
- No cascading deletes to preserve data integrity