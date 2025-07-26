# AI Processing System

A comprehensive multi-agent AI processing pipeline for converting audio transcripts into organized book chapters with storyline analysis and follow-up question generation.

## Overview

This system transforms raw audio transcripts into structured, coherent narrative content through:

1. **Semantic Chunking**: Intelligent text segmentation with entity and temporal extraction
2. **Graph Construction**: Storyline analysis using Neo4j graph database
3. **Multi-Agent Content Generation**: Coordinated AI agents for chapter writing and harmonization
4. **Quality Assurance**: Conflict detection and consistency checking
5. **Follow-up Questions**: Gap analysis and question generation for user interaction

## Architecture

### Core Components

- **SemanticChunker**: Processes transcripts into semantically coherent chunks with NLP analysis
- **GraphBuilder**: Constructs storyline graphs and integrates with Neo4j for persistence
- **Agent System**: Multi-agent framework with specialized agents for different tasks
- **Orchestrator**: Central coordinator managing the entire processing pipeline

### Agents

1. **ChapterWriterAgent**: Generates chapter content using LLM (Claude/template-based)
2. **ChapterHarmonizerAgent**: Ensures consistency across chapters and resolves conflicts
3. **FollowupQuestionsAgent**: Identifies information gaps and generates targeted questions

## Installation

### Prerequisites

- Python 3.8+
- Neo4j database (local or remote)
- spaCy English model
- Optional: Anthropic Claude API key for enhanced content generation

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

3. Configure Neo4j connection (update credentials in orchestrator initialization)

4. Optional: Set Anthropic API key for enhanced content generation

## Usage

### Basic Usage

```python
from ai_implementation import orchestrator

# Process a transcript
session_id = await orchestrator.process_transcript(
    transcript="Your audio transcript text here...",
    user_id="user123",
    user_preferences={
        "writing_style": "memoir",
        "chunk_size": 20,
        "preserve_speakers": True
    }
)

# Monitor progress
status = await orchestrator.get_session_status(session_id)
print(f"Progress: {status['progress_percentage']}%")

# Get results when complete
results = await orchestrator.get_session_results(session_id)
chapters = results['chapters']
questions = results['followup_questions']
```

### Progress Tracking

```python
async def progress_callback(update):
    print(f"Stage: {update.stage.value}")
    print(f"Progress: {update.progress_percentage}%")
    print(f"Task: {update.current_task}")

session_id = await orchestrator.process_transcript(
    transcript=transcript,
    user_id="user123",
    progress_callback=progress_callback
)
```

### Configuration Options

User preferences can customize processing:

```python
user_preferences = {
    "writing_style": "memoir",           # Writing style for chapters
    "tone": "reflective",                # Narrative tone
    "perspective": "first_person",       # Narrative perspective
    "chunk_size": 20,                    # Target words per chunk
    "preserve_speakers": True,           # Maintain speaker boundaries
    "merge_similar": True,               # Merge semantically similar chunks
    "target_word_count": 1200           # Target words per chapter
}
```

## Processing Pipeline

### Stage 1: Semantic Processing (5-25%)
- Text preprocessing and normalization
- Semantic chunking with configurable parameters
- Named entity recognition (people, places, dates)
- Temporal expression extraction
- Embedding generation with sentence transformers

### Stage 2: Graph Construction (25-50%)
- Node creation from semantic chunks
- Relationship edge calculation (semantic, temporal, spatial, causal)
- Community detection using Louvain algorithm
- Storyline identification and ranking
- Neo4j persistence with full graph structure

### Stage 3: Content Generation (50-80%)
- Chapter writing tasks distributed to agents
- Context preparation with relevant chunks and entities
- LLM-powered or template-based content generation
- Quality scoring and validation

### Stage 4: Quality Assurance (80-95%)
- Cross-chapter consistency analysis
- Conflict detection (character, timeline, factual, style)
- Automated harmonization where possible
- Follow-up question generation based on information gaps

### Stage 5: Finalization (95-100%)
- Final formatting and structure
- Comprehensive processing summary
- Result packaging and metadata generation

## Output Structure

### Chapters
Each generated chapter includes:
```python
{
    "storyline_id": "storyline_0",
    "title": "Chapter Title",
    "content": "Full chapter content in markdown...",
    "word_count": 1200,
    "quality_score": 0.85,
    "source_chunks": ["chunk_001", "chunk_002"],
    "participants": ["John", "Mary"],
    "themes": ["family", "childhood"],
    "confidence": 0.9,
    "generation_metadata": {...}
}
```

### Follow-up Questions
Generated questions for user interaction:
```python
{
    "id": "temporal_storyline_0",
    "category": "temporal",
    "question": "When did these events take place?",
    "context": "Storyline summary",
    "priority_score": 0.8,
    "reasoning": "Missing temporal information"
}
```

### Processing Summary
Comprehensive analysis results:
```python
{
    "session_id": "session_123",
    "processing_time_seconds": 45.2,
    "transcript_stats": {...},
    "graph_stats": {...},
    "content_stats": {...},
    "interaction_stats": {...}
}
```

## Testing

Run the test suite:
```bash
pytest ai_implementation/tests/ -v
```

Key test areas:
- Semantic chunking accuracy and configuration
- Graph construction and Neo4j integration
- Agent coordination and task processing
- End-to-end pipeline validation
- Error handling and edge cases

## Performance Considerations

### Optimization Tips

1. **Chunk Size**: Larger chunks reduce processing time but may decrease semantic granularity
2. **Graph Complexity**: Limit similarity threshold to reduce edge density for large transcripts
3. **Concurrent Processing**: Pipeline stages run sequentially but can be optimized for parallelization
4. **Memory Usage**: Large transcripts may require chunked processing and careful memory management

### Scalability

- **Session Management**: Built-in support for multiple concurrent processing sessions
- **Database Optimization**: Neo4j indexing and query optimization for large graphs
- **Agent Pool**: Agents can be scaled horizontally for increased throughput
- **Caching**: Embedding and NLP results can be cached for repeated processing

## Configuration

### Neo4j Setup
```python
# Custom Neo4j configuration
orchestrator = AgentOrchestrator(
    neo4j_uri="bolt://your-neo4j-host:7687",
    neo4j_user="your_username",
    neo4j_password="your_password"
)
```

### Model Configuration
```python
# Custom semantic chunker models
chunker = SemanticChunker(
    model_name='all-mpnet-base-v2',  # Larger, more accurate model
    spacy_model="en_core_web_lg"     # Larger spaCy model
)
```

## Error Handling

The system includes comprehensive error handling:

- **Graceful Degradation**: Falls back to template-based generation if LLM fails
- **Session Recovery**: Checkpoint system allows resuming from failures
- **Validation**: Input validation and sanity checks at each stage
- **Logging**: Structured logging for debugging and monitoring

## Monitoring

Health checks are available for all components:
```python
health = await orchestrator.health_check()
print(health['orchestrator'])  # Overall status
print(health['agents'])        # Individual agent status
print(health['services'])      # Service health (chunker, graph builder)
```

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new functionality
3. Update documentation for API changes
4. Use type hints and proper error handling
5. Follow logging conventions for debugging

## License

[License information here]