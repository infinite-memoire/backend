"""
Tests for semantic chunker service
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from ai_implementation.semantic_chunker import SemanticChunker, SemanticChunk

@pytest.fixture
def semantic_chunker():
    """Create semantic chunker instance for testing"""
    return SemanticChunker()

@pytest.fixture
def sample_transcript():
    """Sample transcript for testing"""
    return """
    Hello, my name is John and I want to tell you about my childhood.
    I grew up in a small town called Springfield in the 1980s.
    My parents, Mary and Robert, owned a bakery on Main Street.
    Every morning, I would help them prepare fresh bread and pastries.
    
    One day, when I was ten years old, something amazing happened.
    A famous chef came to our bakery and tasted my mother's apple pie.
    He said it was the best pie he had ever eaten.
    This changed everything for our family business.
    
    Years later, I went to college to study business.
    I wanted to help expand our family bakery into a larger operation.
    During my studies, I met Sarah, who would later become my wife.
    We both shared a passion for cooking and entrepreneurship.
    """

@pytest.mark.asyncio
async def test_process_transcript_basic(semantic_chunker, sample_transcript):
    """Test basic transcript processing"""
    result = await semantic_chunker.process_transcript(sample_transcript)
    
    assert result is not None
    assert len(result.chunks) > 0
    assert result.total_words > 0
    assert result.creation_timestamp is not None
    assert isinstance(result.processing_metadata, dict)

@pytest.mark.asyncio
async def test_chunk_properties(semantic_chunker, sample_transcript):
    """Test that chunks have proper properties"""
    result = await semantic_chunker.process_transcript(sample_transcript)
    
    for chunk in result.chunks:
        assert isinstance(chunk, SemanticChunk)
        assert chunk.id is not None
        assert chunk.content is not None
        assert len(chunk.content.strip()) > 0
        assert chunk.embedding is not None
        assert isinstance(chunk.embedding, np.ndarray)
        assert chunk.word_count > 0
        assert 0 <= chunk.start_position <= 1
        assert 0 <= chunk.end_position <= 1
        assert chunk.start_position <= chunk.end_position

@pytest.mark.asyncio
async def test_entity_extraction(semantic_chunker, sample_transcript):
    """Test that entities are extracted from chunks"""
    result = await semantic_chunker.process_transcript(sample_transcript)
    
    # Should find person names
    all_entities = []
    for chunk in result.chunks:
        if chunk.entities:
            all_entities.extend(chunk.entities)
    
    person_entities = [ent for ent in all_entities if ent['label'] == 'PERSON']
    assert len(person_entities) > 0
    
    # Should find person names like "John", "Mary", "Robert", "Sarah"
    person_names = [ent['text'] for ent in person_entities]
    expected_names = ['John', 'Mary', 'Robert', 'Sarah']
    found_names = [name for name in expected_names if any(name in pname for pname in person_names)]
    assert len(found_names) > 0

@pytest.mark.asyncio
async def test_temporal_extraction(semantic_chunker):
    """Test temporal information extraction"""
    temporal_transcript = """
    In 1985, I was born in Chicago.
    Last summer, we visited my grandmother.
    Yesterday, I received an important phone call.
    Next week, I will start my new job.
    """
    
    result = await semantic_chunker.process_transcript(temporal_transcript)
    
    # Check for temporal information
    temporal_found = False
    for chunk in result.chunks:
        if chunk.temporal_info:
            if (chunk.temporal_info.get('explicit_dates') or 
                chunk.temporal_info.get('relative_expressions')):
                temporal_found = True
                break
    
    assert temporal_found

@pytest.mark.asyncio
async def test_chunk_size_configuration(semantic_chunker, sample_transcript):
    """Test that chunk size configuration works"""
    # Test with small chunks
    result_small = await semantic_chunker.process_transcript(sample_transcript, chunk_size=10)
    
    # Test with large chunks  
    result_large = await semantic_chunker.process_transcript(sample_transcript, chunk_size=30)
    
    # Small chunks should generally create more chunks
    assert len(result_small.chunks) >= len(result_large.chunks)
    
    # Check average chunk sizes
    avg_small = np.mean([chunk.word_count for chunk in result_small.chunks])
    avg_large = np.mean([chunk.word_count for chunk in result_large.chunks])
    
    # Large chunks should have higher average word count
    assert avg_large > avg_small

@pytest.mark.asyncio
async def test_merge_similar_chunks(semantic_chunker):
    """Test similar chunk merging functionality"""
    # Create transcript with repetitive content
    repetitive_transcript = """
    The weather is nice today. The weather is really nice.
    I love sunny days. I really love sunny days.
    The sun is shining brightly. The bright sun is shining.
    """
    
    # Test with merging enabled
    result_merged = await semantic_chunker.process_transcript(
        repetitive_transcript, merge_similar_chunks=True
    )
    
    # Test with merging disabled
    result_unmerged = await semantic_chunker.process_transcript(
        repetitive_transcript, merge_similar_chunks=False
    )
    
    # Merged should have fewer chunks
    assert len(result_merged.chunks) <= len(result_unmerged.chunks)

@pytest.mark.asyncio
async def test_speaker_boundary_preservation(semantic_chunker):
    """Test speaker boundary preservation"""
    speaker_transcript = """
    John: Hello everyone, my name is John.
    Mary: Nice to meet you John, I'm Mary.
    John: I've been working here for five years.
    Mary: That's wonderful, I just started last week.
    """
    
    result = await semantic_chunker.process_transcript(
        speaker_transcript, preserve_speaker_boundaries=True
    )
    
    # Check that speakers are detected
    speakers_found = set()
    for chunk in result.chunks:
        if chunk.speaker:
            speakers_found.add(chunk.speaker)
    
    # Should detect at least one speaker
    assert len(speakers_found) > 0

@pytest.mark.asyncio
async def test_confidence_scoring(semantic_chunker, sample_transcript):
    """Test that confidence scores are reasonable"""
    result = await semantic_chunker.process_transcript(sample_transcript)
    
    for chunk in result.chunks:
        # Confidence should be between 0 and 1
        assert 0 <= chunk.confidence <= 1
        
        # Well-formed chunks should have reasonable confidence
        if chunk.word_count >= 15 and chunk.entities:
            assert chunk.confidence > 0.3

@pytest.mark.asyncio
async def test_empty_transcript_handling(semantic_chunker):
    """Test handling of empty or minimal transcripts"""
    # Empty transcript
    with pytest.raises(Exception):
        await semantic_chunker.process_transcript("")
    
    # Very short transcript
    result = await semantic_chunker.process_transcript("Hello world.")
    assert len(result.chunks) >= 0  # Should handle gracefully

@pytest.mark.asyncio
async def test_health_check(semantic_chunker):
    """Test semantic chunker health check"""
    health = await semantic_chunker.health_check()
    
    assert health is not None
    assert "status" in health
    assert health["status"] in ["healthy", "unhealthy"]

def test_preprocessing_function(semantic_chunker):
    """Test transcript preprocessing"""
    messy_transcript = "  Hello   world!!  \n\n[INAUDIBLE] Nice day.   [MUSIC]  "
    
    cleaned = semantic_chunker._preprocess_transcript(messy_transcript)
    
    assert "[INAUDIBLE]" not in cleaned
    assert "[MUSIC]" not in cleaned
    assert "  " not in cleaned  # No double spaces
    assert cleaned.strip() == cleaned  # No leading/trailing whitespace

def test_cosine_similarity_calculation(semantic_chunker):
    """Test cosine similarity calculation"""
    # Create two identical embeddings
    embedding1 = np.array([1, 0, 0])
    embedding2 = np.array([1, 0, 0])
    
    similarity = semantic_chunker._calculate_cosine_similarity(embedding1, embedding2)
    assert abs(similarity - 1.0) < 0.001  # Should be 1.0
    
    # Create orthogonal embeddings
    embedding3 = np.array([0, 1, 0])
    similarity2 = semantic_chunker._calculate_cosine_similarity(embedding1, embedding3)
    assert abs(similarity2) < 0.001  # Should be 0.0

@pytest.mark.asyncio
async def test_processing_metadata(semantic_chunker, sample_transcript):
    """Test that processing metadata is complete"""
    result = await semantic_chunker.process_transcript(sample_transcript)
    
    metadata = result.processing_metadata
    
    required_fields = [
        "processing_time_seconds",
        "original_word_count", 
        "total_chunks_created",
        "average_chunk_size",
        "chunk_size_distribution"
    ]
    
    for field in required_fields:
        assert field in metadata
    
    assert metadata["processing_time_seconds"] > 0
    assert metadata["total_chunks_created"] == len(result.chunks)
    assert metadata["original_word_count"] > 0