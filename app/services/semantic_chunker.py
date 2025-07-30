"""
Semantic Chunking Service for AI Processing Pipeline
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from spacy.tokens import Doc
import re
from datetime import datetime
from app.utils.logging_utils import get_logger, log_performance

logger = get_logger("semantic_chunker")

@dataclass
class SemanticChunk:
    """Container for processed text segments with embeddings"""
    id: str
    content: str
    embedding: np.ndarray
    start_position: float
    end_position: float
    speaker: Optional[str] = None
    temporal_info: Optional[Dict] = None
    entities: List[Dict] = None
    confidence: float = 1.0
    word_count: int = 0
    original_transcript_position: Tuple[int, int] = None

@dataclass
class ProcessedTranscript:
    """Complete transcript with metadata and chunking"""
    original_text: str
    chunks: List[SemanticChunk]
    total_words: int
    processing_metadata: Dict[str, Any]
    creation_timestamp: datetime

class SemanticChunker:
    """
    Advanced semantic chunking service that creates coherent text segments
    optimized for downstream AI processing and storyline analysis.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 spacy_model: str = "en_core_web_sm"):
        """
        Initialize semantic chunker with configurable models
        
        Args:
            model_name: SentenceTransformer model for embeddings
            spacy_model: spaCy model for NLP processing
        """
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            self.nlp = spacy.load(spacy_model)
            logger.info("Semantic chunker initialized successfully", 
                       sentence_model=model_name, nlp_model=spacy_model)
        except Exception as e:
            logger.error("Failed to initialize semantic chunker", 
                        error=str(e), sentence_model=model_name, nlp_model=spacy_model)
            raise
            
        # Configuration parameters
        self.default_chunk_size = 20  # Target words per chunk
        self.overlap_size = 10  # Words overlap between chunks
        self.similarity_threshold = 0.75  # Threshold for merging similar chunks
        self.min_chunk_size = 8  # Minimum words per chunk
        self.max_chunk_size = 50  # Maximum words per chunk
        
    @log_performance(logger)
    async def process_transcript(self, 
                               transcript: str,
                               chunk_size: Optional[int] = None,
                               preserve_speaker_boundaries: bool = True,
                               merge_similar_chunks: bool = True) -> ProcessedTranscript:
        """
        Main entry point for processing transcripts into semantic chunks
        
        Args:
            transcript: Raw transcript text
            chunk_size: Target chunk size in words (defaults to self.default_chunk_size)
            preserve_speaker_boundaries: Whether to respect speaker changes
            merge_similar_chunks: Whether to merge semantically similar adjacent chunks
            
        Returns:
            ProcessedTranscript with chunks and metadata
        """
        start_time = datetime.now()
        chunk_size = chunk_size or self.default_chunk_size
        
        logger.info("Starting transcript processing",
                   transcript_length=len(transcript),
                   target_chunk_size=chunk_size,
                   preserve_speakers=preserve_speaker_boundaries)
        
        try:
            # Step 1: Preprocess and validate transcript
            cleaned_transcript = self._preprocess_transcript(transcript)
            
            # Step 2: Detect speaker boundaries if needed
            speaker_segments = []
            if preserve_speaker_boundaries:
                speaker_segments = self._detect_speaker_boundaries(cleaned_transcript)
            
            # Step 3: Create initial chunks with overlap
            initial_chunks = await self._create_initial_chunks(
                cleaned_transcript, chunk_size, speaker_segments
            )
            
            # Step 4: Generate embeddings for all chunks
            chunks_with_embeddings = await self._generate_embeddings(initial_chunks)
            
            # Step 5: Extract entities and temporal information
            enriched_chunks = await self._extract_entities_and_temporal(chunks_with_embeddings)
            
            # Step 6: Merge similar chunks if requested
            final_chunks = enriched_chunks
            if merge_similar_chunks:
                final_chunks = await self._merge_similar_chunks(enriched_chunks)
            
            # Step 7: Validate and clean up chunks
            validated_chunks = self._validate_chunks(final_chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create processing metadata
            metadata = {
                "processing_time_seconds": processing_time,
                "original_word_count": len(transcript.split()),
                "total_chunks_created": len(validated_chunks),
                "average_chunk_size": np.mean([chunk.word_count for chunk in validated_chunks]),
                "chunk_size_distribution": {
                    "min": min([chunk.word_count for chunk in validated_chunks]),
                    "max": max([chunk.word_count for chunk in validated_chunks]),
                    "std": np.std([chunk.word_count for chunk in validated_chunks])
                },
                "speaker_boundaries_detected": len(speaker_segments),
                "merge_similar_applied": merge_similar_chunks,
                "similarity_threshold": self.similarity_threshold
            }
            
            processed_transcript = ProcessedTranscript(
                original_text=transcript,
                chunks=validated_chunks,
                total_words=len(transcript.split()),
                processing_metadata=metadata,
                creation_timestamp=start_time
            )
            
            logger.info("Transcript processing completed",
                       total_chunks=len(validated_chunks),
                       processing_time_seconds=processing_time,
                       average_chunk_size=metadata["average_chunk_size"])
            
            return processed_transcript
            
        except Exception as e:
            logger.error("Transcript processing failed",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        transcript_length=len(transcript))
            raise
    
    def _preprocess_transcript(self, transcript: str) -> str:
        """Clean and normalize transcript text"""
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', transcript.strip())
        
        # Normalize punctuation
        cleaned = re.sub(r'\.{2,}', '...', cleaned)  # Multiple periods to ellipsis
        cleaned = re.sub(r'[?!]{2,}', '!', cleaned)  # Multiple exclamation/question marks
        
        # Remove STT artifacts
        cleaned = re.sub(r'\[INAUDIBLE\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[MUSIC\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[NOISE\]', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def _detect_speaker_boundaries(self, transcript: str) -> List[Dict[str, Any]]:
        """Detect speaker changes in transcript"""
        # Simple pattern matching for common speaker indicators
        speaker_patterns = [
            r'^([A-Z][a-z]+):\s',  # "Speaker: text"
            r'^\s*-\s*([A-Z][a-z]+):\s',  # "- Speaker: text"
            r'>>([A-Z][a-z]+):',  # ">>Speaker:"
        ]
        
        segments = []
        lines = transcript.split('\n')
        current_speaker = None
        current_start = 0
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check for speaker indicators
            speaker_found = None
            for pattern in speaker_patterns:
                match = re.match(pattern, line)
                if match:
                    speaker_found = match.group(1)
                    break
            
            if speaker_found and speaker_found != current_speaker:
                # New speaker detected
                if current_speaker is not None:
                    segments.append({
                        'speaker': current_speaker,
                        'start_line': current_start,
                        'end_line': line_idx - 1
                    })
                current_speaker = speaker_found
                current_start = line_idx
        
        # Add final segment
        if current_speaker:
            segments.append({
                'speaker': current_speaker,
                'start_line': current_start,
                'end_line': len(lines) - 1
            })
        
        return segments
    
    async def _create_initial_chunks(self, 
                                   transcript: str, 
                                   chunk_size: int,
                                   speaker_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Create initial chunks with sliding window approach"""
        words = transcript.split()
        chunks = []
        
        # If no speaker segments, process entire transcript
        if not speaker_segments:
            chunks.extend(self._chunk_text_segment(words, 0, len(words), chunk_size))
        else:
            # Process each speaker segment separately
            for segment in speaker_segments:
                # Convert line numbers to word positions (approximation)
                lines = transcript.split('\n')
                segment_text = '\n'.join(lines[segment['start_line']:segment['end_line'] + 1])
                segment_words = segment_text.split()
                
                if segment_words:
                    speaker_chunks = self._chunk_text_segment(
                        segment_words, 0, len(segment_words), chunk_size, segment['speaker']
                    )
                    chunks.extend(speaker_chunks)
        
        return chunks
    
    def _chunk_text_segment(self, 
                           words: List[str], 
                           start_idx: int, 
                           end_idx: int,
                           chunk_size: int,
                           speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Create chunks from a segment of words with overlap"""
        chunks = []
        i = start_idx
        chunk_id = 0
        
        while i < end_idx:
            # Determine chunk end position
            chunk_end = min(i + chunk_size, end_idx)
            
            # Extract chunk words
            chunk_words = words[i:chunk_end]
            chunk_text = ' '.join(chunk_words)
            
            # Skip empty or very small chunks
            if len(chunk_words) < self.min_chunk_size and i + chunk_size < end_idx:
                i += chunk_size - self.overlap_size
                continue
            
            chunk = {
                'id': f'chunk_{len(chunks)}_{chunk_id}',
                'content': chunk_text,
                'start_position': i / len(words) if words else 0,
                'end_position': chunk_end / len(words) if words else 1,
                'speaker': speaker,
                'word_count': len(chunk_words),
                'original_transcript_position': (i, chunk_end)
            }
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move to next chunk with overlap
            i += max(1, chunk_size - self.overlap_size)
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[SemanticChunk]:
        """Generate embeddings for all chunks"""
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings in batches for efficiency
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None, self.sentence_transformer.encode, texts
        )
        
        semantic_chunks = []
        for chunk_data, embedding in zip(chunks, embeddings):
            semantic_chunk = SemanticChunk(
                id=chunk_data['id'],
                content=chunk_data['content'],
                embedding=embedding,
                start_position=chunk_data['start_position'],
                end_position=chunk_data['end_position'],
                speaker=chunk_data.get('speaker'),
                word_count=chunk_data['word_count'],
                original_transcript_position=chunk_data['original_transcript_position']
            )
            semantic_chunks.append(semantic_chunk)
        
        return semantic_chunks
    
    async def _extract_entities_and_temporal(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Extract entities and temporal information from chunks"""
        enriched_chunks = []
        
        for chunk in chunks:
            # Process with spaCy
            doc = await asyncio.get_event_loop().run_in_executor(
                None, self.nlp, chunk.content
            )
            
            # Extract named entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence by default
                })
            
            # Extract temporal information
            temporal_info = self._extract_temporal_expressions(doc)
            
            # Update chunk with extracted information
            chunk.entities = entities
            chunk.temporal_info = temporal_info
            
            # Calculate confidence based on entity density and text quality
            chunk.confidence = self._calculate_chunk_confidence(chunk, doc)
            
            enriched_chunks.append(chunk)
        
        return enriched_chunks
    
    def _extract_temporal_expressions(self, doc: Doc) -> Dict[str, Any]:
        """Extract temporal expressions from spaCy doc"""
        temporal_info = {
            'explicit_dates': [],
            'relative_expressions': [],
            'temporal_entities': []
        }
        
        # Find DATE entities
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                temporal_info['explicit_dates'].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Find relative temporal expressions
        relative_patterns = [
            r'\b(yesterday|today|tomorrow)\b',
            r'\b(last|next)\s+(week|month|year|summer|winter|spring|fall)\b',
            r'\b(ago|later|before|after|during)\b',
            r'\b\d+\s+(years?|months?|weeks?|days?)\s+(ago|later)\b'
        ]
        
        text = doc.text.lower()
        for pattern in relative_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_info['relative_expressions'].append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'pattern': pattern
                })
        
        return temporal_info
    
    def _calculate_chunk_confidence(self, chunk: SemanticChunk, doc: Doc) -> float:
        """Calculate confidence score for chunk quality"""
        confidence_factors = []
        
        # Entity density (more entities = higher confidence)
        entity_density = len(chunk.entities) / chunk.word_count if chunk.word_count > 0 else 0
        confidence_factors.append(min(entity_density * 2, 1.0))  # Cap at 1.0
        
        # Sentence completeness
        sentences = list(doc.sents)
        complete_sentences = sum(1 for sent in sentences if sent.text.strip().endswith(('.', '!', '?')))
        sentence_completeness = complete_sentences / max(len(sentences), 1)
        confidence_factors.append(sentence_completeness)
        
        # Word count appropriateness
        word_count_score = 1.0
        if chunk.word_count < self.min_chunk_size:
            word_count_score = chunk.word_count / self.min_chunk_size
        elif chunk.word_count > self.max_chunk_size:
            word_count_score = self.max_chunk_size / chunk.word_count
        confidence_factors.append(word_count_score)
        
        # Average confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    async def _merge_similar_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Merge chunks with high semantic similarity"""
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Check similarity with next chunk
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                similarity = self._calculate_cosine_similarity(
                    current_chunk.embedding, next_chunk.embedding
                )
                
                # Merge if similarity exceeds threshold and chunks are from same speaker
                should_merge = (
                    similarity > self.similarity_threshold and
                    current_chunk.speaker == next_chunk.speaker and
                    (current_chunk.word_count + next_chunk.word_count) <= self.max_chunk_size
                )
                
                if should_merge:
                    merged_chunk = self._merge_two_chunks(current_chunk, next_chunk)
                    merged_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                else:
                    merged_chunks.append(current_chunk)
                    i += 1
            else:
                merged_chunks.append(current_chunk)
                i += 1
        
        logger.info("Chunk merging completed",
                   original_chunks=len(chunks),
                   merged_chunks=len(merged_chunks),
                   reduction_ratio=(len(chunks) - len(merged_chunks)) / len(chunks))
        
        return merged_chunks
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    def _merge_two_chunks(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> SemanticChunk:
        """Merge two similar chunks into one"""
        merged_content = f"{chunk1.content} {chunk2.content}"
        merged_embedding = (chunk1.embedding + chunk2.embedding) / 2
        merged_entities = (chunk1.entities or []) + (chunk2.entities or [])
        
        # Merge temporal information
        merged_temporal = {}
        if chunk1.temporal_info:
            merged_temporal.update(chunk1.temporal_info)
        if chunk2.temporal_info:
            for key, value in chunk2.temporal_info.items():
                if key in merged_temporal:
                    if isinstance(value, list):
                        merged_temporal[key].extend(value)
                else:
                    merged_temporal[key] = value
        
        return SemanticChunk(
            id=f"merged_{chunk1.id}_{chunk2.id}",
            content=merged_content,
            embedding=merged_embedding,
            start_position=chunk1.start_position,
            end_position=chunk2.end_position,
            speaker=chunk1.speaker,
            temporal_info=merged_temporal,
            entities=merged_entities,
            confidence=(chunk1.confidence + chunk2.confidence) / 2,
            word_count=chunk1.word_count + chunk2.word_count,
            original_transcript_position=(
                chunk1.original_transcript_position[0],
                chunk2.original_transcript_position[1]
            )
        )
    
    def _validate_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Validate and clean up final chunks"""
        valid_chunks = []
        
        for chunk in chunks:
            # Remove chunks that are too small or have very low confidence
            if chunk.word_count >= self.min_chunk_size and chunk.confidence >= 0.3:
                valid_chunks.append(chunk)
            else:
                logger.debug("Discarding low-quality chunk",
                           chunk_id=chunk.id,
                           word_count=chunk.word_count,
                           confidence=chunk.confidence)
        
        # Re-index chunks
        for i, chunk in enumerate(valid_chunks):
            chunk.id = f"chunk_{i:04d}"
        
        return valid_chunks
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for semantic chunker service"""
        try:
            # Test with sample text
            test_text = "This is a test sentence. It contains multiple words and should be processed correctly."
            test_result = await self.process_transcript(test_text, chunk_size=10, merge_similar_chunks=False)
            
            return {
                "status": "healthy",
                "test_result": {
                    "input_words": len(test_text.split()),
                    "chunks_created": len(test_result.chunks),
                    "processing_time": test_result.processing_metadata["processing_time_seconds"]
                },
                "model_info": {
                    "sentence_transformer_model": self.sentence_transformer._modules,
                    "spacy_model": self.nlp.meta['name'],
                    "chunk_size_config": {
                        "default": self.default_chunk_size,
                        "min": self.min_chunk_size,
                        "max": self.max_chunk_size,
                        "overlap": self.overlap_size
                    }
                }
            }
        except Exception as e:
            logger.error("Semantic chunker health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Global semantic chunker instance
semantic_chunker = SemanticChunker()