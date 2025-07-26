# Graph-Based Semantic Processing Design

## Overview

This design implements a hybrid approach combining the strengths of vector embeddings, entity extraction, and LLM-guided analysis to create a robust semantic processing pipeline for converting audio transcripts into organized storylines and chapters.

## Core Architecture

### 1. Semantic Chunking Pipeline

**Chunking Strategy:**
- Initial chunking based on 15-25 words (average spoken sentence length)
- 10-word overlap between chunks to maintain context
- Post-processing to align chunks with sentence boundaries
- Separate processing tracks for different speakers

**Boundary Detection:**
```python
class SemanticChunker:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.75
        
    def create_semantic_chunks(self, transcript: str) -> List[SemanticChunk]:
        # 1. Initial windowed chunking
        # 2. Generate embeddings for each chunk
        # 3. Calculate similarity between adjacent chunks
        # 4. Merge chunks with high similarity (>0.75)
        # 5. Split chunks with low similarity and significant vector changes
```

### 2. Graph Data Model

**Node Structure:**
```python
@dataclass
class StorylineNode:
    id: str
    summary: str                    # AI-generated summary of the content
    temporal_info: TemporalInfo     # Extracted dates/times with confidence
    participants: List[str]         # People mentioned in this node
    semantic_embedding: np.ndarray  # 384-dim vector for similarity
    original_position: float        # Position in original transcript (0.0-1.0)
    confidence_score: float         # Overall confidence in node accuracy
    chunk_ids: List[str]           # Associated text chunks
```

**Relationship Types:**
```python
class RelationshipType(Enum):
    TEMPORAL = "follows_in_time"      # Chronological sequence
    CAUSAL = "causes_or_leads_to"     # Cause-effect relationships
    THEMATIC = "shares_theme_with"    # Similar topics/themes
    SPATIAL = "co_located_with"       # Same location references
    PARTICIPANT = "involves_same_people" # Shared participants
```

### 3. Multi-Stage Processing Pipeline

**Stage 1: Text Processing & Entity Extraction**
```python
class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.temporal_extractor = TemporalExtractor()
        
    def process_chunks(self, chunks: List[str]) -> List[ProcessedChunk]:
        # 1. Named Entity Recognition (PERSON, ORG, GPE, DATE, TIME)
        # 2. Temporal expression extraction with NLTK
        # 3. Dependency parsing for relationship hints
        # 4. Sentiment analysis for emotional tone
```

**Stage 2: Graph Construction**
```python
class GraphBuilder:
    def build_storyline_graph(self, processed_chunks: List[ProcessedChunk]) -> nx.Graph:
        # 1. Create nodes from semantically similar chunks
        # 2. Calculate edge weights using multiple factors:
        #    - Semantic similarity (cosine distance of embeddings)
        #    - Temporal proximity (closer dates = stronger edges)
        #    - Entity overlap (shared people/places)
        #    - Positional proximity in original transcript
        # 3. Apply community detection (Louvain algorithm)
        # 4. Identify main storylines as high-centrality node clusters
```

**Stage 3: LLM Validation & Enhancement**
```python
class LLMEnhancer:
    def validate_and_enhance_graph(self, graph: nx.Graph) -> nx.Graph:
        # 1. Use Claude to validate detected relationships
        # 2. Generate improved summaries for nodes
        # 3. Resolve temporal conflicts through LLM reasoning
        # 4. Identify missing connections between storylines
```

## Temporal Processing System

### Temporal Information Extraction
```python
class TemporalExtractor:
    def extract_temporal_info(self, text: str) -> TemporalInfo:
        # 1. Use NLTK's temporal expression parser
        # 2. Apply regex patterns for common date formats
        # 3. Resolve relative references ("last summer" → actual date range)
        # 4. Assign confidence scores based on extraction method
        
    def resolve_temporal_conflicts(self, nodes: List[StorylineNode]) -> None:
        # 1. Build temporal constraint graph
        # 2. Identify conflicts (A before B, B before C, C before A)
        # 3. Use positional information in transcript to resolve
        # 4. Flag unresolvable conflicts for user review
```

### Timeline Construction
- **Primary Timeline**: Chronological ordering based on extracted dates
- **Narrative Timeline**: Order based on appearance in transcript
- **User Toggle**: Allow switching between chronological and narrative views

## Neo4j Database Schema

### Node Labels and Properties
```cypher
// Main storyline nodes
CREATE (:Storyline {
    id: String,
    summary: String,
    temporal_start: DateTime,
    temporal_end: DateTime,
    confidence: Float,
    participants: [String],
    themes: [String],
    position_in_transcript: Float
})

// Text chunk nodes
CREATE (:TextChunk {
    id: String,
    content: String,
    start_time: Float,
    end_time: Float,
    speaker: String,
    embedding: [Float]
})

// Entity nodes
CREATE (:Person {name: String, mentions: Integer})
CREATE (:Place {name: String, mentions: Integer})
CREATE (:Event {name: String, date: DateTime})
```

### Relationship Schema
```cypher
// Storyline relationships
(:Storyline)-[:TEMPORAL_SEQUENCE {confidence: Float}]->(:Storyline)
(:Storyline)-[:CAUSAL_LINK {strength: Float}]->(:Storyline)
(:Storyline)-[:THEMATIC_SIMILARITY {score: Float}]->(:Storyline)

// Chunk associations
(:Storyline)-[:CONTAINS {relevance: Float}]->(:TextChunk)
(:TextChunk)-[:MENTIONS]->(:Person|Place|Event)
```

## Chapter Organization Algorithm

### Main Storyline Identification
```python
def identify_main_storylines(graph: nx.Graph) -> List[StorylineNode]:
    # 1. Calculate centrality measures (degree, betweenness, eigenvector)
    # 2. Apply community detection (Louvain algorithm)
    # 3. Rank communities by:
    #    - Total content volume (number of chunks)
    #    - Temporal span (how much time they cover)
    #    - Entity richness (number of unique people/places)
    # 4. Select top N communities as main storylines
```

### Chapter Assignment Strategy
```python
def assign_chunks_to_chapters(storylines: List[StorylineNode], 
                             chunks: List[TextChunk]) -> Dict[int, List[TextChunk]]:
    # 1. Each main storyline becomes a chapter
    # 2. Secondary nodes assigned to nearest main node (shortest path)
    # 3. Nodes can belong to multiple chapters if distance is similar
    # 4. Temporal ordering within chapters
    # 5. Cross-references generated for multi-chapter nodes
```

## Quality Control & Validation

### Automated Quality Metrics
- **Graph Coherence**: Measure clustering coefficient and modularity
- **Temporal Consistency**: Validate chronological constraints
- **Entity Consistency**: Check for name variations and conflicts
- **Coverage**: Ensure all chunks are assigned to storylines

### User Interaction Points
1. **Temporal Disambiguation**: Present conflicts for user resolution
2. **Chapter Review**: Allow manual storyline merging/splitting
3. **Entity Validation**: Confirm person/place identifications
4. **Theme Labeling**: User-provided chapter titles and themes

## Implementation Priority

1. **Phase 3.1**: Basic semantic chunking and embedding generation
2. **Phase 3.2**: Graph construction with simple similarity edges
3. **Phase 3.3**: Entity extraction and temporal processing
4. **Phase 3.4**: Neo4j integration and storage
5. **Phase 3.5**: Chapter organization and multi-agent system
6. **Phase 3.6**: LLM enhancement and quality validation

This design provides a solid foundation for converting raw transcripts into meaningful, organized content while maintaining flexibility for future enhancements and user customization.