# Phase 3: AI Processing System - Open Questions for Refinement

## Semantic Chunking & Text Processing

### Chunking Strategy
- **Rolling Window Implementation**:
  - What's the optimal chunk size for semantic coherence vs processing efficiency?
  >> 15-25 words - avg spoken sentence length
  - Overlap percentage between chunks to maintain context?
  >> 10 words?
  - How to handle sentence/paragraph boundaries in chunking?
  >> Is it necessary?
  - Should chunk size vary based on content type (narrative vs dialogue)?
  >> Longer chunks for narrative texts.

- **Semantic Boundaries**:
  - How to identify natural semantic breaks in transcripts?
  >> word embedding and checking for drastic changes in final vector.
  - Topic change detection algorithms and thresholds?
  >> assume topic changes when the avg vector changes after a certain number of windows.
  - Handling of speaker changes and dialogue segmentation?
  >> separate vectors by speakers.
  - Integration with transcript timestamps for temporal chunking?
  >> yes
### Text Preprocessing
- **Transcript Cleaning**:
  - Handling of STT errors and confidence scores in chunking?
  >> My tests last night using whisper showed really good STT results, maybe we just ignore it and check if there are inconsistencies later.
  - Punctuation and formatting normalization strategies?
  >> Ignore.
  - Handling of filler words, repetitions, and speech artifacts?
  >> ignore commonly used words in the semantic boundaries.
  - Name and location standardization/anonymization?
  >> user input?

## Storyline Graph Generation

### Graph Data Model
- **Node Structure**:
  - What properties should each node contain (summary, temporal info, participants)?
  >> summary, temp info, participants, average position on original transcript.
  - How to represent temporal information when dates are missing/ambiguous?
  >> Try to estimate it through the neighboring info on original transcript.
  - Node similarity scoring and clustering algorithms?
  >> avg vector similarity + temp info?
  - Handling of abstract concepts vs concrete events?
  >> Ignore the difference at first, see results.

- **Relationship Modeling**:
  - What types of relationships to capture (temporal, causal, thematic, spatial)?
  >> keep track of all four types, estimate primary type of relationship from # of hits on the original transcript.
  - Edge weight calculation based on semantic similarity?
  - Handling of conflicting or contradictory information?
  >> Separation of info through positional emb. Clustering of these info.
  - Bidirectional vs unidirectional relationships?


### Temporal Extraction & Annotation
- **Date/Time Identification**:
  - NLP models for temporal expression extraction (spaCy, NLTK, custom)?
  >> NLTK
  - Handling of relative temporal references ("last summer", "years ago")?
  >> NLTK should work for it too.
  - Confidence scoring for temporal assignments?
  
  - Default temporal assignment strategies for missing dates?
  >> Check neighboring sections' dates on original transcript and/or near nodes.

- **Timeline Construction**:
  - How to resolve temporal conflicts and ambiguities?
  >> 
  - Chronological ordering vs narrative ordering?
  >> Give the user the option? 
  - Handling of flashbacks and non-linear storytelling?
  >> Consideration of the vicinity in the original transcript.
  - User interaction for temporal disambiguation?
  >> Avoid if possible.
## Neo4j Graph Database Integration

### Database Schema Design
- **Graph Structure**:
  - Node labels and relationship types taxonomy?
  >> autogenerate labels.
  - Property schemas for different node and relationship types?
  >> positioning on original transcript, temp refs, semantic similarity.
  - Indexing strategy for fast querying and traversal?
  >> use native neo4j methods.
  - Graph versioning and historical changes tracking?
  >> save a version of the graph prior to whenever the user adds input.

- **Performance Optimization**:
  - Query optimization for large graphs (>10k nodes)?
  >> check neo4j doc for methods.
  - Memory management for graph operations?
  >> check neo4j doc for methods.
  - Caching strategies for frequently accessed subgraphs?
  >> check neo4j doc for methods.
  - Concurrent access patterns and locking strategies?
  >>check neo4j doc for methods.

### Data Synchronization
- **PostgreSQL Integration**:
  - How to maintain consistency between relational and graph data?
  - Change propagation strategies between databases?
  - Transaction coordination across dual database setup?
  - Backup and recovery procedures for both systems?

## Multi-Agent System Architecture

### Agent Coordination
- **Agent Communication**:
  - Message passing vs shared state for agent coordination?
  >> message passing.
  - Task distribution and load balancing between agents?
  
  - Agent failure handling and task reassignment?
  
  - Monitoring and debugging of agent interactions?

- **Workflow Orchestration**:
  - Sequential vs parallel agent execution?
  >> sequential tasks
  - Dependency management between agent tasks?
  >> wait for other agents messages to launch tasks.
  - Dynamic workflow adjustment based on content analysis?
  >> yes but not sure how.
  - Rollback and retry mechanisms for failed agent tasks?
  >> retry for writing and harmonization.

### Chapter Writer Agent
- **Content Generation**:
  - LLM choice for chapter writing (GPT-4, Claude, local models)?
  >> Claude
  - Prompt engineering for consistent style and quality?
  >> Use a pregenerated prompt where we only add the user input in the middle.
  - Context window management for large storylines?
  >> keep track of semantic closeness between topics and reorganize large storylines accordingly.
  - Quality assessment and iteration strategies?
  >> Pass the chapter through an NLP to check for internal inconsistencies or errors.

- **Tool Integration**:
  - Graph querying interface design for agent access?
  >> yes
  - Adjacent node loading strategies and caching?
  >> yes, try the most simple strategies
  - Web research integration scope and rate limiting?
  >> not at first but support functionality.
  - Content validation and fact-checking mechanisms?
  >> not a priority


### Chapter Harmonizer Agent
- **Content Consistency**:
  - Conflict detection algorithms between chapters?
  >> yes.
  - Style consistency analysis and harmonization?
  >> yes.
  - Character and timeline consistency checking?
  >> yes.
  - Cross-reference validation and link generation?
  >> not at first but support functionality.

- **Editing Capabilities**:
  - Automated editing vs suggestion-based editing?
  >> automated at first, add support for suggestion-based.
  - Version control for chapter iterations?
  >> keep the 3 last versions.
  - User approval workflows for major changes?
  >> ask for user approval.
  - Rollback capabilities for editing decisions?
  >> yes.


### Follow-up Questions Agent
- **Gap Identification**:
  - Algorithms for detecting missing connections in the graph?
  >> use the NLP.
  - Information completeness scoring for nodes?
  >> use the NLP.  
  - Question prioritization based on narrative importance?
  >> ask questions about more recurrent topics first.
  - Duplicate question detection and consolidation?
  >> reconsider question if user says it's a duplicate.

- **Question Generation**:
  - Question type taxonomy (factual, temporal, emotional, contextual)?
  >> all of them, prioritize according to # of questions per type.
  - Question formatting and user experience design?
  >> use the NLP to generate user-friendly questions.
  - Question storage format (structured vs natural language)?
  >> Structured + original transcription.
  - Question lifecycle management and tracking?
  >> Keep track of questions. Check for repeating questions.

## AI Model Management

### Model Selection & Deployment
- **LLM Integration**:
  - Local deployment vs API-based models (OpenAI, Anthropic)?
  >> Anthropic
  - Model switching based on task type and complexity?
  >> Not a priority
  - Cost optimization and usage tracking?
  >> Not a priority
  - Model performance monitoring and A/B testing?
  >> Not a priority


- **Specialized Models**:
  - Named Entity Recognition (NER) for person/place extraction?
  >> yes
  - Sentiment analysis for emotional tone detection?
  >> yes
  - Topic modeling for theme identification?
  >> yes
  - Summarization models for content condensation?
  >> yes

### Prompt Engineering & Fine-tuning
- **Prompt Templates**:
  - Standardized prompt formats for different agent tasks?
  >> yes.
  - Context injection strategies for maintaining coherence?
  >> yes.
  - Few-shot learning examples for consistent output?
  >> yes.
  - Prompt versioning and A/B testing framework?
  >> yes.

- **Model Customization**:
  - Fine-tuning requirements for domain-specific language?
  >> not a priority but add functionality.
  - Training data collection and annotation strategies?
  >> yes, but ask user.
  - Model evaluation metrics for chapter quality?
  >> yes
  - Continuous learning from user feedback?
  >> yes

## Processing Pipeline & Workflow

### Pipeline Architecture
- **Stage Dependencies**:
  - Sequential vs parallel processing stages?
  >> sequential
  - Checkpoint and resume capabilities for long-running processes?
  >> checkpoint every 5 min.
  - Error propagation and recovery strategies?
  >> yes
  - Progress tracking and user communication?
  >> yes.


- **Resource Management**:
  - Compute resource allocation for different processing stages?
  >> yes
  - Memory management for large graph operations?
  >> check available memory + adapt.
  - GPU utilization for AI model inference?
  >> if available.
  - Queue management for concurrent processing requests?
  >> yes.


### Quality Control & Validation
- **Output Quality**:
  - Automated quality metrics for generated chapters?
  >> yes
  - Coherence and consistency validation algorithms?
  >> yes
  - User feedback integration for quality improvement?
  >> yes
  - A/B testing framework for different approaches?
  >> yes

## User Interaction & Feedback

### Follow-up Question Interface
- **Question Presentation**:
  - UI/UX design for question display and answering?
  >> not a priority.
  - Question categorization and prioritization for users?
  >> yes.
  - Progress tracking for question completion?
  >> yes
  - Skip/defer options for questions?
  >> yes

- **Answer Processing**:
  - Natural language processing for user answers?
  >> yes.
  - Answer validation and consistency checking?
  >> yes.
  - Graph update strategies based on new information?
  >> yes
  - Reprocessing triggers for significant changes?
  >> yes.

### Human-in-the-Loop
- **User Control Points**:
  - Where should users have intervention capabilities?
  >> whenever a chapter version is finished.
  - Approval workflows for AI-generated content?
  >> yes
  - Manual override capabilities for AI decisions?
  >> avoid.
  - User preference learning and adaptation?
  >> yes but not a priority.

## Performance & Scalability

### Processing Optimization
- **Computational Efficiency**:
  - Parallel processing strategies for independent tasks?
  >> not a priority.
  - Caching strategies for repeated operations?
  >> no
  - Incremental processing for content updates?
  >> yes
  - Resource pooling and sharing across users?
  >> not a priority.


- **Scalability Considerations**:
  - Horizontal scaling strategies for AI processing?
  >> not a priority
  - Load balancing for concurrent user requests?
  >> yes
  - Database sharding strategies for large graphs?
  >> yes
  - Memory optimization for graph operations?
  >> yes
  
## Questions Requiring Immediate Decision
1. **Graph database schema design** - foundational for all AI processing
2. **LLM selection and deployment strategy** - impacts cost and performance
3. **Agent coordination mechanism** - affects system reliability and complexity
4. **Temporal extraction approach** - critical for storyline organization
5. **User interaction points** - defines the human-AI collaboration model
6. **Chunking strategy** - affects quality of downstream processing
7. **Quality control mechanisms** - impacts user satisfaction and content reliability