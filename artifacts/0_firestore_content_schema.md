# Firestore Content Schema for Output Management

## I. Content Hierarchy & Organization
### A. User-Centric Structure
   1. User Collections
      - User profiles and preferences
      - Book ownership and permissions
      - Processing session history
   2. User-Book Relationship Management
      - Book metadata and status tracking
      - Version control at book level
      - Publication status management

### B. Book-Centric Structure  
   1. Book Document Structure
      - Book metadata (title, description, creation date)
      - Processing configuration and preferences
      - Publication settings and status
   2. Book Version Management
      - Version directory structure in document
      - Chapter file organization within versions
      - Draft vs published version tracking
   3. Book Content Organization
      - Chapter sequence and ordering
      - Cross-chapter reference management
      - Table of contents generation

## II. Chapter Storage & Management
### A. Chapter Document Design
   1. Chapter Metadata
      - Chapter number and title
      - Word count and quality metrics
      - Source audio/transcript references
      - Generation timestamp and agent attribution
   2. Chapter Content Storage
      - Markdown content as document field
      - Source chunk references for traceability
      - Storyline node connections
      - Participant and theme tracking
   3. Chapter Version Control
      - Version number and change tracking
      - Processing status (draft, reviewed, published)
      - Quality assessment scores

### B. Content Relationships
   1. Chapter-to-Source Mapping
      - Link to original audio sessions
      - Reference to transcript chunks
      - Connection to storyline graph nodes
   2. Cross-Chapter Dependencies
      - Shared characters and timeline references
      - Theme continuation tracking
      - Narrative flow connections

## III. AI Processing Integration
### A. Processing Session Storage
   1. Session Metadata
      - Session ID and user association
      - Processing stage and progress tracking
      - Error state and recovery information
   2. Processing Artifacts
      - Semantic chunks with metadata
      - Storyline nodes and relationships
      - Agent generation logs and decisions
   3. Quality Metrics
      - Chapter quality scores
      - User satisfaction ratings
      - Processing time and efficiency metrics

### B. Agent Output Management
   1. Chapter Writer Agent Outputs
      - Generated chapter content
      - Source material citations
      - Quality self-assessment
   2. Harmonizer Agent Results
      - Inter-chapter consistency changes
      - Style and tone adjustments
      - Narrative flow improvements
   3. Follow-up Questions Agent
      - Generated question sets
      - Question categorization and priority
      - User response tracking

## IV. Follow-up Question System
### A. Question Storage Structure
   1. Question Documents
      - Question ID and category
      - Associated storyline context
      - Priority scoring and reasoning
   2. Question-Chapter Relationships
      - Source chapter references
      - Impact scope for answers
      - Reprocessing trigger conditions
   3. Question Lifecycle Management
      - Creation timestamp and agent source
      - User response status
      - Answer integration tracking

### B. Answer Integration System
   1. User Response Storage
      - Answer content and confidence level
      - Response timestamp and session
      - Answer quality validation
   2. Reprocessing Triggers
      - Content update scope determination
      - Chapter regeneration requirements
      - Graph update procedures

## V. Publishing & Export Management
### A. Publication Metadata
   1. Publication Status Tracking
      - Draft, review, published states
      - Publication timestamp and version
      - Format conversion status
   2. Export Configuration
      - HTML template preferences
      - Pandoc conversion settings
      - Style and formatting options
   3. Marketplace Integration
      - Publication visibility settings
      - Content description and keywords
      - User access permissions

### B. Content Validation & Quality
   1. Markdown Validation
      - Syntax checking and error reporting
      - Content structure validation
      - Cross-reference verification
   2. Publication Readiness
      - Completeness assessment
      - Quality threshold verification
      - User approval confirmation

## VI. Storage Optimization & Performance
### A. Document Structure Design
   1. Collection Organization
      - User → Books → Chapters hierarchy
      - Separate collections for questions and sessions
      - Index optimization for common queries
   2. Document Size Management
      - Chapter content chunking strategies
      - Large book handling approaches
      - Query performance optimization
   3. Metadata Tagging System
      - Flexible tagging for content discovery
      - Search index optimization
      - Category and theme organization

### B. Access Patterns & Caching
   1. Read Optimization
      - Frequently accessed content identification
      - Chapter loading strategies
      - Preview generation for UI
   2. Write Optimization
      - Batch operations for bulk updates
      - Atomic operations for consistency
      - Conflict resolution strategies