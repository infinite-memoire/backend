# Phase 1: Backend Foundation - Open Questions for Refinement

## Technology Stack & Framework Selection

### Python Web Framework
- **FastAPI vs Django vs Flask**: Which framework best suits our needs for:
  - High-performance audio file handling?
  - WebSocket support for real-time progress updates?
  - Built-in async support for long-running AI processing?
  - API documentation and testing capabilities?
  >>> use fastapi

### Database Architecture
- **PostgreSQL configuration**: 
  - What specific schemas do we need for users, audio metadata, processing jobs?
  - Should we use separate databases for operational vs analytical data?
  - What indexing strategy for audio file lookups and user queries?
  >>> we will use firestore nosql instead of postgresql
    - the firestore will hold: 
      - the audio files and metadata
      - the transcripts with id and temporal marker (optional, isoformat, null by default)
      - the user data
      - other metadata

- **Neo4j integration**:
  - How do we handle the dual database setup (firestore + Neo4j)?
  >>> neo4j will hold the graph database
  >>> each node will link to a list of transcript chunk ids relating to chunks in the firestore
  >>> each node will have a list of chapters
  >>> each main node (=node with high number of edges) will be identified as a chapter 

  - What's the data synchronization strategy between relational and graph data?
    >>> no sync, the dbs handle different data & data points are linked via ids
  - Should user authentication data live in PostgreSQL only?
    >>> ignore user auth. This is a n MVP
  
### Authentication & Security
- **Firebase Integration**: 
  - How do we validate Firebase JWT tokens on the backend?
    >>> ignore user auth. This is a n MVP
  - Should we replicate user data in PostgreSQL or query Firebase on each request?
    >>> ignore user auth. This is a n MVP
  - What's the strategy for handling Firebase service account credentials?
    >>> ignore user auth. This is a n MVP

- **API Security**:
  - Rate limiting requirements for audio upload endpoints?
    >>> ignore this, its an MVP
  - File upload size limits and validation rules?
    >>> ignore this, its an MVP
  - CORS configuration for mobile app integration?
    >>> permissive, its an MVP

## Infrastructure & Deployment

### Environment Management
- **Development vs Production**:
  - How many environments do we need (dev/staging/prod)?
    >>> only dev
  - Should we use Docker containers for local development?
    >>> no,  but write Dockerfiles for cloud deployment
  - What's the database seeding strategy for different environments?
    >>> no seeding, its an MVP

### File Storage Strategy
- **Audio File Storage**:
  - Local filesystem vs cloud storage (S3, GCS) for audio files?
    >>> use firestore
  - Should we integrate with Firebase Storage or use separate storage?
    >>> use firestore
  - What's the backup and disaster recovery strategy?
    >>> ignore this, its an MVP
  - File retention policies and cleanup strategies?
    >>> ignore this, its an MVP

### Performance & Scalability
- **Async Processing**:
  - Celery vs FastAPI BackgroundTasks vs custom job queue?
    >>> use fastapi background tasks
  - Redis vs RabbitMQ for job queuing?
    >>> use async for event queuing, that shoul be enough
  - How do we handle job persistence and failure recovery?
    >>> ignore this, its an MVP

## API Design & Integration

### Mobile App Integration
- **API Endpoints Structure**:
  - RESTful vs GraphQL for complex data relationships?
    >>> restful
  - WebSocket endpoints for real-time progress updates?
    >>> no
  - Pagination strategy for large datasets (stories, recordings)?
    >>> no

### Audio Processing Pipeline
- **File Upload Handling**:
  - Streaming upload for large files vs chunked upload?
    >>> chunked upload
  - Progress tracking and resume capability for failed uploads?
    >>> no
  - Audio format validation and conversion pipeline?
    >>> keep it simple, its an MVP

## Development Workflow

### Testing Strategy
- **Test Coverage Requirements**:
  - Unit test coverage targets for each component?
    >>> ignore this, its an MVP
  - Integration test strategy for database operations?
    >>> ignore this, its an MVP
  - End-to-end test strategy with mobile app integration?
    >>> ignore this, its an MVP

### Code Organization
- **Project Structure**:
  - Monolithic vs microservices architecture?
    >>> monolithic
  - How to organize models, services, and API routes?
    >>> keep it robust and simple without over engineering
  - Dependency injection pattern for services?
    >>> ignore this, its an MVP

## Configuration & Secrets Management
- **Environment Variables**:
  - Which configuration values should be environment-specific?
    >>> all non-secret values are to be placed in a config file
  - How to handle sensitive data (API keys, database passwords)?
    >>> all secret values are to be placed in a config file
  - Local development setup for team members?
    >>> ignore this, its an MVP

## Monitoring & Logging
- **Observability Requirements**:
  - What metrics should we track (API response times, upload speeds)?
    >>> ignore this, its an MVP
  - Logging strategy for debugging and monitoring?
    >>> use python logging extensively for quick and easy debugging
  - Error tracking and alerting requirements?
    >>> ignore this, its an MVP

## Migration & Data Management
- **Database Migrations**:
  - Alembic for PostgreSQL schema migrations?
    >>> no postgresql
  - How to handle Neo4j schema changes and data migrations?
    >>> ignore this, the schema is fixed
  - Data backup and restore procedures?
    >>> ignore this, its an MVP

## Questions Requiring Immediate Decision
1. **Primary web framework choice** - impacts all subsequent architecture decisions
2. **Database connection strategy** - affects performance and complexity
3. **File storage location** - impacts deployment and scaling strategy
4. **Authentication flow** - must align with mobile app implementation
5. **Development environment setup** - needed for team productivity