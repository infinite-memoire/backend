# Output Management & Publishing System

This module implements the complete output management and publishing pipeline for the Memoire AI Book Generator, including content storage, version control, HTML conversion, and marketplace publishing.

## Architecture Overview

The system consists of several interconnected services:

1. **Content Storage Service** - Manages book and chapter content in Firestore
2. **Version Tracking Service** - Handles database-based version control
3. **Sequential Chapter Workflow** - Coordinates AI chapter generation
4. **Markdown Validation Service** - Validates content quality and structure
5. **Chat Interface Service** - Manages follow-up question interactions
6. **HTML Conversion Pipeline** - Converts markdown to formatted HTML using Pandoc
7. **Publishing Workflow Service** - Handles manual publishing to marketplace
8. **Marketplace Service** - Manages book listings and discovery

## Features

### Content Management
- Firestore-based content storage with metadata
- Database-driven version control (no Git dependency)
- Sequential chapter generation with AI agents
- Comprehensive markdown validation
- Cross-reference and consistency tracking

### Publishing Pipeline
- Pandoc-based HTML conversion with templates
- Manual publishing workflow with validation
- Preview generation and quality checks
- Marketplace integration with metadata
- Multi-format export capabilities

### User Interaction
- Chat interface for follow-up questions
- Progressive question presentation
- Answer integration and content updates
- Real-time status tracking

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Pandoc (required for HTML conversion)
# Ubuntu/Debian:
sudo apt-get install pandoc

# macOS:
brew install pandoc

# Windows:
# Download from https://pandoc.org/installing.html
```

## Configuration

Copy the environment template and configure:

```bash
cp .env.template .env
# Edit .env with your settings
```

Required environment variables:
- `FIRESTORE_PROJECT_ID` - Google Cloud Firestore project
- `ANTHROPIC_API_KEY` - For AI agent interactions
- `NEO4J_URI` - Neo4j database connection
- `NEO4J_USER` - Neo4j username  
- `NEO4J_PASSWORD` - Neo4j password

## Usage

### Starting the Service

```bash
# Development mode
uvicorn main:app --reload --port 8000

# Production mode
gunicorn main:app -w 4 -k uvicorn.workers.UnicornWorker
```

### API Endpoints

#### Health & System
- `GET /health` - Basic health check and system status

#### Audio Upload & Processing
- `POST /upload/initiate` - Initiate chunked audio upload session
- `PUT /upload/chunk/{upload_id}/{chunk_index}` - Upload audio chunk
- `POST /upload/complete/{upload_id}` - Complete upload and start processing
- `GET /upload/status/{upload_id}` - Get upload session status

#### AI Processing & Transcript Analysis
- `POST /ai/process-transcript` - Start AI-powered transcript processing
- `GET /ai/status/{session_id}` - Check processing status and progress
- `GET /ai/results/{session_id}` - Get final processing results and generated chapters
- `POST /ai/answer-question/{session_id}` - Answer follow-up questions for content refinement
- `POST /ai/cancel/{session_id}` - Cancel active processing session
- `GET /ai/health` - AI service health check
- `GET /ai/sessions/{user_id}` - Get user's processing sessions
- `GET /ai/metrics` - Get AI processing performance metrics

#### Publishing Workflow (Phase 4)
- `POST /publishing/start/{book_id}` - Start manual publishing workflow
- `PUT /publishing/{workflow_id}/metadata` - Update publication metadata (title, description, author, etc.)
- `PUT /publishing/{workflow_id}/settings` - Update publication settings (visibility, pricing, licensing)
- `POST /publishing/{workflow_id}/validate` - Validate publication readiness and quality
- `POST /publishing/{workflow_id}/preview` - Generate HTML preview of published book
- `POST /publishing/{workflow_id}/publish` - Submit book for publication to marketplace
- `GET /publishing/{workflow_id}/status` - Get current publishing workflow status
- `DELETE /publishing/{workflow_id}` - Cancel publishing workflow
- `GET /publishing/workflows` - List all user publishing workflows

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test modules
pytest tests/test_publishing.py
pytest tests/test_conversion.py
```

## API Endpoint Details

### Audio Upload Flow
1. **Initiate Upload** - Creates chunked upload session for large audio files
2. **Upload Chunks** - Streams audio data in manageable chunks with progress tracking
3. **Complete Upload** - Finalizes upload and triggers automatic transcript processing
4. **Monitor Status** - Track upload progress and processing state

### AI Processing Pipeline
1. **Process Transcript** - Analyzes raw transcript text using multiple AI agents:
   - Semantic chunking and theme extraction
   - Storyline construction and narrative flow analysis
   - Chapter generation with quality scoring
   - Neo4j graph database population for relationships
2. **Status Monitoring** - Real-time progress tracking with stage-by-stage updates
3. **Results Retrieval** - Access generated chapters, themes, and storylines
4. **Interactive Refinement** - Answer follow-up questions to improve content quality
5. **Session Management** - Cancel, restart, or monitor multiple processing sessions

### Publishing Workflow (Phase 4)
The manual publishing system provides complete control over book publication:

1. **Workflow Initiation** - Start publishing process with automatic metadata generation
2. **Metadata Management** - Edit title, description, author bio, categories, tags, and content warnings
3. **Settings Configuration** - Control visibility (private/public/featured), licensing, and marketplace options
4. **Quality Validation** - Comprehensive checks for content completeness, metadata quality, and publication readiness
5. **Preview Generation** - Create HTML preview showing final book appearance
6. **Publication** - Submit to marketplace with automatic approval (MVP) or manual review
7. **Status Tracking** - Monitor workflow progress and handle errors
8. **Workflow Management** - Cancel, restart, or manage multiple publishing workflows

### Project Structure
```
backend/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── health.py - System health endpoints
│   │       ├── upload.py - Chunked audio upload
│   │       ├── ai_processing.py - AI transcript processing
│   │       └── publishing.py - Manual publishing workflow
│   ├── models/
│   │   ├── content.py - Book and chapter data models
│   │   ├── publishing.py - Publishing workflow models
│   │   ├── ai_processing.py - AI processing models
│   │   └── upload_session.py - Upload session models
│   ├── services/
│   │   ├── content_storage.py - Firestore content management
│   │   ├── publishing.py - Publishing workflow service
│   │   ├── marketplace.py - Marketplace management
│   │   ├── html_conversion.py - HTML generation service
│   │   ├── orchestrator.py - AI agent orchestration
│   │   ├── upload_service.py - Upload session management
│   │   └── firestore.py - Firestore database service
│   ├── utils/
│   │   ├── auth.py - JWT authentication
│   │   ├── exceptions.py - Custom exception classes
│   │   ├── logging.py - Structured logging
│   │   └── upload_validation.py - File validation
│   ├── dependencies.py - FastAPI dependency injection
│   └── main.py - Application entry point
├── templates/ - HTML templates for book generation
├── tests/ - Test suites
├── requirements.txt - Python dependencies
└── Dockerfile - Container configuration
```

## Development

### Adding New Features

1. Define data models in `app/models/`
2. Implement business logic in `app/services/`
3. Create API endpoints in `app/api/routes/`
4. Add comprehensive tests in `tests/`
5. Update documentation

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all function signatures
- Maintain test coverage above 80%
- Document all public APIs with docstrings

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t memoire-output-management .

# Run container
docker run -p 8000:8000 \
  -e FIRESTORE_PROJECT_ID=your-project \
  -e ANTHROPIC_API_KEY=your-key \
  memoire-output-management
```

### Production Considerations

- Use environment-specific configuration
- Set up proper logging and monitoring
- Configure rate limiting for API endpoints
- Use Redis for caching frequently accessed content
- Set up CDN for static assets and generated HTML
- Monitor storage usage and implement cleanup policies

## Monitoring

The system includes built-in monitoring endpoints:

- `/health` - Basic health check
- `/metrics` - Prometheus metrics
- `/api/v1/system/status` - Detailed system status

## Troubleshooting

### Common Issues

1. **Pandoc conversion fails**
   - Ensure Pandoc is installed and in PATH
   - Check template file permissions
   - Verify markdown syntax validity

2. **Firestore connection errors**
   - Check project ID and credentials
   - Verify network connectivity
   - Ensure proper IAM permissions

3. **AI processing timeouts**
   - Check Anthropic API key validity
   - Monitor rate limits
   - Verify Neo4j connectivity

### Logs

Application logs are written to:
- Development: Console output
- Production: `/var/log/memoire/output-management.log`

Log levels can be configured via `LOG_LEVEL` environment variable.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linting: `flake8 app/ tests/`
5. Run tests: `pytest`
6. Submit pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.