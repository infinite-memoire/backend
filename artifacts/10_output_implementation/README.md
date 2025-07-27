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

#### Content Management
- `POST /api/v1/books` - Create new book
- `GET /api/v1/books/{book_id}/chapters` - Get book chapters
- `POST /api/v1/books/{book_id}/chapters` - Add new chapter
- `PUT /api/v1/chapters/{chapter_id}` - Update chapter content

#### AI Processing
- `POST /api/v1/ai/process-transcript` - Start AI processing
- `GET /api/v1/ai/status/{session_id}` - Check processing status
- `GET /api/v1/ai/results/{session_id}` - Get processing results

#### Publishing
- `POST /api/v1/publishing/start/{book_id}` - Start publishing workflow
- `PUT /api/v1/publishing/{workflow_id}/metadata` - Update publication metadata
- `POST /api/v1/publishing/{workflow_id}/preview` - Generate preview
- `POST /api/v1/publishing/{workflow_id}/publish` - Publish book

#### Marketplace
- `GET /api/v1/marketplace/books` - Browse published books
- `GET /api/v1/marketplace/book/{book_id}` - Get book details
- `POST /api/v1/marketplace/book/{book_id}/unpublish` - Unpublish book

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

## Development

### Project Structure
```
output_implementation/
├── app/
│   ├── services/
│   │   ├── content_storage.py
│   │   ├── version_tracking.py
│   │   ├── chapter_workflow.py
│   │   ├── validation.py
│   │   ├── chat_interface.py
│   │   ├── html_conversion.py
│   │   └── publishing.py
│   ├── api/
│   │   └── routes/
│   │       ├── content.py
│   │       ├── publishing.py
│   │       └── marketplace.py
│   ├── models/
│   │   ├── content.py
│   │   ├── publishing.py
│   │   └── validation.py
│   └── templates/
│       ├── book.html5
│       └── styles/
├── tests/
├── docs/
└── config/
```

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