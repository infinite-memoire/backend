# Memoire Backend - MVP Foundation Implementation

This is the production-ready implementation of the Memoire Backend MVP, featuring FastAPI, Firestore, Neo4j, and chunked file upload capabilities.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **Chunked File Upload**: Robust handling of large audio files with validation and retry logic
- **Firestore Integration**: NoSQL database for audio metadata and upload sessions
- **Neo4j Integration**: Graph database for storyline relationships
- **Background Task Processing**: Async processing for long-running operations
- **Structured Logging**: JSON-formatted logging with performance tracking
- **Configuration Management**: Environment-based configuration with validation
- **Docker Support**: Containerized deployment with development and production configs

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git

### Development Setup

1. **Clone and setup environment**:
```bash
cp .env.template .env
# Edit .env with your configuration
```

2. **Start development environment**:
```bash
docker-compose up -d
```

3. **Install dependencies (for local development)**:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

4. **Run tests**:
```bash
pytest
```

5. **Start development server**:
```bash
uvicorn app.main:app --reload
```

### Services

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474 (user: neo4j, password: development_password)
- **Firestore Emulator UI**: http://localhost:9090

## API Endpoints

### Upload Endpoints

- `POST /api/v1/upload/initiate` - Initiate chunked upload
- `PUT /api/v1/upload/chunk/{upload_id}/{chunk_index}` - Upload chunk
- `POST /api/v1/upload/complete/{upload_id}` - Complete upload
- `GET /api/v1/upload/status/{upload_id}` - Get upload status

### Health Check

- `GET /api/v1/health` - Service health check

## Architecture

### Project Structure

```
app/
├── api/
│   └── routes/          # API route handlers
├── config/              # Configuration management
├── models/              # Data models
├── services/            # Business logic services
├── utils/               # Utility functions
└── middleware/          # Custom middleware

tests/                   # Test suite
config/                  # Configuration files
docs/                    # Documentation
```

### Database Schema

**Firestore Collections**:
- `audio_files` - Audio file metadata
- `upload_sessions` - Chunked upload session data
- `upload_chunks` - Individual chunk data
- `processing_tasks` - Background task status

**Neo4j Schema**:
- Audio nodes with metadata
- Chunk relationships for storyline analysis

## Configuration

### Environment Variables

All configuration is managed through environment variables with the following prefixes:

- `APP_*` - Application settings
- `*` - Database connection settings
- `UPLOAD_*` - File upload configuration
- `TASK_*` - Background task settings
- `LOG_*` - Logging configuration

### Key Settings

- `UPLOAD_MAX_UPLOAD_SIZE_MB`: Maximum file size (default: 100MB)
- `UPLOAD_CHUNK_SIZE_MB`: Chunk size for uploads (default: 5MB)
- `FIRESTORE_PROJECT_ID`: Firebase project ID
- `NEO4J_URI`: Neo4j connection URI
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_upload_service.py
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/

# Security scan
bandit -r app/
```

### Database Setup

**For Development (using emulators)**:
- Firestore emulator runs automatically with docker-compose
- Neo4j runs in development mode with default credentials

**For Production**:
- Configure Firebase service account credentials
- Set up production Neo4j instance
- Update environment variables accordingly

## Deployment

### Docker Production

```bash
# Build production image
docker build -t memoire-backend:latest .

# Run production stack
docker-compose -f docker-compose.prod.yml up -d
```

### Environment-Specific Deployment

1. **Development**: Uses emulators and debug settings
2. **Production**: Uses real databases and optimized settings

### Health Checks

The application includes comprehensive health checks:
- Database connectivity
- Service availability
- Resource utilization

## Monitoring

### Logging

- Structured JSON logging in production
- Human-readable text logging in development
- Performance tracking for all API calls
- Request/response logging with correlation IDs

### Metrics

- Upload success/failure rates
- Processing times
- Background task completion rates
- Database connection health

## Security

- Input validation for all endpoints
- File type and size restrictions
- Secure file storage in Firestore
- Non-root Docker containers
- Environment-based secrets management

## Troubleshooting

### Common Issues

1. **Connection refused errors**: Ensure all services are running with `docker-compose ps`
2. **Upload failures**: Check file size limits and chunk validation
3. **Database connection issues**: Verify environment variables and service health

### Debug Mode

Set `APP_DEBUG=true` and `LOG_LEVEL=DEBUG` for detailed logging.

### Log Analysis

```bash
# View application logs
docker-compose logs backend

# Follow logs in real-time
docker-compose logs -f backend

# View specific service logs
docker-compose logs neo4j
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

## License

MIT License - see LICENSE file for details