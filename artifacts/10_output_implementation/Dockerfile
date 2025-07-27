# Multi-stage Dockerfile for Output Management & Publishing System
# Optimized for production deployment with minimal image size

# Stage 1: Base Python image with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Pandoc for document conversion
    pandoc \
    # System utilities
    curl \
    wget \
    git \
    # Build tools for Python packages
    gcc \
    g++ \
    make \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    # PDF processing
    poppler-utils \
    # Font support
    fonts-liberation \
    fonts-dejavu-core \
    # LaTeX for advanced document generation
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base as dependencies

# Create application directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies only if DEV_MODE is set
ARG DEV_MODE=false
RUN if [ "$DEV_MODE" = "true" ] ; then \
    pip install --no-cache-dir -r requirements-dev.txt ; \
    fi

# Stage 3: Application
FROM dependencies as application

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY tests/ ./tests/
COPY config/ ./config/
COPY templates/ ./templates/
COPY docs/ ./docs/

# Copy configuration files
COPY .env.template .
COPY docker-compose.yml .
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/uploads \
    /app/outputs \
    /app/previews \
    /app/cache \
    /app/static

# Set proper permissions
RUN chown -R appuser:appuser /app
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Create templates directory structure
RUN mkdir -p /app/templates/styles && \
    mkdir -p /app/templates/scripts && \
    mkdir -p /app/templates/images

# Copy default templates
COPY templates/ /app/templates/

# Install additional NLP models if needed
RUN python -m spacy download en_core_web_sm 2>/dev/null || true

# Stage 4: Production
FROM application as production

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 5: Development
FROM application as development

# Install development tools
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit \
    jupyter \
    ipdb

# Switch to non-root user
USER appuser

# Expose port and debugger port
EXPOSE 8000 5678

# Development command with hot reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# Build arguments for multi-stage selection
ARG BUILD_ENV=production
FROM ${BUILD_ENV} as final

# Labels for metadata
LABEL maintainer="Memoire AI Team" \
      version="1.0.0" \
      description="Output Management & Publishing System for Memoire AI Book Generator" \
      org.opencontainers.image.title="Memoire Output Management" \
      org.opencontainers.image.description="Backend service for book content management and publishing" \
      org.opencontainers.image.vendor="Memoire AI" \
      org.opencontainers.image.version="1.0.0"

# Runtime environment variables
ENV PYTHONPATH=/app \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    MAX_WORKERS=8 \
    TIMEOUT=300 \
    KEEP_ALIVE=2

# Volume mounts for persistent data
VOLUME ["/app/uploads", "/app/outputs", "/app/logs", "/app/cache"]

# Set the final command
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS} --timeout-keep-alive ${KEEP_ALIVE}"]