# Docker Deployment Configuration

Based on configuration and logging setup for cloud deployment of the MVP backend.

## 1. Dockerfile for Backend Application

### 1.1 Multi-stage Production Dockerfile
```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY *.yaml ./

# Create necessary directories
RUN mkdir -p /var/log/memoire && \
    chown -R appuser:appuser /app /var/log/memoire

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 1.2 Development Dockerfile
```dockerfile
# Development version with debugging tools
FROM python:3.11-slim as development

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including debugging tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies with development packages
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy application code
COPY . .

# Create log directory
RUN mkdir -p /var/log/memoire

EXPOSE 8000

# Development command with auto-reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

## 2. Docker Compose Configuration

### 2.1 Complete Development Stack
```yaml
# docker-compose.yml
version: '3.8'

services:
  # Backend API service
  backend:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    container_name: memoire-backend
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - APP_DEBUG=true
      - DATABASE_FIRESTORE_EMULATOR_HOST=firestore:8080
      - DATABASE_NEO4J_URI=bolt://neo4j:7687
      - DATABASE_NEO4J_PASSWORD=development_password
      - LOG_LEVEL=DEBUG
      - LOG_FORMAT=text
    volumes:
      - ./app:/app/app
      - ./config:/app/config
      - ./logs:/var/log/memoire
    depends_on:
      - neo4j
      - firestore
    networks:
      - memoire-network
    restart: unless-stopped

  # Neo4j Graph Database
  neo4j:
    image: neo4j:5.14
    container_name: memoire-neo4j
    ports:
      - "7474:7474"  # Browser interface
      - "7687:7687"  # Bolt protocol
    environment:
      - NEO4J_AUTH=neo4j/development_password
      - NEO4J_PLUGINS=["graph-data-science"]
      - NEO4J_dbms_security_procedures_unrestricted=gds.*
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    networks:
      - memoire-network
    restart: unless-stopped

  # Firestore Emulator
  firestore:
    image: google/cloud-sdk:alpine
    container_name: memoire-firestore
    ports:
      - "8080:8080"  # Firestore emulator
      - "9090:9090"  # Firestore UI
    command: >
      sh -c "
        gcloud components install cloud-firestore-emulator --quiet &&
        gcloud beta emulators firestore start 
          --host-port=0.0.0.0:8080 
          --project=infinite-memoire
      "
    networks:
      - memoire-network
    restart: unless-stopped

  # Redis for caching (future use)
  redis:
    image: redis:7-alpine
    container_name: memoire-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - memoire-network
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
  redis_data:

networks:
  memoire-network:
    driver: bridge
```

### 2.2 Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
      target: base
    container_name: memoire-backend-prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - APP_DEBUG=false
      - DATABASE_FIRESTORE_PROJECT_ID=infinite-memoire
      - DATABASE_NEO4J_URI=bolt://neo4j:7687
      - DATABASE_NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
      - LOG_FILE_PATH=/var/log/memoire/app.log
    volumes:
      - ./logs:/var/log/memoire
      - ./secrets:/app/secrets:ro
    depends_on:
      - neo4j
    networks:
      - memoire-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  neo4j:
    image: neo4j:5.14
    container_name: memoire-neo4j-prod
    ports:
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
      - NEO4J_PLUGINS=["graph-data-science"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - ./neo4j-backups:/backups
    networks:
      - memoire-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G

volumes:
  neo4j_data:
  neo4j_logs:

networks:
  memoire-network:
    driver: bridge
```

## 3. Environment Configuration Files

### 3.1 Environment Variable Templates
```bash
# .env.template
# Copy to .env and fill in actual values

# Application Settings
ENVIRONMENT=development
APP_DEBUG=true
APP_NAME="Memoire Backend API"
APP_VERSION="1.0.0"

# Database Settings
DATABASE_FIRESTORE_PROJECT_ID=infinite-memoire
DATABASE_FIRESTORE_EMULATOR_HOST=localhost:8080
DATABASE_NEO4J_URI=bolt://localhost:7687
DATABASE_NEO4J_USER=neo4j
DATABASE_NEO4J_PASSWORD=your_password_here
DATABASE_NEO4J_DATABASE=neo4j

# Upload Settings
UPLOAD_MAX_UPLOAD_SIZE_MB=100
UPLOAD_CHUNK_SIZE_MB=5
UPLOAD_TIMEOUT_SECONDS=300

# Task Settings
TASK_TIMEOUT_SECONDS=3600
TASK_MAX_CONCURRENT_TASKS=5
TASK_RETRY_ATTEMPTS=3

# Logging Settings
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_PATH=/var/log/memoire/app.log
LOG_ROTATION_SIZE=10MB
LOG_RETENTION_DAYS=30

# CORS Settings
APP_CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

### 3.2 Production Environment
```bash
# .env.production
ENVIRONMENT=production
APP_DEBUG=false

# Use actual Firebase project (no emulator)
DATABASE_FIRESTORE_EMULATOR_HOST=

# Production Neo4j
DATABASE_NEO4J_URI=bolt://production-neo4j:7687
DATABASE_NEO4J_PASSWORD=${NEO4J_PRODUCTION_PASSWORD}

# Production logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_PATH=/var/log/memoire/app.log

# Restricted CORS for production
APP_CORS_ORIGINS=["https://yourdomain.com"]

# Resource limits
TASK_MAX_CONCURRENT_TASKS=10
UPLOAD_MAX_UPLOAD_SIZE_MB=200
```

## 4. Kubernetes Deployment Configuration

### 4.1 Backend Deployment
```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memoire-backend
  labels:
    app: memoire-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: memoire-backend
  template:
    metadata:
      labels:
        app: memoire-backend
    spec:
      containers:
      - name: backend
        image: memoire-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_FIRESTORE_PROJECT_ID
          value: "infinite-memoire"
        - name: DATABASE_NEO4J_URI
          value: "bolt://neo4j-service:7687"
        - name: DATABASE_NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neo4j-secret
              key: password
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          limits:
            cpu: 1000m
            memory: 1Gi
          requests:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: log-volume
          mountPath: /var/log/memoire
      volumes:
      - name: log-volume
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: memoire-backend
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 4.2 Neo4j Deployment
```yaml
# k8s/neo4j-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
spec:
  serviceName: neo4j-service
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.14
        ports:
        - containerPort: 7687
        - containerPort: 7474
        env:
        - name: NEO4J_AUTH
          valueFrom:
            secretKeyRef:
              name: neo4j-secret
              key: auth
        - name: NEO4J_PLUGINS
          value: '["graph-data-science"]'
        resources:
          limits:
            cpu: 2000m
            memory: 2Gi
          requests:
            cpu: 1000m
            memory: 1Gi
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: neo4j-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: neo4j-service
spec:
  selector:
    app: neo4j
  ports:
    - name: bolt
      protocol: TCP
      port: 7687
      targetPort: 7687
    - name: http
      protocol: TCP
      port: 7474
      targetPort: 7474
```

## 5. Build and Deployment Scripts

### 5.1 Build Script
```bash
#!/bin/bash
# scripts/build.sh

set -e

echo "Building Memoire Backend..."

# Build production image
docker build -t memoire-backend:latest .

# Tag for registry
if [ -n "$REGISTRY_URL" ]; then
    docker tag memoire-backend:latest $REGISTRY_URL/memoire-backend:latest
    docker tag memoire-backend:latest $REGISTRY_URL/memoire-backend:$(git rev-parse --short HEAD)
fi

echo "Build completed successfully!"
```

### 5.2 Development Setup Script
```bash
#!/bin/bash
# scripts/dev-setup.sh

set -e

echo "Setting up Memoire Backend development environment..."

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit .env file with your configuration"
fi

# Create log directory
mkdir -p logs

# Start development environment
echo "Starting development environment..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo "Checking service health..."
docker-compose ps

echo "Development environment is ready!"
echo "Backend API: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Neo4j Browser: http://localhost:7474"
echo "Firestore Emulator: http://localhost:9090"
```

### 5.3 Production Deployment Script
```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=${1:-production}

echo "Deploying to $ENVIRONMENT environment..."

# Build and push image
./scripts/build.sh

if [ -n "$REGISTRY_URL" ]; then
    echo "Pushing to registry..."
    docker push $REGISTRY_URL/memoire-backend:latest
    docker push $REGISTRY_URL/memoire-backend:$(git rev-parse --short HEAD)
fi

# Deploy based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Deploying to production..."
    docker-compose -f docker-compose.prod.yml up -d
elif [ "$ENVIRONMENT" = "k8s" ]; then
    echo "Deploying to Kubernetes..."
    kubectl apply -f k8s/
else
    echo "Unknown environment: $ENVIRONMENT"
    exit 1
fi

echo "Deployment completed!"
```

## 6. CI/CD Configuration

### 6.1 GitHub Actions Workflow
```yaml
# .github/workflows/backend.yml
name: Backend CI/CD

on:
  push:
    branches: [main, develop]
    paths: ['backend/**']
  pull_request:
    branches: [main]
    paths: ['backend/**']

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      neo4j:
        image: neo4j:5.14
        env:
          NEO4J_AUTH: neo4j/test_password
        ports:
          - 7687:7687
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt -r requirements-dev.txt
    
    - name: Run tests
      run: |
        cd backend
        pytest --cov=app --cov-report=xml
      env:
        DATABASE_NEO4J_URI: bolt://localhost:7687
        DATABASE_NEO4J_PASSWORD: test_password
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./backend/coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        cd backend
        docker build -t memoire-backend:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo "${{ secrets.REGISTRY_PASSWORD }}" | docker login ${{ secrets.REGISTRY_URL }} -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
        docker tag memoire-backend:${{ github.sha }} ${{ secrets.REGISTRY_URL }}/memoire-backend:${{ github.sha }}
        docker tag memoire-backend:${{ github.sha }} ${{ secrets.REGISTRY_URL }}/memoire-backend:latest
        docker push ${{ secrets.REGISTRY_URL }}/memoire-backend:${{ github.sha }}
        docker push ${{ secrets.REGISTRY_URL }}/memoire-backend:latest

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        # Add deployment logic here
        echo "Deploying to production..."
```

## 7. Monitoring and Logging

### 7.1 Docker Logging Configuration
```yaml
# docker-compose.override.yml for logging
version: '3.8'

services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        
  neo4j:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 7.2 Health Check Scripts
```bash
#!/bin/bash
# scripts/health-check.sh

echo "Checking service health..."

# Check backend API
BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/health)
if [ "$BACKEND_STATUS" = "200" ]; then
    echo "✓ Backend API is healthy"
else
    echo "✗ Backend API is unhealthy (Status: $BACKEND_STATUS)"
fi

# Check Neo4j
NEO4J_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7474)
if [ "$NEO4J_STATUS" = "200" ]; then
    echo "✓ Neo4j is healthy"
else
    echo "✗ Neo4j is unhealthy (Status: $NEO4J_STATUS)"
fi

echo "Health check completed!"
```