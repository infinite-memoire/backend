# Configuration and Logging Setup - Hierarchical Breakdown

Based on background task system requirements and MVP configuration strategy.

## 1. Configuration Management System
### 1.1 Environment Configuration Structure
#### 1.1.1 Settings Architecture
```python
# app/config/settings.py
from pydantic import BaseSettings, Field
from typing import Optional, List
import os
from pathlib import Path

class DatabaseSettings(BaseSettings):
    # Firestore Configuration
    firestore_project_id: str = Field(..., description="Firebase project ID")
    firestore_credentials_path: Optional[str] = Field(None, description="Path to service account JSON")
    firestore_emulator_host: Optional[str] = Field(None, description="Firestore emulator host for development")
    
    # Neo4j Configuration
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field("neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")
    neo4j_database: str = Field("neo4j", description="Neo4j database name")
    
    class Config:
        env_prefix = "DATABASE_"

class ApplicationSettings(BaseSettings):
    # Basic Application Settings
    app_name: str = Field("Memoire Backend API", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    debug: bool = Field(False, description="Debug mode")
    environment: str = Field("development", description="Environment name")
    
    # API Configuration
    api_prefix: str = Field("/api/v1", description="API route prefix")
    docs_url: str = Field("/docs", description="API documentation URL")
    redoc_url: str = Field("/redoc", description="ReDoc documentation URL")
    
    # CORS Configuration
    cors_origins: List[str] = Field(["*"], description="Allowed CORS origins")
    cors_allow_credentials: bool = Field(True, description="Allow CORS credentials")
    cors_allow_methods: List[str] = Field(["*"], description="Allowed CORS methods")
    cors_allow_headers: List[str] = Field(["*"], description="Allowed CORS headers")
    
    class Config:
        env_prefix = "APP_"

class FileUploadSettings(BaseSettings):
    # Upload Configuration
    max_upload_size_mb: int = Field(100, description="Maximum upload size in MB")
    chunk_size_mb: int = Field(5, description="Upload chunk size in MB")
    allowed_audio_formats: List[str] = Field(
        ["audio/mpeg", "audio/wav", "audio/mp4", "audio/aac", "audio/ogg"],
        description="Allowed audio MIME types"
    )
    upload_timeout_seconds: int = Field(300, description="Upload timeout in seconds")
    
    class Config:
        env_prefix = "UPLOAD_"

class BackgroundTaskSettings(BaseSettings):
    # Task Configuration
    task_timeout_seconds: int = Field(3600, description="Default task timeout")
    max_concurrent_tasks: int = Field(5, description="Maximum concurrent background tasks")
    task_retry_attempts: int = Field(3, description="Maximum task retry attempts")
    task_cleanup_interval_hours: int = Field(24, description="Task cleanup interval")
    
    class Config:
        env_prefix = "TASK_"

class LoggingSettings(BaseSettings):
    # Logging Configuration
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("json", description="Log format: json or text")
    log_file_path: Optional[str] = Field(None, description="Log file path")
    log_rotation_size: str = Field("10MB", description="Log file rotation size")
    log_retention_days: int = Field(30, description="Log retention period")
    
    class Config:
        env_prefix = "LOG_"

class Settings(BaseSettings):
    # Composite Settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    app: ApplicationSettings = Field(default_factory=ApplicationSettings)
    upload: FileUploadSettings = Field(default_factory=FileUploadSettings)
    tasks: BackgroundTaskSettings = Field(default_factory=BackgroundTaskSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

#### 1.1.2 Configuration Files Structure
```yaml
# config.yaml - Base configuration
app:
  name: "Memoire Backend API"
  version: "1.0.0"
  debug: false
  environment: "development"

database:
  firestore_project_id: "infinite-memoire"
  neo4j_uri: "bolt://localhost:7687"
  neo4j_user: "neo4j"
  neo4j_database: "memoire"

upload:
  max_upload_size_mb: 100
  chunk_size_mb: 5
  allowed_audio_formats:
    - "audio/mpeg"
    - "audio/wav"
    - "audio/mp4"
    - "audio/aac"
    - "audio/ogg"

tasks:
  task_timeout_seconds: 3600
  max_concurrent_tasks: 5
  task_retry_attempts: 3

logging:
  log_level: "INFO"
  log_format: "json"
  log_retention_days: 30
```

### 1.2 Environment-Specific Configuration
#### 1.2.1 Development Configuration
```yaml
# config-dev.yaml
app:
  debug: true
  environment: "development"
  docs_url: "/docs"
  redoc_url: "/redoc"

database:
  firestore_emulator_host: "localhost:8080"
  neo4j_uri: "bolt://localhost:7687"

logging:
  log_level: "DEBUG"
  log_format: "text"

tasks:
  task_timeout_seconds: 1800
  max_concurrent_tasks: 3
```

#### 1.2.2 Production Configuration
```yaml
# config-prod.yaml
app:
  debug: false
  environment: "production"
  docs_url: null  # Disable docs in production
  redoc_url: null

database:
  firestore_emulator_host: null

logging:
  log_level: "INFO"
  log_format: "json"
  log_file_path: "/var/log/memoire/app.log"

tasks:
  task_timeout_seconds: 3600
  max_concurrent_tasks: 10
```

## 2. Logging System Implementation
### 2.1 Structured Logging Setup
#### 2.1.1 Logger Configuration
```python
# app/config/logging.py
import logging
import logging.config
from pathlib import Path
from pythonjsonlogger import jsonlogger
from app.config.settings import get_settings
import sys

def setup_logging():
    """Setup application logging configuration"""
    settings = get_settings()
    
    # Create logs directory if it doesn't exist
    if settings.logging.log_file_path:
        log_file = Path(settings.logging.log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    if settings.logging.log_format == "json":
        _setup_json_logging(settings)
    else:
        _setup_text_logging(settings)
    
    # Set up request ID context
    _setup_request_context()

def _setup_json_logging(settings):
    """Setup JSON structured logging"""
    
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.logging.log_level))
    
    handlers = [console_handler]
    
    # File handler if configured
    if settings.logging.log_file_path:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            settings.logging.log_file_path,
            maxBytes=_parse_size(settings.logging.log_rotation_size),
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, settings.logging.log_level))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.logging.log_level),
        handlers=handlers,
        force=True
    )

def _setup_text_logging(settings):
    """Setup text-based logging for development"""
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.logging.log_level))
    
    logging.basicConfig(
        level=getattr(logging, settings.logging.log_level),
        handlers=[console_handler],
        force=True
    )

def _setup_request_context():
    """Setup request context for logging"""
    import contextvars
    
    # Request ID context variable
    request_id_var = contextvars.ContextVar('request_id', default=None)
    
    # Custom formatter that includes request ID
    class RequestContextFormatter(logging.Formatter):
        def format(self, record):
            request_id = request_id_var.get()
            if request_id:
                record.request_id = request_id
            else:
                record.request_id = "no-request"
            return super().format(record)
    
    return request_id_var

def _parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes"""
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)
```

#### 2.1.2 Application-Specific Loggers
```python
# app/utils/logging.py
import logging
from typing import Dict, Any
import functools
import time
import traceback

class AppLogger:
    """Application-specific logger with structured logging"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self._log(logging.INFO, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self._log(logging.ERROR, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self._log(logging.WARNING, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with structured data"""
        extra = {
            'component': self.logger.name,
            'data': kwargs
        }
        self.logger.log(level, message, extra=extra)

def get_logger(name: str) -> AppLogger:
    """Get application logger for a specific component"""
    return AppLogger(name)

# Performance logging decorator
def log_performance(logger: AppLogger):
    """Decorator to log function performance"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__qualname__}"
            
            try:
                logger.debug(f"Starting {function_name}", 
                           function=function_name, args_count=len(args), kwargs_count=len(kwargs))
                
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(f"Completed {function_name}",
                          function=function_name, duration_seconds=duration, status="success")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {function_name}",
                           function=function_name, duration_seconds=duration, 
                           status="error", error_type=type(e).__name__, error_message=str(e))
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__qualname__}"
            
            try:
                logger.debug(f"Starting {function_name}",
                           function=function_name, args_count=len(args), kwargs_count=len(kwargs))
                
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(f"Completed {function_name}",
                          function=function_name, duration_seconds=duration, status="success")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {function_name}",
                           function=function_name, duration_seconds=duration,
                           status="error", error_type=type(e).__name__, error_message=str(e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

### 2.2 Service-Specific Logging
#### 2.2.1 Database Operation Logging
```python
# app/services/firestore.py (logging integration)
from app.utils.logging import get_logger, log_performance

logger = get_logger("firestore")

class FirestoreService:
    def __init__(self):
        self.db = firestore.Client(project=settings.firestore_project_id)
        logger.info("Firestore service initialized", project_id=settings.firestore_project_id)
    
    @log_performance(logger)
    async def create_audio_record(self, audio_data: dict) -> str:
        try:
            doc_ref = self.db.collection('audio_files').document()
            doc_ref.set(audio_data)
            
            logger.info("Audio record created",
                       document_id=doc_ref.id,
                       collection="audio_files",
                       filename=audio_data.get("filename"),
                       file_size=audio_data.get("file_size_bytes"))
            
            return doc_ref.id
            
        except Exception as e:
            logger.error("Failed to create audio record",
                        collection="audio_files",
                        filename=audio_data.get("filename"),
                        error_type=type(e).__name__,
                        error_message=str(e))
            raise
    
    @log_performance(logger)
    async def get_audio_record(self, audio_id: str) -> dict:
        try:
            doc_ref = self.db.collection('audio_files').document(audio_id)
            doc = doc_ref.get()
            
            exists = doc.exists
            logger.debug("Audio record retrieved",
                        document_id=audio_id,
                        collection="audio_files",
                        exists=exists)
            
            return doc.to_dict() if exists else None
            
        except Exception as e:
            logger.error("Failed to retrieve audio record",
                        document_id=audio_id,
                        collection="audio_files",
                        error_type=type(e).__name__,
                        error_message=str(e))
            raise
```

#### 2.2.2 Background Task Logging
```python
# app/services/background_tasks.py (logging integration)
from app.utils.logging import get_logger, log_performance

logger = get_logger("background_tasks")

class BackgroundTaskManager:
    @log_performance(logger)
    async def create_task(self, task_type: TaskType, task_function: Callable, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        
        logger.info("Background task created",
                   task_id=task_id,
                   task_type=task_type.value,
                   function_name=task_function.__name__,
                   metadata=kwargs.get("metadata", {}))
        
        # ... task creation logic
        
        return task_id
    
    @log_performance(logger)
    async def _execute_task(self, task_id: str, task_function: Callable, *args, **kwargs):
        logger.info("Task execution started",
                   task_id=task_id,
                   function_name=task_function.__name__)
        
        try:
            result = await task_function(task_id, *args, **kwargs)
            
            logger.info("Task execution completed successfully",
                       task_id=task_id,
                       function_name=task_function.__name__,
                       result_size=len(str(result)) if result else 0)
            
            return result
            
        except Exception as e:
            logger.error("Task execution failed",
                        task_id=task_id,
                        function_name=task_function.__name__,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=traceback.format_exc())
            raise
```

## 3. Configuration Loading and Validation
### 3.1 Configuration Loader
```python
# app/config/loader.py
import yaml
from pathlib import Path
from typing import Dict, Any
import os
from app.utils.logging import get_logger

logger = get_logger("config")

class ConfigurationLoader:
    """Load and merge configuration from multiple sources"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.environment = os.getenv("ENVIRONMENT", "development")
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration with environment-specific overrides"""
        
        # Load base configuration
        base_config = self._load_yaml_file("config.yaml")
        logger.info("Loaded base configuration", file="config.yaml")
        
        # Load environment-specific configuration
        env_config_file = f"config-{self.environment}.yaml"
        env_config = self._load_yaml_file(env_config_file)
        
        if env_config:
            logger.info("Loaded environment configuration",
                       environment=self.environment,
                       file=env_config_file)
            base_config = self._merge_configs(base_config, env_config)
        else:
            logger.warning("No environment-specific configuration found",
                          environment=self.environment,
                          expected_file=env_config_file)
        
        # Apply environment variable overrides
        config_with_env = self._apply_env_overrides(base_config)
        
        logger.info("Configuration loaded successfully",
                   environment=self.environment,
                   config_keys=list(config_with_env.keys()))
        
        return config_with_env
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to load configuration file",
                        file=filename,
                        error_type=type(e).__name__,
                        error_message=str(e))
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # This would implement environment variable override logic
        # For now, return config as-is since Pydantic BaseSettings handles this
        return config

# Global configuration loader
config_loader = ConfigurationLoader()
```

### 3.2 Configuration Validation
```python
# app/config/validation.py
from app.config.settings import get_settings
from app.utils.logging import get_logger
import sys

logger = get_logger("config_validation")

def validate_configuration():
    """Validate application configuration on startup"""
    
    try:
        settings = get_settings()
        
        # Validate database settings
        _validate_database_config(settings.database)
        
        # Validate upload settings
        _validate_upload_config(settings.upload)
        
        # Validate task settings
        _validate_task_config(settings.tasks)
        
        logger.info("Configuration validation successful",
                   environment=settings.app.environment)
        
    except Exception as e:
        logger.error("Configuration validation failed",
                    error_type=type(e).__name__,
                    error_message=str(e))
        sys.exit(1)

def _validate_database_config(db_config):
    """Validate database configuration"""
    
    # Validate Firestore
    if not db_config.firestore_project_id:
        raise ValueError("Firestore project ID is required")
    
    # Validate Neo4j
    if not db_config.neo4j_uri:
        raise ValueError("Neo4j URI is required")
    
    if not db_config.neo4j_password:
        raise ValueError("Neo4j password is required")
    
    logger.debug("Database configuration valid",
                firestore_project=db_config.firestore_project_id,
                neo4j_uri=db_config.neo4j_uri)

def _validate_upload_config(upload_config):
    """Validate upload configuration"""
    
    if upload_config.max_upload_size_mb <= 0:
        raise ValueError("Maximum upload size must be positive")
    
    if upload_config.chunk_size_mb <= 0:
        raise ValueError("Chunk size must be positive")
    
    if upload_config.chunk_size_mb > upload_config.max_upload_size_mb:
        raise ValueError("Chunk size cannot be larger than maximum upload size")
    
    logger.debug("Upload configuration valid",
                max_size_mb=upload_config.max_upload_size_mb,
                chunk_size_mb=upload_config.chunk_size_mb)

def _validate_task_config(task_config):
    """Validate task configuration"""
    
    if task_config.task_timeout_seconds <= 0:
        raise ValueError("Task timeout must be positive")
    
    if task_config.max_concurrent_tasks <= 0:
        raise ValueError("Maximum concurrent tasks must be positive")
    
    logger.debug("Task configuration valid",
                timeout_seconds=task_config.task_timeout_seconds,
                max_concurrent=task_config.max_concurrent_tasks)
```

## 4. Application Initialization
### 4.1 Startup Configuration
```python
# app/main.py (enhanced with configuration and logging)
from fastapi import FastAPI
from app.config.settings import get_settings
from app.config.logging import setup_logging
from app.config.validation import validate_configuration
from app.utils.logging import get_logger

# Setup logging first
setup_logging()
logger = get_logger("main")

# Validate configuration
validate_configuration()

# Get settings
settings = get_settings()

def create_app() -> FastAPI:
    """Create FastAPI application with proper configuration"""
    
    logger.info("Starting application initialization",
               app_name=settings.app.app_name,
               version=settings.app.app_version,
               environment=settings.app.environment)
    
    app = FastAPI(
        title=settings.app.app_name,
        version=settings.app.app_version,
        debug=settings.app.debug,
        docs_url=settings.app.docs_url,
        redoc_url=settings.app.redoc_url
    )
    
    # Configure CORS
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=settings.app.cors_allow_credentials,
        allow_methods=settings.app.cors_allow_methods,
        allow_headers=settings.app.cors_allow_headers,
    )
    
    # Add request logging middleware
    from app.middleware.logging import LoggingMiddleware
    app.add_middleware(LoggingMiddleware)
    
    # Include API routes
    from app.api.routes import audio, transcripts, health
    app.include_router(audio.router, prefix=f"{settings.app.api_prefix}/audio")
    app.include_router(transcripts.router, prefix=f"{settings.app.api_prefix}/transcripts")
    app.include_router(health.router, prefix=f"{settings.app.api_prefix}/health")
    
    logger.info("Application initialization completed",
               routes_count=len(app.routes),
               middleware_count=len(app.user_middleware))
    
    return app

app = create_app()

@app.on_event("startup")
async def startup_event():
    """Application startup event handler"""
    logger.info("Application startup initiated")
    
    # Initialize database connections
    from app.services.firestore import firestore_service
    from app.services.neo4j import neo4j_service
    
    # Test database connections
    await firestore_service.test_connection()
    await neo4j_service.test_connection()
    
    logger.info("Application startup completed successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler"""
    logger.info("Application shutdown initiated")
    
    # Close database connections
    from app.services.neo4j import neo4j_service
    neo4j_service.close()
    
    logger.info("Application shutdown completed")
```