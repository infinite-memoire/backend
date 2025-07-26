from pydantic_settings import BaseSettings
from pydantic import Field
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