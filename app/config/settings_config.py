from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List

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
    api_prefix: str = Field("/api", description="API route prefix")
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
    timeout_seconds: int = Field(3600, description="Default task timeout")
    max_concurrent_tasks: int = Field(5, description="Maximum concurrent background tasks")
    retry_attempts: int = Field(3, description="Maximum task retry attempts")
    cleanup_interval_hours: int = Field(24, description="Task cleanup interval")
    
    class Config:
        env_prefix = "TASK_"

class AISettings(BaseSettings):
    # AI Service Configuration
    provider: str = Field("anthropic", description="AI provider: 'anthropic', 'mistral', or 'gemini'")
    
    # Anthropic Configuration
    anthropic_api_key: str = Field(..., description="Anthropic Claude API key")
    anthropic_model: str = Field("claude-3-sonnet-20240229", description="Default Claude model")
    anthropic_max_tokens: int = Field(2000, description="Maximum tokens for Claude responses")
    anthropic_timeout_seconds: int = Field(60, description="API request timeout")
    
    # Mistral Configuration
    mistral_api_key: str = Field(..., description="Mistral AI API key")
    mistral_model: str = Field("mistral-large-latest", description="Default Mistral model")
    mistral_max_tokens: int = Field(2000, description="Maximum tokens for Mistral responses")
    mistral_timeout_seconds: int = Field(60, description="API request timeout")
    
    # Google Gemini Configuration
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field("gemini-1.5-pro", description="Default Gemini model")
    gemini_max_tokens: int = Field(2000, description="Maximum tokens for Gemini responses")
    gemini_timeout_seconds: int = Field(60, description="API request timeout")
    gemini_temperature: float = Field(0.7, description="Gemini generation temperature")
    
    class Config:
        env_prefix = "AI_"

class LoggingSettings(BaseSettings):
    # Logging Configuration
    level: str = Field("INFO", description="Logging level")
    format: str = Field("text", description="Log format: json or text")
    file_path: Optional[str] = Field('app.log', description="Log file path")
    rotation_size: str = Field("10MB", description="Log file rotation size")
    retention_days: int = Field(30, description="Log retention period")
    
    class Config:
        env_prefix = "LOG_"

class Settings(BaseSettings):
    # Composite Settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    app: ApplicationSettings = Field(default_factory=ApplicationSettings)
    upload: FileUploadSettings = Field(default_factory=FileUploadSettings)
    task: BackgroundTaskSettings = Field(default_factory=BackgroundTaskSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ai: AISettings = Field(default_factory=AISettings)
    
    class Config:
        case_sensitive = False
        # Use absolute path to .env file relative to the backend directory
        env_file = Path(__file__).parent.parent.parent / ".env"
        env_file_encoding = "utf-8"
        extra = "allow"

# Global settings instance
_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings