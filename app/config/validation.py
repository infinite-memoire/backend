"""Configuration validation module"""
import os
from pathlib import Path
from app.config.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger("config.validation")

def validate_configuration():
    """Validate application configuration and environment"""
    logger.info("Starting configuration validation")
    
    settings = get_settings()
    
    # Validate database configuration
    _validate_database_config(settings.database)
    
    # Validate logging configuration
    _validate_logging_config(settings.logging)
    
    # Validate upload configuration
    _validate_upload_config(settings.upload)
    
    # Validate task configuration
    _validate_task_config(settings.task)
    
    logger.info("Configuration validation completed successfully")

def _validate_database_config(db_settings):
    """Validate database configuration"""
    logger.debug("Validating database configuration")
    
    # Validate Firestore configuration
    if not db_settings.firestore_project_id:
        raise ValueError("Firestore project ID is required")
    
    # Check if credentials file exists (if specified)
    if db_settings.firestore_credentials_path:
        creds_path = Path(db_settings.firestore_credentials_path)
        if not creds_path.exists():
            logger.warning("Firestore credentials file not found",
                          path=str(creds_path))
    
    # Validate Neo4j configuration
    if not db_settings.neo4j_password:
        raise ValueError("Neo4j password is required")
    
    logger.debug("Database configuration validated")

def _validate_logging_config(log_settings):
    """Validate logging configuration"""
    logger.debug("Validating logging configuration")
    
    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_settings.level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {log_settings.level}. Must be one of {valid_levels}")
    
    # Validate log format
    valid_formats = ["json", "text"]
    if log_settings.format not in valid_formats:
        raise ValueError(f"Invalid log format: {log_settings.format}. Must be one of {valid_formats}")
    
    # Check log file directory (if specified)
    if log_settings.file_path:
        log_file = Path(log_settings.file_path)
        log_dir = log_file.parent
        
        # Create directory if it doesn't exist
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create log directory",
                          directory=str(log_dir),
                          error=str(e))
    
    logger.debug("Logging configuration validated")

def _validate_upload_config(upload_settings):
    """Validate upload configuration"""
    logger.debug("Validating upload configuration")
    
    # Validate upload size limits
    if upload_settings.max_upload_size_mb <= 0:
        raise ValueError("Maximum upload size must be positive")
    
    if upload_settings.chunk_size_mb <= 0:
        raise ValueError("Chunk size must be positive")
    
    if upload_settings.chunk_size_mb > upload_settings.max_upload_size_mb:
        raise ValueError("Chunk size cannot be larger than maximum upload size")
    
    # Validate timeout
    if upload_settings.upload_timeout_seconds <= 0:
        raise ValueError("Upload timeout must be positive")
    
    # Validate audio formats
    if not upload_settings.allowed_audio_formats:
        raise ValueError("At least one audio format must be allowed")
    
    logger.debug("Upload configuration validated")

def _validate_task_config(task_settings):
    """Validate background task configuration"""
    logger.debug("Validating task configuration")
    
    # Validate timeout
    if task_settings.timeout_seconds <= 0:
        raise ValueError("Task timeout must be positive")
    
    # Validate concurrency limits
    if task_settings.max_concurrent_tasks <= 0:
        raise ValueError("Maximum concurrent tasks must be positive")
    
    # Validate retry attempts
    if task_settings.retry_attempts < 0:
        raise ValueError("Task retry attempts cannot be negative")
    
    # Validate cleanup interval
    if task_settings.cleanup_interval_hours <= 0:
        raise ValueError("Task cleanup interval must be positive")
    
    logger.debug("Task configuration validated")
