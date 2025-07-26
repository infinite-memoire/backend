import logging
import logging.config
from pathlib import Path
from pythonjsonlogger import jsonlogger
from app.config.settings import get_settings
import sys
import contextvars

def setup_logging():
    """Setup application logging configuration"""
    settings = get_settings()
    
    # Create logs directory if it doesn't exist
    if hasattr(settings.logging, 'log_file_path') and settings.logging.log_file_path:
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
    if hasattr(settings.logging, 'log_file_path') and settings.logging.log_file_path:
        from logging.handlers import RotatingFileHandler
        
        # Default rotation size if not specified
        rotation_size = "10MB"
        if hasattr(settings.logging, 'log_rotation_size'):
            rotation_size = settings.logging.log_rotation_size
            
        file_handler = RotatingFileHandler(
            settings.logging.log_file_path,
            maxBytes=_parse_size(rotation_size),
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
