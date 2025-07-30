import logging
import logging.config
from pathlib import Path
try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    # Fallback if pythonjsonlogger is not available
    jsonlogger = None
from app.config.settings_config import get_settings
import sys
import contextvars

def setup_logging():
    """Setup application logging configuration"""
    settings = get_settings()
    
    # Create logs directory if it doesn't exist
    if hasattr(settings.logging, 'log_file_path') and settings.logging.file_path:
        log_file = Path(settings.logging.file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    if settings.logging.format == "json":
        _setup_json_logging(settings)
    else:
        _setup_text_logging(settings)
    
    # Set up request ID context
    _setup_request_context()

def _setup_json_logging(settings):
    """Setup JSON structured logging"""
    
    if jsonlogger is None:
        # Fallback to text logging if jsonlogger is not available
        _setup_text_logging(settings)
        return
    
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.logging.level))
    
    handlers = [console_handler]
    
    # File handler if configured
    if hasattr(settings.logging, 'log_file_path') and settings.logging.file_path:
        from logging.handlers import RotatingFileHandler
        
        # Default rotation size if not specified
        rotation_size = "10MB"
        if hasattr(settings.logging, 'log_rotation_size'):
            rotation_size = settings.logging.rotation_size
            
        file_handler = RotatingFileHandler(
            settings.logging.file_path,
            maxBytes=_parse_size(rotation_size),
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, settings.logging.level))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        handlers=handlers,
        force=True
    )

def _setup_text_logging(settings):
    """Setup text-based logging for development"""
    
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors and better structure"""
        
        COLORS = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
        }
        RESET = '\033[0m'
        
        def format(self, record):
            # Add color to log level
            if hasattr(record, 'levelname'):
                color = self.COLORS.get(record.levelname, '')
                record.colored_levelname = f"{color}{record.levelname}{self.RESET}"
            else:
                record.colored_levelname = record.levelname
            
            # Format component name
            component = getattr(record, 'component', record.name)
            record.short_component = component.split('.')[-1][:15]  # Last part, max 15 chars
            
            return super().format(record)
    
    formatter = ColoredFormatter(
        fmt='%(asctime)s | %(short_component)-15s | %(colored_levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.logging.level))
    
    # Clear existing handlers and force our configuration
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        handlers=[console_handler],
        force=True
    )
    
    print(f"Text logging configured. Format: {formatter._fmt}, Level: {settings.logging.level}")  # Debug print

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
