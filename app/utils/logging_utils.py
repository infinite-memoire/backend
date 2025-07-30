import logging
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
        try:
            # Import here to avoid circular imports
            from app.config.logging_config import get_current_logging_format
            current_format = get_current_logging_format()
        except ImportError:
            # Fallback if config not available
            current_format = "text"
        
        extra = {
            'component': self.logger.name
        }
        
        if current_format == "json":
            # For JSON logging, add kwargs to extra and keep message clean
            extra.update(kwargs)
            self.logger.log(level, message, extra=extra)
        else:
            # For text logging, format kwargs as readable key=value pairs in message
            if kwargs:
                formatted_data = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                message = f"{message} | {formatted_data}"
            
            # Only add component to extra for text logging
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
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Function {func.__name__} completed",
                           duration_seconds=duration,
                           function=func.__name__)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Function {func.__name__} failed",
                            duration_seconds=duration,
                            function=func.__name__,
                            error=str(e),
                            traceback=traceback.format_exc())
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Function {func.__name__} completed",
                           duration_seconds=duration,
                           function=func.__name__)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Function {func.__name__} failed",
                            duration_seconds=duration,
                            function=func.__name__,
                            error=str(e),
                            traceback=traceback.format_exc())
                raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def log_exception(logger: AppLogger, message: str = "An error occurred"):
    """Decorator to log exceptions with context"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(message,
                            function=func.__name__,
                            error=str(e),
                            error_type=type(e).__name__,
                            traceback=traceback.format_exc())
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(message,
                            function=func.__name__,
                            error=str(e),
                            error_type=type(e).__name__,
                            traceback=traceback.format_exc())
                raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
