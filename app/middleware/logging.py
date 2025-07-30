import time
import uuid
import contextvars
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.logging_utils import get_logger

# Request ID context variable
request_id_var = contextvars.ContextVar('request_id', default=None)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging with structured data"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger("middleware.logging")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        
        # Start timing
        start_time = time.time()
        
        # Log incoming request
        self.logger.info("Request started",
                        request_id=request_id,
                        method=request.method,
                        url=str(request.url),
                        path=request.url.path,
                        query_params=dict(request.query_params),
                        client_host=request.client.host if request.client else None,
                        user_agent=request.headers.get("user-agent"))
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful response
            self.logger.info("Request completed",
                            request_id=request_id,
                            method=request.method,
                            path=request.url.path,
                            status_code=response.status_code,
                            duration_seconds=round(duration, 4),
                            response_size=response.headers.get("content-length"))
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration = time.time() - start_time
            
            # Log failed request
            self.logger.error("Request failed",
                             request_id=request_id,
                             method=request.method,
                             path=request.url.path,
                             duration_seconds=round(duration, 4),
                             error=str(e),
                             error_type=type(e).__name__)
            
            # Re-raise the exception
            raise

def get_request_id() -> str:
    """Get the current request ID from context"""
    return request_id_var.get() or "no-request"
