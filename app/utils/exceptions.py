"""
Custom Exceptions

Application-specific exception classes for better error handling and messaging.
"""


class BaseAppException(Exception):
    """Base exception for all application-specific exceptions"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ValidationError(BaseAppException):
    """Raised when validation fails"""
    
    def __init__(self, message: str, field: str = None, value=None, details: dict = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value


class PublishingError(BaseAppException):
    """Raised when publishing operations fail"""
    
    def __init__(self, message: str, workflow_id: str = None, details: dict = None):
        super().__init__(message, "PUBLISHING_ERROR", details)
        self.workflow_id = workflow_id


class StorageError(BaseAppException):
    """Raised when storage operations fail"""
    
    def __init__(self, message: str, operation: str = None, details: dict = None):
        super().__init__(message, "STORAGE_ERROR", details)
        self.operation = operation


class AuthenticationError(BaseAppException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed", details: dict = None):
        super().__init__(message, "AUTH_ERROR", details)


class AuthorizationError(BaseAppException):
    """Raised when authorization fails"""
    
    def __init__(self, message: str = "Access denied", resource: str = None, details: dict = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)
        self.resource = resource


class NotFoundError(BaseAppException):
    """Raised when a resource is not found"""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, details: dict = None):
        super().__init__(message, "NOT_FOUND_ERROR", details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(BaseAppException):
    """Raised when there's a conflict in the operation"""
    
    def __init__(self, message: str, resource_type: str = None, details: dict = None):
        super().__init__(message, "CONFLICT_ERROR", details)
        self.resource_type = resource_type


class ProcessingError(BaseAppException):
    """Raised when processing operations fail"""
    
    def __init__(self, message: str, process_type: str = None, details: dict = None):
        super().__init__(message, "PROCESSING_ERROR", details)
        self.process_type = process_type


class ExternalServiceError(BaseAppException):
    """Raised when external service calls fail"""
    
    def __init__(self, message: str, service_name: str = None, status_code: int = None, details: dict = None):
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", details)
        self.service_name = service_name
        self.status_code = status_code


class RateLimitError(BaseAppException):
    """Raised when rate limits are exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, details: dict = None):
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.retry_after = retry_after


class ConfigurationError(BaseAppException):
    """Raised when there's a configuration error"""
    
    def __init__(self, message: str, config_key: str = None, details: dict = None):
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key


class BusinessLogicError(BaseAppException):
    """Raised when business logic constraints are violated"""
    
    def __init__(self, message: str, rule: str = None, details: dict = None):
        super().__init__(message, "BUSINESS_LOGIC_ERROR", details)
        self.rule = rule