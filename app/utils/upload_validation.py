from app.config.settings import get_settings
from app.utils.logging import get_logger
import mimetypes
from typing import Tuple, Optional

logger = get_logger("upload_validation")

class UploadValidationError(Exception):
    """Exception raised for upload validation errors"""
    pass

class FileValidator:
    def __init__(self):
        self.settings = get_settings()
        
    def validate_upload_request(
        self, 
        filename: str, 
        file_size: int, 
        content_type: str,
        chunk_size: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate upload request parameters"""
        
        try:
            # Validate file size
            max_size_bytes = self.settings.upload.max_upload_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                raise UploadValidationError(
                    f"File size {file_size} exceeds maximum allowed size {max_size_bytes}"
                )
            
            if file_size <= 0:
                raise UploadValidationError("File size must be greater than 0")
            
            # Validate content type
            if content_type not in self.settings.upload.allowed_audio_formats:
                raise UploadValidationError(
                    f"Content type {content_type} not allowed. "
                    f"Allowed types: {self.settings.upload.allowed_audio_formats}"
                )
            
            # Validate filename
            if not filename or len(filename.strip()) == 0:
                raise UploadValidationError("Filename cannot be empty")
            
            if len(filename) > 255:
                raise UploadValidationError("Filename too long (max 255 characters)")
            
            # Validate chunk size
            max_chunk_size = self.settings.upload.chunk_size_mb * 1024 * 1024
            if chunk_size > max_chunk_size:
                raise UploadValidationError(
                    f"Chunk size {chunk_size} exceeds maximum {max_chunk_size}"
                )
            
            if chunk_size <= 0:
                raise UploadValidationError("Chunk size must be greater than 0")
            
            # Validate file extension matches content type
            expected_content_type = mimetypes.guess_type(filename)[0]
            if expected_content_type and expected_content_type != content_type:
                logger.warning("Content type mismatch",
                             filename=filename,
                             declared_type=content_type,
                             expected_type=expected_content_type)
            
            logger.info("Upload validation successful",
                       filename=filename,
                       file_size=file_size,
                       content_type=content_type,
                       chunk_size=chunk_size)
            
            return True, None
            
        except UploadValidationError as e:
            logger.error("Upload validation failed",
                        filename=filename,
                        error_message=str(e))
            return False, str(e)
        except Exception as e:
            logger.error("Unexpected validation error",
                        filename=filename,
                        error_type=type(e).__name__,
                        error_message=str(e))
            return False, f"Validation error: {str(e)}"

    def validate_chunk_data(
        self, 
        chunk_data: bytes, 
        expected_size: int, 
        chunk_index: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate individual chunk data"""
        
        try:
            actual_size = len(chunk_data)
            
            # For last chunk, size may be smaller
            if actual_size > expected_size:
                raise UploadValidationError(
                    f"Chunk {chunk_index} size {actual_size} exceeds expected {expected_size}"
                )
            
            if actual_size == 0:
                raise UploadValidationError(f"Chunk {chunk_index} is empty")
            
            logger.debug("Chunk validation successful",
                        chunk_index=chunk_index,
                        actual_size=actual_size,
                        expected_size=expected_size)
            
            return True, None
            
        except UploadValidationError as e:
            logger.error("Chunk validation failed",
                        chunk_index=chunk_index,
                        error_message=str(e))
            return False, str(e)
        except Exception as e:
            logger.error("Unexpected chunk validation error",
                        chunk_index=chunk_index,
                        error_type=type(e).__name__,
                        error_message=str(e))
            return False, f"Chunk validation error: {str(e)}"

# Global validator instance
file_validator = FileValidator()
