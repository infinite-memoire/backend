import io
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import tempfile
import os
import asyncio
from pydub import AudioSegment
from pydub.utils import which
from app.utils.logging_utils import get_logger, log_performance
from app.config.settings_config import get_settings

logger = get_logger("audio_preprocessor")

class AudioFormat:
    """Supported audio format definitions"""
    MP3 = "mp3"
    WAV = "wav"
    M4A = "m4a"
    AAC = "aac"
    OGG = "ogg"
    FLAC = "flac"
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        return [cls.MP3, cls.WAV, cls.M4A, cls.AAC, cls.OGG, cls.FLAC]
    
    @classmethod
    def get_mime_type(cls, format_type: str) -> str:
        mime_types = {
            cls.MP3: "audio/mpeg",
            cls.WAV: "audio/wav",
            cls.M4A: "audio/mp4",
            cls.AAC: "audio/aac",
            cls.OGG: "audio/ogg",
            cls.FLAC: "audio/flac"
        }
        return mime_types.get(format_type, "audio/unknown")

class AudioPreprocessor:
    """
    Comprehensive audio preprocessing service supporting multiple formats
    and standardization to 16kHz 16-bit PCM WAV for STT processing.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.target_sample_rate = 16000  # Standard for most STT models
        self.target_channels = 1  # Mono
        self.target_format = AudioFormat.WAV
        self.chunk_duration_seconds = 30  # 30 second chunks for long files
        self.overlap_seconds = 2  # 2 second overlap between chunks
        
        # Ensure ffmpeg is available for pydub
        if not which("ffmpeg"):
            logger.warning("ffmpeg not found - some audio formats may not be supported")
        
        logger.info("Audio Preprocessor initialized",
                   target_sample_rate=self.target_sample_rate,
                   target_channels=self.target_channels,
                   chunk_duration=self.chunk_duration_seconds)
    
    @log_performance(logger)
    async def validate_audio_file(
        self,
        audio_data: bytes,
        filename: str,
        max_duration_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate audio file format, size, and basic properties"""
        try:
            # Detect format from filename extension
            file_extension = Path(filename).suffix.lower().lstrip('.')
            if file_extension not in AudioFormat.get_supported_formats():
                return {
                    "valid": False,
                    "error": f"Unsupported format: {file_extension}",
                    "supported_formats": AudioFormat.get_supported_formats()
                }
            
            # Load and analyze audio in thread pool
            audio_info = await asyncio.get_event_loop().run_in_executor(
                None, self._analyze_audio_sync, audio_data, file_extension
            )
            
            if audio_info.get("error"):
                return {
                    "valid": False,
                    "error": audio_info["error"],
                    "filename": filename
                }
            
            # Check duration limit
            duration = audio_info["duration_seconds"]
            if max_duration_seconds and duration > max_duration_seconds:
                return {
                    "valid": False,
                    "error": f"Duration {duration:.1f}s exceeds limit {max_duration_seconds}s",
                    "duration_seconds": duration
                }
            
            # Check file size (basic sanity check)
            file_size_mb = len(audio_data) / (1024 * 1024)
            max_size_mb = self.settings.upload.max_upload_size_mb
            if file_size_mb > max_size_mb:
                return {
                    "valid": False,
                    "error": f"File size {file_size_mb:.1f}MB exceeds limit {max_size_mb}MB",
                    "file_size_mb": file_size_mb
                }
            
            logger.info("Audio file validation successful",
                       filename=filename,
                       format=file_extension,
                       duration_seconds=duration,
                       sample_rate=audio_info["sample_rate"],
                       channels=audio_info["channels"],
                       file_size_mb=file_size_mb)
            
            return {
                "valid": True,
                "metadata": {
                    "filename": filename,
                    "format": file_extension,
                    "duration_seconds": duration,
                    "sample_rate": audio_info["sample_rate"],
                    "channels": audio_info["channels"],
                    "file_size_bytes": len(audio_data),
                    "file_size_mb": file_size_mb,
                    "estimated_chunks": max(1, int(duration / self.chunk_duration_seconds))
                }
            }
            
        except Exception as e:
            logger.error("Audio validation failed",
                        filename=filename,
                        error_type=type(e).__name__,
                        error_message=str(e))
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "filename": filename
            }
    
    def _analyze_audio_sync(self, audio_data: bytes, file_extension: str) -> Dict[str, Any]:
        """Synchronous audio analysis"""
        try:
            with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                try:
                    # Load with pydub for format support
                    audio = AudioSegment.from_file(tmp_file.name)
                    
                    return {
                        "duration_seconds": len(audio) / 1000.0,
                        "sample_rate": audio.frame_rate,
                        "channels": audio.channels,
                        "format": file_extension
                    }
                    
                finally:
                    os.unlink(tmp_file.name)
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for audio preprocessing service"""
        try:
            # Test with a minimal audio file
            test_audio = AudioSegment.silent(duration=1000, frame_rate=44100)  # 1 second of silence
            test_buffer = io.BytesIO()
            test_audio.export(test_buffer, format="wav")
            test_data = test_buffer.getvalue()
            
            # Test basic validation
            result = await self.validate_audio_file(
                audio_data=test_data,
                filename="test.wav"
            )
            
            return {
                "status": "healthy" if result["valid"] else "unhealthy",
                "test_result": {
                    "validation_successful": result["valid"]
                },
                "capabilities": {
                    "supported_formats": AudioFormat.get_supported_formats(),
                    "ffmpeg_available": which("ffmpeg") is not None,
                    "target_sample_rate": self.target_sample_rate,
                    "max_chunk_duration": self.chunk_duration_seconds
                }
            }
            
        except Exception as e:
            logger.error("Audio preprocessor health check failed", error_message=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "capabilities": {
                    "supported_formats": AudioFormat.get_supported_formats(),
                    "ffmpeg_available": which("ffmpeg") is not None
                }
            }

# Global audio preprocessor instance
audio_preprocessor = AudioPreprocessor()