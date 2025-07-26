import io
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import tempfile
import os
import asyncio
from pydub import AudioSegment
from pydub.utils import which
import librosa
import torchaudio
from app.utils.logging import get_logger, log_performance
from app.config.settings import get_settings

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
        """
        Validate audio file format, size, and basic properties
        
        Args:
            audio_data: Raw audio file bytes
            filename: Original filename for format detection
            max_duration_seconds: Maximum allowed duration
            
        Returns:
            Validation result with metadata
        """
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
            return {"error": str(e)}\n    \n    @log_performance(logger)\n    async def standardize_audio(\n        self,\n        audio_data: bytes,\n        source_format: str,\n        normalize: bool = True\n    ) -> Tuple[bytes, Dict[str, Any]]:\n        \"\"\"\n        Convert audio to standardized format (16kHz, 16-bit, mono WAV)\n        \n        Args:\n            audio_data: Raw audio file bytes\n            source_format: Source audio format\n            normalize: Whether to normalize audio levels\n            \n        Returns:\n            Tuple of (standardized_audio_bytes, metadata)\n        \"\"\"\n        try:\n            # Process in thread pool to avoid blocking\n            result = await asyncio.get_event_loop().run_in_executor(\n                None, self._standardize_audio_sync, audio_data, source_format, normalize\n            )\n            \n            if \"error\" in result:\n                raise Exception(result[\"error\"])\n            \n            logger.info(\"Audio standardization completed\",\n                       source_format=source_format,\n                       target_sample_rate=self.target_sample_rate,\n                       target_channels=self.target_channels,\n                       normalized=normalize,\n                       output_size_bytes=len(result[\"audio_data\"]))\n            \n            return result[\"audio_data\"], result[\"metadata\"]\n            \n        except Exception as e:\n            logger.error(\"Audio standardization failed\",\n                        source_format=source_format,\n                        error_type=type(e).__name__,\n                        error_message=str(e))\n            raise\n    \n    def _standardize_audio_sync(\n        self, \n        audio_data: bytes, \n        source_format: str, \n        normalize: bool\n    ) -> Dict[str, Any]:\n        \"\"\"Synchronous audio standardization\"\"\"\n        try:\n            with tempfile.NamedTemporaryFile(suffix=f'.{source_format}', delete=False) as input_file:\n                input_file.write(audio_data)\n                input_file.flush()\n                \n                try:\n                    # Load audio with pydub\n                    audio = AudioSegment.from_file(input_file.name)\n                    \n                    # Store original metadata\n                    original_metadata = {\n                        \"original_sample_rate\": audio.frame_rate,\n                        \"original_channels\": audio.channels,\n                        \"original_duration_seconds\": len(audio) / 1000.0,\n                        \"original_format\": source_format\n                    }\n                    \n                    # Convert to mono\n                    if audio.channels > 1:\n                        audio = audio.set_channels(1)\n                    \n                    # Resample to target sample rate\n                    if audio.frame_rate != self.target_sample_rate:\n                        audio = audio.set_frame_rate(self.target_sample_rate)\n                    \n                    # Normalize volume if requested\n                    if normalize:\n                        # Normalize to -3dB to avoid clipping\n                        audio = audio.normalize(headroom=3.0)\n                    \n                    # Convert to 16-bit\n                    audio = audio.set_sample_width(2)  # 2 bytes = 16 bits\n                    \n                    # Export to WAV bytes\n                    output_buffer = io.BytesIO()\n                    audio.export(output_buffer, format=\"wav\")\n                    standardized_data = output_buffer.getvalue()\n                    \n                    # Final metadata\n                    final_metadata = {\n                        **original_metadata,\n                        \"final_sample_rate\": self.target_sample_rate,\n                        \"final_channels\": 1,\n                        \"final_duration_seconds\": len(audio) / 1000.0,\n                        \"final_format\": \"wav\",\n                        \"normalized\": normalize,\n                        \"size_reduction_ratio\": len(standardized_data) / len(audio_data)\n                    }\n                    \n                    return {\n                        \"audio_data\": standardized_data,\n                        \"metadata\": final_metadata\n                    }\n                    \n                finally:\n                    os.unlink(input_file.name)\n                    \n        except Exception as e:\n            return {\"error\": str(e)}\n    \n    @log_performance(logger)\n    async def chunk_long_audio(\n        self,\n        audio_data: bytes,\n        max_chunk_duration: Optional[int] = None,\n        overlap_duration: Optional[int] = None\n    ) -> List[Dict[str, Any]]:\n        \"\"\"\n        Split long audio files into manageable chunks for processing\n        \n        Args:\n            audio_data: Standardized audio data (WAV format)\n            max_chunk_duration: Maximum chunk duration in seconds\n            overlap_duration: Overlap between chunks in seconds\n            \n        Returns:\n            List of chunk dictionaries with audio data and metadata\n        \"\"\"\n        chunk_duration = max_chunk_duration or self.chunk_duration_seconds\n        overlap_duration = overlap_duration or self.overlap_seconds\n        \n        try:\n            # Process in thread pool\n            chunks = await asyncio.get_event_loop().run_in_executor(\n                None, self._chunk_audio_sync, audio_data, chunk_duration, overlap_duration\n            )\n            \n            if \"error\" in chunks:\n                raise Exception(chunks[\"error\"])\n            \n            chunk_list = chunks[\"chunks\"]\n            \n            logger.info(\"Audio chunking completed\",\n                       total_chunks=len(chunk_list),\n                       chunk_duration_seconds=chunk_duration,\n                       overlap_seconds=overlap_duration,\n                       original_size_bytes=len(audio_data))\n            \n            return chunk_list\n            \n        except Exception as e:\n            logger.error(\"Audio chunking failed\",\n                        error_type=type(e).__name__,\n                        error_message=str(e))\n            raise\n    \n    def _chunk_audio_sync(\n        self, \n        audio_data: bytes, \n        chunk_duration: int, \n        overlap_duration: int\n    ) -> Dict[str, Any]:\n        \"\"\"Synchronous audio chunking\"\"\"\n        try:\n            # Load WAV data\n            audio = AudioSegment.from_wav(io.BytesIO(audio_data))\n            total_duration_ms = len(audio)\n            chunk_duration_ms = chunk_duration * 1000\n            overlap_ms = overlap_duration * 1000\n            \n            # If audio is shorter than chunk duration, return as single chunk\n            if total_duration_ms <= chunk_duration_ms:\n                return {\n                    \"chunks\": [{\n                        \"chunk_index\": 0,\n                        \"audio_data\": audio_data,\n                        \"start_time_seconds\": 0.0,\n                        \"end_time_seconds\": total_duration_ms / 1000.0,\n                        \"duration_seconds\": total_duration_ms / 1000.0,\n                        \"size_bytes\": len(audio_data),\n                        \"is_final_chunk\": True\n                    }]\n                }\n            \n            chunks = []\n            chunk_index = 0\n            start_ms = 0\n            \n            while start_ms < total_duration_ms:\n                # Calculate end time for this chunk\n                end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)\n                \n                # Extract chunk\n                chunk_audio = audio[start_ms:end_ms]\n                \n                # Export chunk to bytes\n                chunk_buffer = io.BytesIO()\n                chunk_audio.export(chunk_buffer, format=\"wav\")\n                chunk_data = chunk_buffer.getvalue()\n                \n                # Create chunk metadata\n                chunk_info = {\n                    \"chunk_index\": chunk_index,\n                    \"audio_data\": chunk_data,\n                    \"start_time_seconds\": start_ms / 1000.0,\n                    \"end_time_seconds\": end_ms / 1000.0,\n                    \"duration_seconds\": (end_ms - start_ms) / 1000.0,\n                    \"size_bytes\": len(chunk_data),\n                    \"is_final_chunk\": end_ms >= total_duration_ms\n                }\n                \n                chunks.append(chunk_info)\n                \n                # Move to next chunk with overlap\n                start_ms = end_ms - overlap_ms\n                chunk_index += 1\n                \n                # Prevent infinite loop\n                if start_ms >= end_ms:\n                    break\n            \n            return {\"chunks\": chunks}\n            \n        except Exception as e:\n            return {\"error\": str(e)}\n    \n    @log_performance(logger)\n    async def process_audio_file(\n        self,\n        audio_data: bytes,\n        filename: str,\n        max_duration_seconds: Optional[int] = None,\n        normalize: bool = True,\n        chunk_long_files: bool = True\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Complete audio preprocessing pipeline\n        \n        Args:\n            audio_data: Raw audio file bytes\n            filename: Original filename\n            max_duration_seconds: Maximum allowed duration\n            normalize: Whether to normalize audio levels\n            chunk_long_files: Whether to chunk long files\n            \n        Returns:\n            Complete processing result with chunks and metadata\n        \"\"\"\n        # Step 1: Validate audio file\n        validation_result = await self.validate_audio_file(\n            audio_data, filename, max_duration_seconds\n        )\n        \n        if not validation_result[\"valid\"]:\n            return {\n                \"success\": False,\n                \"error\": validation_result[\"error\"],\n                \"validation_result\": validation_result\n            }\n        \n        original_metadata = validation_result[\"metadata\"]\n        \n        # Step 2: Standardize audio format\n        try:\n            standardized_audio, standardization_metadata = await self.standardize_audio(\n                audio_data=audio_data,\n                source_format=original_metadata[\"format\"],\n                normalize=normalize\n            )\n        except Exception as e:\n            return {\n                \"success\": False,\n                \"error\": f\"Standardization failed: {str(e)}\",\n                \"original_metadata\": original_metadata\n            }\n        \n        # Step 3: Chunk long files if needed\n        chunks = []\n        if chunk_long_files and original_metadata[\"duration_seconds\"] > self.chunk_duration_seconds:\n            try:\n                chunks = await self.chunk_long_audio(standardized_audio)\n            except Exception as e:\n                return {\n                    \"success\": False,\n                    \"error\": f\"Chunking failed: {str(e)}\",\n                    \"original_metadata\": original_metadata,\n                    \"standardization_metadata\": standardization_metadata\n                }\n        else:\n            # Single chunk for short files\n            chunks = [{\n                \"chunk_index\": 0,\n                \"audio_data\": standardized_audio,\n                \"start_time_seconds\": 0.0,\n                \"end_time_seconds\": original_metadata[\"duration_seconds\"],\n                \"duration_seconds\": original_metadata[\"duration_seconds\"],\n                \"size_bytes\": len(standardized_audio),\n                \"is_final_chunk\": True\n            }]\n        \n        logger.info(\"Audio preprocessing pipeline completed\",\n                   filename=filename,\n                   original_duration=original_metadata[\"duration_seconds\"],\n                   total_chunks=len(chunks),\n                   standardized_size_bytes=len(standardized_audio))\n        \n        return {\n            \"success\": True,\n            \"original_metadata\": original_metadata,\n            \"standardization_metadata\": standardization_metadata,\n            \"chunks\": chunks,\n            \"total_chunks\": len(chunks),\n            \"processing_summary\": {\n                \"input_format\": original_metadata[\"format\"],\n                \"output_format\": \"wav\",\n                \"sample_rate_conversion\": f\"{standardization_metadata['original_sample_rate']} -> {standardization_metadata['final_sample_rate']}\",\n                \"channels_conversion\": f\"{standardization_metadata['original_channels']} -> {standardization_metadata['final_channels']}\",\n                \"normalized\": normalize,\n                \"chunked\": len(chunks) > 1\n            }\n        }\n    \n    async def health_check(self) -> Dict[str, Any]:\n        \"\"\"Health check for audio preprocessing service\"\"\"\n        try:\n            # Test with a minimal audio file\n            test_audio = AudioSegment.silent(duration=1000, frame_rate=44100)  # 1 second of silence\n            test_buffer = io.BytesIO()\n            test_audio.export(test_buffer, format=\"wav\")\n            test_data = test_buffer.getvalue()\n            \n            # Test the preprocessing pipeline\n            result = await self.process_audio_file(\n                audio_data=test_data,\n                filename=\"test.wav\",\n                normalize=False,\n                chunk_long_files=False\n            )\n            \n            return {\n                \"status\": \"healthy\" if result[\"success\"] else \"unhealthy\",\n                \"test_result\": {\n                    \"preprocessing_successful\": result[\"success\"],\n                    \"chunks_created\": result.get(\"total_chunks\", 0)\n                },\n                \"capabilities\": {\n                    \"supported_formats\": AudioFormat.get_supported_formats(),\n                    \"ffmpeg_available\": which(\"ffmpeg\") is not None,\n                    \"target_sample_rate\": self.target_sample_rate,\n                    \"max_chunk_duration\": self.chunk_duration_seconds\n                }\n            }\n            \n        except Exception as e:\n            logger.error(\"Audio preprocessor health check failed\", error_message=str(e))\n            return {\n                \"status\": \"unhealthy\",\n                \"error\": str(e),\n                \"capabilities\": {\n                    \"supported_formats\": AudioFormat.get_supported_formats(),\n                    \"ffmpeg_available\": which(\"ffmpeg\") is not None\n                }\n            }\n\n# Global audio preprocessor instance\naudio_preprocessor = AudioPreprocessor()"
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
    
    @log_performance(logger)
    async def standardize_audio(
        self,
        audio_data: bytes,
        source_format: str,
        normalize: bool = True
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Convert audio to standardized format (16kHz, 16-bit, mono WAV)
        
        Args:
            audio_data: Raw audio file bytes
            source_format: Source audio format
            normalize: Whether to normalize audio levels
            
        Returns:
            Tuple of (standardized_audio_bytes, metadata)
        """
        try:
            # Process in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._standardize_audio_sync, audio_data, source_format, normalize
            )
            
            if "error" in result:
                raise Exception(result["error"])
            
            logger.info("Audio standardization completed",
                       source_format=source_format,
                       target_sample_rate=self.target_sample_rate,
                       target_channels=self.target_channels,
                       normalized=normalize,
                       output_size_bytes=len(result["audio_data"]))
            
            return result["audio_data"], result["metadata"]
            
        except Exception as e:
            logger.error("Audio standardization failed",
                        source_format=source_format,
                        error_type=type(e).__name__,
                        error_message=str(e))
            raise
    
    def _standardize_audio_sync(
        self, 
        audio_data: bytes, 
        source_format: str, 
        normalize: bool
    ) -> Dict[str, Any]:
        """Synchronous audio standardization"""
        try:
            with tempfile.NamedTemporaryFile(suffix=f'.{source_format}', delete=False) as input_file:
                input_file.write(audio_data)
                input_file.flush()
                
                try:
                    # Load audio with pydub
                    audio = AudioSegment.from_file(input_file.name)
                    
                    # Store original metadata
                    original_metadata = {
                        "original_sample_rate": audio.frame_rate,
                        "original_channels": audio.channels,
                        "original_duration_seconds": len(audio) / 1000.0,
                        "original_format": source_format
                    }
                    
                    # Convert to mono
                    if audio.channels > 1:
                        audio = audio.set_channels(1)
                    
                    # Resample to target sample rate
                    if audio.frame_rate != self.target_sample_rate:
                        audio = audio.set_frame_rate(self.target_sample_rate)
                    
                    # Normalize volume if requested
                    if normalize:
                        # Normalize to -3dB to avoid clipping
                        audio = audio.normalize(headroom=3.0)
                    
                    # Convert to 16-bit
                    audio = audio.set_sample_width(2)  # 2 bytes = 16 bits
                    
                    # Export to WAV bytes
                    output_buffer = io.BytesIO()
                    audio.export(output_buffer, format="wav")
                    standardized_data = output_buffer.getvalue()
                    
                    # Final metadata
                    final_metadata = {
                        **original_metadata,
                        "final_sample_rate": self.target_sample_rate,
                        "final_channels": 1,
                        "final_duration_seconds": len(audio) / 1000.0,
                        "final_format": "wav",
                        "normalized": normalize,
                        "size_reduction_ratio": len(standardized_data) / len(audio_data)
                    }
                    
                    return {
                        "audio_data": standardized_data,
                        "metadata": final_metadata
                    }
                    
                finally:
                    os.unlink(input_file.name)
                    
        except Exception as e:
            return {"error": str(e)}
    
    @log_performance(logger)
    async def chunk_long_audio(
        self,
        audio_data: bytes,
        max_chunk_duration: Optional[int] = None,
        overlap_duration: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Split long audio files into manageable chunks for processing
        
        Args:
            audio_data: Standardized audio data (WAV format)
            max_chunk_duration: Maximum chunk duration in seconds
            overlap_duration: Overlap between chunks in seconds
            
        Returns:
            List of chunk dictionaries with audio data and metadata
        """
        chunk_duration = max_chunk_duration or self.chunk_duration_seconds
        overlap_duration = overlap_duration or self.overlap_seconds
        
        try:
            # Process in thread pool
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, self._chunk_audio_sync, audio_data, chunk_duration, overlap_duration
            )
            
            if "error" in chunks:
                raise Exception(chunks["error"])
            
            chunk_list = chunks["chunks"]
            
            logger.info("Audio chunking completed",
                       total_chunks=len(chunk_list),
                       chunk_duration_seconds=chunk_duration,
                       overlap_seconds=overlap_duration,
                       original_size_bytes=len(audio_data))
            
            return chunk_list
            
        except Exception as e:
            logger.error("Audio chunking failed",
                        error_type=type(e).__name__,
                        error_message=str(e))
            raise
    
    def _chunk_audio_sync(
        self, 
        audio_data: bytes, 
        chunk_duration: int, 
        overlap_duration: int
    ) -> Dict[str, Any]:
        """Synchronous audio chunking"""
        try:
            # Load WAV data
            audio = AudioSegment.from_wav(io.BytesIO(audio_data))
            total_duration_ms = len(audio)
            chunk_duration_ms = chunk_duration * 1000
            overlap_ms = overlap_duration * 1000
            
            # If audio is shorter than chunk duration, return as single chunk
            if total_duration_ms <= chunk_duration_ms:
                return {
                    "chunks": [{
                        "chunk_index": 0,
                        "audio_data": audio_data,
                        "start_time_seconds": 0.0,
                        "end_time_seconds": total_duration_ms / 1000.0,
                        "duration_seconds": total_duration_ms / 1000.0,
                        "size_bytes": len(audio_data),
                        "is_final_chunk": True
                    }]
                }
            
            chunks = []
            chunk_index = 0
            start_ms = 0
            
            while start_ms < total_duration_ms:
                # Calculate end time for this chunk
                end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
                
                # Extract chunk
                chunk_audio = audio[start_ms:end_ms]
                
                # Export chunk to bytes
                chunk_buffer = io.BytesIO()
                chunk_audio.export(chunk_buffer, format="wav")
                chunk_data = chunk_buffer.getvalue()
                
                # Create chunk metadata
                chunk_info = {
                    "chunk_index": chunk_index,
                    "audio_data": chunk_data,
                    "start_time_seconds": start_ms / 1000.0,
                    "end_time_seconds": end_ms / 1000.0,
                    "duration_seconds": (end_ms - start_ms) / 1000.0,
                    "size_bytes": len(chunk_data),
                    "is_final_chunk": end_ms >= total_duration_ms
                }
                
                chunks.append(chunk_info)
                
                # Move to next chunk with overlap
                start_ms = end_ms - overlap_ms
                chunk_index += 1
                
                # Prevent infinite loop
                if start_ms >= end_ms:
                    break
            
            return {"chunks": chunks}
            
        except Exception as e:
            return {"error": str(e)}
    
    @log_performance(logger)
    async def process_audio_file(
        self,
        audio_data: bytes,
        filename: str,
        max_duration_seconds: Optional[int] = None,
        normalize: bool = True,
        chunk_long_files: bool = True
    ) -> Dict[str, Any]:
        """
        Complete audio preprocessing pipeline
        
        Args:
            audio_data: Raw audio file bytes
            filename: Original filename
            max_duration_seconds: Maximum allowed duration
            normalize: Whether to normalize audio levels
            chunk_long_files: Whether to chunk long files
            
        Returns:
            Complete processing result with chunks and metadata
        """
        # Step 1: Validate audio file
        validation_result = await self.validate_audio_file(
            audio_data, filename, max_duration_seconds
        )
        
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"],
                "validation_result": validation_result
            }
        
        original_metadata = validation_result["metadata"]
        
        # Step 2: Standardize audio format
        try:
            standardized_audio, standardization_metadata = await self.standardize_audio(
                audio_data=audio_data,
                source_format=original_metadata["format"],
                normalize=normalize
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Standardization failed: {str(e)}",
                "original_metadata": original_metadata
            }
        
        # Step 3: Chunk long files if needed
        chunks = []
        if chunk_long_files and original_metadata["duration_seconds"] > self.chunk_duration_seconds:
            try:
                chunks = await self.chunk_long_audio(standardized_audio)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Chunking failed: {str(e)}",
                    "original_metadata": original_metadata,
                    "standardization_metadata": standardization_metadata
                }
        else:
            # Single chunk for short files
            chunks = [{
                "chunk_index": 0,
                "audio_data": standardized_audio,
                "start_time_seconds": 0.0,
                "end_time_seconds": original_metadata["duration_seconds"],
                "duration_seconds": original_metadata["duration_seconds"],
                "size_bytes": len(standardized_audio),
                "is_final_chunk": True
            }]
        
        logger.info("Audio preprocessing pipeline completed",
                   filename=filename,
                   original_duration=original_metadata["duration_seconds"],
                   total_chunks=len(chunks),
                   standardized_size_bytes=len(standardized_audio))
        
        return {
            "success": True,
            "original_metadata": original_metadata,
            "standardization_metadata": standardization_metadata,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "processing_summary": {
                "input_format": original_metadata["format"],
                "output_format": "wav",
                "sample_rate_conversion": f"{standardization_metadata['original_sample_rate']} -> {standardization_metadata['final_sample_rate']}",
                "channels_conversion": f"{standardization_metadata['original_channels']} -> {standardization_metadata['final_channels']}",
                "normalized": normalize,
                "chunked": len(chunks) > 1
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for audio preprocessing service"""
        try:
            # Test with a minimal audio file
            test_audio = AudioSegment.silent(duration=1000, frame_rate=44100)  # 1 second of silence
            test_buffer = io.BytesIO()
            test_audio.export(test_buffer, format="wav")
            test_data = test_buffer.getvalue()
            
            # Test the preprocessing pipeline
            result = await self.process_audio_file(
                audio_data=test_data,
                filename="test.wav",
                normalize=False,
                chunk_long_files=False
            )
            
            return {
                "status": "healthy" if result["success"] else "unhealthy",
                "test_result": {
                    "preprocessing_successful": result["success"],
                    "chunks_created": result.get("total_chunks", 0)
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