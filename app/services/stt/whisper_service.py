import torch
import whisper
from typing import Dict, List, Optional, Callable, Any
import asyncio
import numpy as np
from pathlib import Path
import hashlib
import time
from app.utils.logging import get_logger, log_performance
from app.config.settings import get_settings

logger = get_logger("whisper_service")

class WhisperSTTService:
    """
    Whisper Speech-to-Text service with GPU optimization and batch processing.
    Implements local hosting strategy with fallback capabilities.
    """
    
    def __init__(self, model_size: str = "large-v2", device: Optional[str] = None):
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_hash = None
        self.settings = get_settings()
        self._model_lock = asyncio.Lock()
        
        logger.info("Initializing Whisper STT Service", 
                   model_size=model_size, 
                   device=self.device,
                   cuda_available=torch.cuda.is_available())
    
    async def initialize_model(self) -> bool:
        """Initialize and load the Whisper model"""
        async with self._model_lock:
            try:
                if self.model is not None:
                    logger.info("Model already loaded", model_size=self.model_size)
                    return True
                
                logger.info("Loading Whisper model", model_size=self.model_size, device=self.device)
                start_time = time.time()
                
                # Load model in thread pool to avoid blocking
                self.model = await asyncio.get_event_loop().run_in_executor(
                    None, whisper.load_model, self.model_size, self.device
                )
                
                load_time = time.time() - start_time
                
                # Calculate model hash for version tracking
                model_state = str(self.model.state_dict()).encode()
                self.model_hash = hashlib.md5(model_state).hexdigest()[:8]
                
                logger.info("Whisper model loaded successfully",
                           model_size=self.model_size,
                           device=self.device,
                           load_time_seconds=load_time,
                           model_hash=self.model_hash)
                
                return True
                
            except Exception as e:
                logger.error("Failed to load Whisper model",
                           model_size=self.model_size,
                           device=self.device,
                           error_type=type(e).__name__,
                           error_message=str(e))
                raise
    
    @log_performance(logger)
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        task: str = "transcribe",
        progress_callback: Optional[Callable[[float], None]] = None,
        word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio data using Whisper model
        
        Args:
            audio_data: Raw audio bytes
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate"
            progress_callback: Callback for progress updates
            word_timestamps: Include word-level timestamps
            
        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            await self.initialize_model()
        
        try:
            # Convert bytes to numpy array in thread pool
            audio_array = await self._bytes_to_audio_array(audio_data)
            
            if progress_callback:
                progress_callback(10.0)  # Audio loaded
            
            # Prepare transcription options
            options = {
                "task": task,
                "language": language,
                "word_timestamps": word_timestamps,
                "verbose": False
            }
            
            if progress_callback:
                progress_callback(20.0)  # Starting transcription
            
            # Run transcription in thread pool
            start_time = time.time()
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._transcribe_with_options, audio_array, options
            )
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(90.0)  # Transcription complete
            
            # Process and format results
            formatted_result = self._format_transcription_result(result, processing_time)
            
            if progress_callback:
                progress_callback(100.0)  # Complete
            
            logger.info("Audio transcription completed",
                       text_length=len(formatted_result["text"]),
                       word_count=len(formatted_result["text"].split()),
                       processing_time_seconds=processing_time,
                       confidence_score=formatted_result["confidence_score"],
                       model_used=f"whisper-{self.model_size}")
            
            return formatted_result
            
        except Exception as e:
            logger.error("Audio transcription failed",
                        model_size=self.model_size,
                        error_type=type(e).__name__,
                        error_message=str(e))
            raise
    
    async def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array suitable for Whisper"""
        def _convert():
            # Use whisper's built-in audio loading
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                try:
                    # Load and preprocess audio
                    audio = whisper.load_audio(tmp_file.name)
                    # Ensure audio is padded or trimmed to 30 seconds max per chunk
                    audio = whisper.pad_or_trim(audio)
                    return audio
                finally:
                    os.unlink(tmp_file.name)
        
        return await asyncio.get_event_loop().run_in_executor(None, _convert)
    
    def _transcribe_with_options(self, audio_array: np.ndarray, options: Dict) -> Dict:
        """Synchronous transcription call"""
        return self.model.transcribe(audio_array, **options)
    
    def _format_transcription_result(self, raw_result: Dict, processing_time: float) -> Dict[str, Any]:
        """Format Whisper output into standardized result format"""
        # Extract word-level timestamps if available
        word_timestamps = []
        if "segments" in raw_result:
            for segment in raw_result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word_timestamps.append({
                            "word": word_info.get("word", "").strip(),
                            "start_time": word_info.get("start", 0.0),
                            "end_time": word_info.get("end", 0.0),
                            "confidence": word_info.get("probability", 0.0)
                        })
        
        # Calculate overall confidence score
        overall_confidence = 0.0
        if word_timestamps:
            total_confidence = sum(w["confidence"] for w in word_timestamps)
            overall_confidence = total_confidence / len(word_timestamps)
        
        return {
            "text": raw_result.get("text", "").strip(),
            "language": raw_result.get("language", "unknown"),
            "confidence_score": overall_confidence,
            "word_timestamps": word_timestamps,
            "segments": raw_result.get("segments", []),
            "model_used": f"whisper-{self.model_size}",
            "model_hash": self.model_hash,
            "processing_time_seconds": processing_time,
            "device_used": self.device
        }
    
    async def batch_transcribe(
        self,
        audio_batch: List[bytes],
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch transcribe multiple audio files for efficiency
        
        Args:
            audio_batch: List of audio data bytes
            language: Language code for all files
            progress_callback: Callback with (completed, total) progress
            
        Returns:
            List of transcription results
        """
        if self.model is None:
            await self.initialize_model()
        
        results = []
        total_files = len(audio_batch)
        
        logger.info("Starting batch transcription",
                   batch_size=total_files,
                   model_size=self.model_size)
        
        for i, audio_data in enumerate(audio_batch):
            try:
                result = await self.transcribe_audio(
                    audio_data=audio_data,
                    language=language,
                    word_timestamps=True
                )
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total_files)
                    
            except Exception as e:
                logger.error("Batch transcription item failed",
                           batch_index=i,
                           total_batch_size=total_files,
                           error_message=str(e))
                # Add error result to maintain batch consistency
                results.append({
                    "text": "",
                    "error": str(e),
                    "confidence_score": 0.0,
                    "word_timestamps": [],
                    "model_used": f"whisper-{self.model_size}",
                    "processing_time_seconds": 0.0
                })
        
        logger.info("Batch transcription completed",
                   total_files=total_files,
                   successful_files=len([r for r in results if "error" not in r]),
                   failed_files=len([r for r in results if "error" in r]))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": "whisper",
            "model_size": self.model_size,
            "model_hash": self.model_hash,
            "device": self.device,
            "loaded": self.model is not None,
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_mb": torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 if torch.cuda.is_available() else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the STT service"""
        try:
            if self.model is None:
                return {
                    "status": "unhealthy",
                    "reason": "Model not loaded",
                    "model_info": self.get_model_info()
                }
            
            # Test with minimal audio
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            test_audio_bytes = test_audio.tobytes()
            
            start_time = time.time()
            result = await self.transcribe_audio(test_audio_bytes, word_timestamps=False)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_seconds": response_time,
                "model_info": self.get_model_info(),
                "test_result": {
                    "text_length": len(result.get("text", "")),
                    "confidence": result.get("confidence_score", 0.0)
                }
            }
            
        except Exception as e:
            logger.error("STT health check failed", error_message=str(e))
            return {
                "status": "unhealthy",
                "reason": str(e),
                "model_info": self.get_model_info()
            }

# Global STT service instance
stt_service = WhisperSTTService()