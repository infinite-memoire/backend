import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from typing import Dict, List, Optional, Callable, Any
import asyncio
import numpy as np
import librosa
import time
from app.utils.logging_utils import get_logger, log_performance
from app.config.settings_config import get_settings

logger = get_logger("wav2vec2_service")

class Wav2Vec2STTService:
    """
    Wav2Vec2 Speech-to-Text service as fallback option.
    Optimized for speed and resource efficiency when Whisper fails.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-large-960h-lv60-self", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.settings = get_settings()
        self._model_lock = asyncio.Lock()
        
        logger.info("Initializing Wav2Vec2 STT Service", 
                   model_name=model_name, 
                   device=self.device,
                   cuda_available=torch.cuda.is_available())
    
    async def initialize_model(self) -> bool:
        """Initialize and load the Wav2Vec2 model"""
        async with self._model_lock:
            try:
                if self.model is not None:
                    logger.info("Wav2Vec2 model already loaded", model_name=self.model_name)
                    return True
                
                logger.info("Loading Wav2Vec2 model", model_name=self.model_name, device=self.device)
                start_time = time.time()
                
                # Load components in thread pool
                def _load_model():
                    processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)
                    model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
                    model.to(self.device)
                    model.eval()
                    return processor, tokenizer, model
                
                self.processor, self.tokenizer, self.model = await asyncio.get_event_loop().run_in_executor(
                    None, _load_model
                )
                
                load_time = time.time() - start_time
                
                logger.info("Wav2Vec2 model loaded successfully",
                           model_name=self.model_name,
                           device=self.device,
                           load_time_seconds=load_time)
                
                return True
                
            except Exception as e:
                logger.error("Failed to load Wav2Vec2 model",
                           model_name=self.model_name,
                           device=self.device,
                           error_type=type(e).__name__,
                           error_message=str(e))
                raise
    
    @log_performance(logger)
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        word_timestamps: bool = False  # Wav2Vec2 has limited timestamp support
    ) -> Dict[str, Any]:
        """
        Transcribe audio data using Wav2Vec2 model
        
        Args:
            audio_data: Raw audio bytes
            language: Language code (ignored for this model)
            progress_callback: Callback for progress updates
            word_timestamps: Include word-level timestamps (limited support)
            
        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            await self.initialize_model()
        
        try:
            # Convert bytes to audio array
            audio_array = await self._bytes_to_audio_array(audio_data)
            
            if progress_callback:
                progress_callback(20.0)  # Audio loaded and preprocessed
            
            # Run inference in thread pool
            start_time = time.time()
            transcription_text = await asyncio.get_event_loop().run_in_executor(
                None, self._run_inference, audio_array
            )
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(90.0)  # Transcription complete
            
            # Format results
            result = self._format_transcription_result(transcription_text, processing_time)
            
            if progress_callback:
                progress_callback(100.0)  # Complete
            
            logger.info("Wav2Vec2 transcription completed",
                       text_length=len(result["text"]),
                       word_count=len(result["text"].split()),
                       processing_time_seconds=processing_time,
                       model_used=f"wav2vec2-fallback")
            
            return result
            
        except Exception as e:
            logger.error("Wav2Vec2 transcription failed",
                        model_name=self.model_name,
                        error_type=type(e).__name__,
                        error_message=str(e))
            raise
    
    async def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array suitable for Wav2Vec2"""
        def _convert():
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                try:
                    # Load audio using librosa (16kHz for Wav2Vec2)
                    audio, sr = librosa.load(tmp_file.name, sr=16000, mono=True)
                    return audio
                finally:
                    os.unlink(tmp_file.name)
        
        return await asyncio.get_event_loop().run_in_executor(None, _convert)
    
    def _run_inference(self, audio_array: np.ndarray) -> str:
        """Synchronous inference call"""
        # Preprocess audio
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription.lower().strip()
    
    def _format_transcription_result(self, text: str, processing_time: float) -> Dict[str, Any]:
        """Format Wav2Vec2 output into standardized result format"""
        # Wav2Vec2 doesn't provide detailed timestamps or confidence scores
        # We'll estimate basic confidence based on text characteristics
        confidence_score = min(0.95, max(0.5, len(text.split()) / 10.0))  # Basic heuristic
        
        return {
            "text": text,
            "language": "en",  # Most Wav2Vec2 models are English-only
            "confidence_score": confidence_score,
            "word_timestamps": [],  # Limited support in base model
            "segments": [{"text": text, "start": 0.0, "end": processing_time}],
            "model_used": "wav2vec2-fallback",
            "model_name": self.model_name,
            "processing_time_seconds": processing_time,
            "device_used": self.device,
            "note": "Fallback model - limited timestamp support"
        }
    
    async def batch_transcribe(
        self,
        audio_batch: List[bytes],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """Batch transcribe multiple audio files"""
        if self.model is None:
            await self.initialize_model()
        
        results = []
        total_files = len(audio_batch)
        
        logger.info("Starting Wav2Vec2 batch transcription",
                   batch_size=total_files,
                   model_name=self.model_name)
        
        for i, audio_data in enumerate(audio_batch):
            try:
                result = await self.transcribe_audio(audio_data=audio_data)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total_files)
                    
            except Exception as e:
                logger.error("Wav2Vec2 batch transcription item failed",
                           batch_index=i,
                           total_batch_size=total_files,
                           error_message=str(e))
                results.append({
                    "text": "",
                    "error": str(e),
                    "confidence_score": 0.0,
                    "word_timestamps": [],
                    "model_used": "wav2vec2-fallback",
                    "processing_time_seconds": 0.0
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": "wav2vec2",
            "model_full_name": self.model_name,
            "device": self.device,
            "loaded": self.model is not None,
            "cuda_available": torch.cuda.is_available(),
            "purpose": "fallback_stt_service"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the fallback STT service"""
        try:
            if self.model is None:
                return {
                    "status": "unhealthy",
                    "reason": "Fallback model not loaded",
                    "model_info": self.get_model_info()
                }
            
            # Test with minimal audio
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            test_audio_bytes = test_audio.tobytes()
            
            start_time = time.time()
            result = await self.transcribe_audio(test_audio_bytes)
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
            logger.error("Wav2Vec2 health check failed", error_message=str(e))
            return {
                "status": "unhealthy",
                "reason": str(e),
                "model_info": self.get_model_info()
            }

# Global fallback STT service instance
fallback_stt_service = Wav2Vec2STTService()