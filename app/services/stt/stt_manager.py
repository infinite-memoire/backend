from typing import Dict, List, Optional, Callable, Any, Union
import asyncio
from enum import Enum
from app.utils.logging_utils import get_logger, log_performance
from app.config.settings_config import get_settings

logger = get_logger("stt_manager")

class ModelType(Enum):
    WHISPER_LARGE = "whisper-large-v2"
    WHISPER_MEDIUM = "whisper-medium"
    WHISPER_BASE = "whisper-base"
    WAV2VEC2_FALLBACK = "wav2vec2-fallback"

class QualityTier(Enum):
    FAST = "fast"          # Faster, lower accuracy
    ACCURATE = "accurate"  # Slower, higher accuracy

class STTManager:
    """
    STT Manager that coordinates between different models and provides
    intelligent routing, fallback handling, and quality tier management.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.primary_service = None
        self.fallback_service = None
        self.model_info = {}
        self._initialization_lock = asyncio.Lock()
        
        # Quality tier configuration
        self.quality_tiers = {
            QualityTier.FAST: {
                "model": ModelType.WHISPER_BASE,
                "max_processing_time": 60,  # seconds
                "batch_size": 4
            },
            QualityTier.ACCURATE: {
                "model": ModelType.WHISPER_LARGE,
                "max_processing_time": 300,  # seconds
                "batch_size": 2
            }
        }
        
        logger.info("STT Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize STT services with primary and fallback models"""
        async with self._initialization_lock:
            try:
                # Import here to avoid circular imports
                from .whisper_service import stt_service
                from .wav2vec2_fallback import fallback_stt_service
                
                # Initialize primary service (Whisper)
                logger.info("Initializing primary STT service (Whisper)")
                self.primary_service = stt_service
                await self.primary_service.initialize_model()
                
                # Initialize fallback service (Wav2Vec2)
                logger.info("Initializing fallback STT service (Wav2Vec2)")
                self.fallback_service = fallback_stt_service
                await self.fallback_service.initialize_model()
                
                # Store model information
                self.model_info = {
                    "primary": self.primary_service.get_model_info(),
                    "fallback": self.fallback_service.get_model_info()
                }
                
                logger.info("STT Manager initialization completed",
                           primary_model=self.model_info["primary"]["model_name"],
                           fallback_model=self.model_info["fallback"]["model_name"])
                
                return True
                
            except Exception as e:
                logger.error("STT Manager initialization failed",
                           error_type=type(e).__name__,
                           error_message=str(e))
                raise
    
    @log_performance(logger)
    async def transcribe_audio(
        self,
        audio_data: bytes,
        quality_tier: QualityTier = QualityTier.ACCURATE,
        language: Optional[str] = None,
        user_id: Optional[str] = None,
        priority: str = "normal",
        progress_callback: Optional[Callable[[float], None]] = None,
        attempt_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio with intelligent model selection and fallback handling
        
        Args:
            audio_data: Raw audio bytes
            quality_tier: Quality vs speed preference
            language: Language code
            user_id: User identifier for logging
            priority: Processing priority (normal, high, premium)
            progress_callback: Progress callback function
            attempt_fallback: Whether to try fallback on primary failure
            
        Returns:
            Transcription result with metadata
        """
        if self.primary_service is None:
            await self.initialize()
        
        tier_config = self.quality_tiers[quality_tier]
        selected_model = tier_config["model"]
        
        logger.info("Starting audio transcription",
                   quality_tier=quality_tier.value,
                   selected_model=selected_model.value,
                   user_id=user_id,
                   priority=priority,
                   audio_size_bytes=len(audio_data))
        
        # Try primary service first
        try:
            if progress_callback:
                progress_callback(5.0)  # Starting
            
            # Choose service based on quality tier
            if selected_model in [ModelType.WHISPER_LARGE, ModelType.WHISPER_MEDIUM, ModelType.WHISPER_BASE]:
                service = self.primary_service
                # Adjust model size if needed (this would require model reloading in production)
                if selected_model == ModelType.WHISPER_BASE and quality_tier == QualityTier.FAST:
                    logger.info("Using fast processing mode")
            else:
                service = self.fallback_service
            
            result = await service.transcribe_audio(
                audio_data=audio_data,
                language=language,
                progress_callback=progress_callback,
                word_timestamps=quality_tier == QualityTier.ACCURATE
            )
            
            # Add STT Manager metadata
            result["stt_manager"] = {
                "quality_tier": quality_tier.value,
                "model_selected": selected_model.value,
                "service_used": "primary",
                "fallback_attempted": False,
                "user_id": user_id,
                "priority": priority
            }
            
            logger.info("Primary transcription successful",
                       quality_tier=quality_tier.value,
                       model_used=result.get("model_used"),
                       text_length=len(result.get("text", "")),
                       confidence=result.get("confidence_score", 0.0),
                       user_id=user_id)
            
            return result
            
        except Exception as primary_error:
            logger.warning("Primary STT service failed",
                         model=selected_model.value,
                         user_id=user_id,
                         error_message=str(primary_error))
            
            if not attempt_fallback:
                raise primary_error
            
            # Try fallback service
            try:
                logger.info("Attempting fallback transcription",
                           user_id=user_id,
                           fallback_model=self.model_info["fallback"]["model_name"])
                
                if progress_callback:
                    progress_callback(10.0)  # Restarting with fallback
                
                result = await self.fallback_service.transcribe_audio(
                    audio_data=audio_data,
                    language=language,
                    progress_callback=progress_callback,
                    word_timestamps=False  # Fallback has limited timestamp support
                )
                
                # Add fallback metadata
                result["stt_manager"] = {
                    "quality_tier": quality_tier.value,
                    "model_selected": selected_model.value,
                    "service_used": "fallback",
                    "fallback_attempted": True,
                    "primary_error": str(primary_error),
                    "user_id": user_id,
                    "priority": priority
                }
                
                logger.info("Fallback transcription successful",
                           fallback_model=result.get("model_used"),
                           text_length=len(result.get("text", "")),
                           user_id=user_id)
                
                return result
                
            except Exception as fallback_error:
                logger.error("Both primary and fallback STT services failed",
                           primary_error=str(primary_error),
                           fallback_error=str(fallback_error),
                           user_id=user_id)
                raise Exception(f"STT processing failed. Primary: {primary_error}, Fallback: {fallback_error}")
    
    async def batch_transcribe(
        self,
        audio_batch: List[bytes],
        quality_tier: QualityTier = QualityTier.ACCURATE,
        language: Optional[str] = None,
        user_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """Batch transcribe multiple audio files"""
        if self.primary_service is None:
            await self.initialize()
        
        tier_config = self.quality_tiers[quality_tier]
        batch_size = tier_config["batch_size"]
        
        logger.info("Starting batch transcription",
                   total_files=len(audio_batch),
                   quality_tier=quality_tier.value,
                   batch_size=batch_size,
                   user_id=user_id)
        
        results = []
        total_files = len(audio_batch)
        
        # Process in batches
        for i in range(0, total_files, batch_size):
            batch_slice = audio_batch[i:i + batch_size]
            
            try:
                batch_results = await self.primary_service.batch_transcribe(
                    audio_batch=batch_slice,
                    language=language,
                    progress_callback=None  # Handle progress at manager level
                )
                
                # Add manager metadata to each result
                for result in batch_results:
                    if "error" not in result:
                        result["stt_manager"] = {
                            "quality_tier": quality_tier.value,
                            "service_used": "primary",
                            "batch_processing": True,
                            "user_id": user_id
                        }
                
                results.extend(batch_results)
                
                if progress_callback:
                    progress_callback(min(i + batch_size, total_files), total_files)
                
            except Exception as e:
                logger.error("Batch processing failed",
                           batch_start=i,
                           batch_size=len(batch_slice),
                           error_message=str(e),
                           user_id=user_id)
                
                # Add error results for failed batch
                for _ in batch_slice:
                    results.append({
                        "text": "",
                        "error": str(e),
                        "confidence_score": 0.0,
                        "stt_manager": {
                            "quality_tier": quality_tier.value,
                            "service_used": "failed",
                            "batch_processing": True,
                            "user_id": user_id
                        }
                    })
        
        successful_count = len([r for r in results if "error" not in r])
        logger.info("Batch transcription completed",
                   total_files=total_files,
                   successful_files=successful_count,
                   failed_files=total_files - successful_count,
                   user_id=user_id)
        
        return results
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get information about available STT models"""
        if self.primary_service is None:
            await self.initialize()
        
        return {
            "models": {
                "primary": self.model_info.get("primary", {}),
                "fallback": self.model_info.get("fallback", {})
            },
            "quality_tiers": {
                tier.value: {
                    "model": config["model"].value,
                    "max_processing_time": config["max_processing_time"],
                    "batch_size": config["batch_size"]
                }
                for tier, config in self.quality_tiers.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all STT services"""
        if self.primary_service is None:
            try:
                await self.initialize()
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "reason": f"Initialization failed: {str(e)}",
                    "services": {}
                }
        
        # Check primary service
        primary_health = await self.primary_service.health_check()
        
        # Check fallback service
        fallback_health = await self.fallback_service.health_check()
        
        # Overall status
        overall_status = "healthy"
        if primary_health["status"] != "healthy" and fallback_health["status"] != "healthy":
            overall_status = "unhealthy"
        elif primary_health["status"] != "healthy":
            overall_status = "degraded"  # Fallback available
        
        return {
            "status": overall_status,
            "services": {
                "primary": primary_health,
                "fallback": fallback_health
            },
            "quality_tiers": list(self.quality_tiers.keys()),
            "manager_info": {
                "initialized": self.primary_service is not None,
                "available_models": len(self.model_info)
            }
        }

# Global STT manager instance
stt_manager = STTManager()