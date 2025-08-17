from fastapi import APIRouter, HTTPException
from app.utils.logging_utils import get_logger
from app.utils.ai_utils import (
    get_ai_provider_status, 
    validate_ai_configuration, 
    get_ai_model_info,
    get_recommended_settings
)
from datetime import datetime
import sys

router = APIRouter(tags=["health"])
logger = get_logger("health")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    
    try:
        # Basic health check
        # Get AI provider status
        ai_status = get_ai_provider_status()
        ai_valid, ai_error = validate_ai_configuration()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "python_version": sys.version,
            "services": {
                "api": "operational",
                "ai_providers": {
                    "status": "available" if ai_status["ai_enabled"] else "template_only",
                    "configured_provider": ai_status["configured_provider"],
                    "current_provider_available": ai_status["current_provider_available"],
                    "anthropic_available": ai_status["providers"]["anthropic"]["available"],
                    "mistral_available": ai_status["providers"]["mistral"]["available"],
                    "validation_error": ai_error
                }
            }
        }
        
        # Test database connections would go here
        # For MVP, we'll keep it simple
        
        logger.info("Health check passed")
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error_message=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")

@router.get("/ai-status")
async def ai_status_check():
    """Detailed AI provider status and configuration information"""
    
    try:
        ai_status = get_ai_provider_status()
        ai_valid, ai_error = validate_ai_configuration()
        model_info = get_ai_model_info()
        recommended_settings = get_recommended_settings()
        
        return {
            "ai_configuration": ai_status,
            "validation": {
                "is_valid": ai_valid,
                "error": ai_error
            },
            "available_models": model_info,
            "recommended_settings": recommended_settings,
            "setup_instructions": {
                "step_1": "Choose AI provider: Set AI_PROVIDER to 'anthropic' or 'mistral'",
                "step_2": "Configure API key: Set AI_ANTHROPIC_API_KEY or AI_MISTRAL_API_KEY",
                "step_3": "Optional: Customize model with AI_ANTHROPIC_MODEL or AI_MISTRAL_MODEL",
                "step_4": "Restart the application to apply changes"
            }
        }
        
    except Exception as e:
        logger.error("AI status check failed", error_message=str(e))
        raise HTTPException(status_code=500, detail=f"AI status check failed: {str(e)}")