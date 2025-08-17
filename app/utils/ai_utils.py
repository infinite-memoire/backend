"""
AI Provider Utilities for Infinite Memoire
"""

from typing import Dict, Any, Optional
from app.config.settings_config import get_settings
from app.utils.logging_utils import get_logger

logger = get_logger("ai_utils")

def get_ai_provider_status() -> Dict[str, Any]:
    """
    Get status of all AI providers and current configuration
    
    Returns:
        Dictionary with provider status information
    """
    settings = get_settings()
    
    status = {
        "configured_provider": settings.ai.provider,
        "providers": {
            "anthropic": {
                "available": False,
                "model": settings.ai.anthropic_model,
                "max_tokens": settings.ai.anthropic_max_tokens,
                "api_key_configured": False
            },
            "mistral": {
                "available": False,
                "model": settings.ai.mistral_model,
                "max_tokens": settings.ai.mistral_max_tokens,
                "api_key_configured": False
            },
            "gemini": {
                "available": False,
                "model": settings.ai.gemini_model,
                "max_tokens": settings.ai.gemini_max_tokens,
                "temperature": settings.ai.gemini_temperature,
                "api_key_configured": False
            }
        },
        "template_fallback": True
    }
    
    # Check Anthropic configuration
    try:
        anthropic_key = settings.ai.anthropic_api_key
        if anthropic_key and anthropic_key != "your-anthropic-api-key-here":
            status["providers"]["anthropic"]["api_key_configured"] = True
            status["providers"]["anthropic"]["available"] = True
            logger.debug("Anthropic API key is configured")
    except Exception as e:
        logger.debug(f"Anthropic configuration check failed: {e}")
    
    # Check Mistral configuration
    try:
        mistral_key = settings.ai.mistral_api_key
        if mistral_key and mistral_key != "your-mistral-api-key-here":
            status["providers"]["mistral"]["api_key_configured"] = True
            status["providers"]["mistral"]["available"] = True
            logger.debug("Mistral API key is configured")
    except Exception as e:
        logger.debug(f"Mistral configuration check failed: {e}")
    
    # Check Gemini configuration
    try:
        gemini_key = settings.ai.gemini_api_key
        if gemini_key and gemini_key != "your-gemini-api-key-here":
            status["providers"]["gemini"]["api_key_configured"] = True
            status["providers"]["gemini"]["available"] = True
            logger.debug("Google Gemini API key is configured")
    except Exception as e:
        logger.debug(f"Gemini configuration check failed: {e}")
    
    # Determine if any AI provider is available
    any_available = any(
        provider_info["available"] 
        for provider_info in status["providers"].values()
    )
    
    status["ai_enabled"] = any_available
    status["current_provider_available"] = status["providers"].get(
        settings.ai.provider, {}
    ).get("available", False)
    
    return status

def validate_ai_configuration() -> tuple[bool, Optional[str]]:
    """
    Validate the current AI configuration
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        settings = get_settings()
        provider = settings.ai.provider.lower()
        
        if provider not in ["anthropic", "mistral", "gemini"]:
            return False, f"Invalid AI provider '{provider}'. Must be 'anthropic', 'mistral', or 'gemini'"
        
        status = get_ai_provider_status()
        
        if not status["current_provider_available"]:
            return False, f"Selected provider '{provider}' is not properly configured. Check API key."
        
        return True, None
        
    except Exception as e:
        return False, f"Configuration validation failed: {str(e)}"

def get_ai_model_info() -> Dict[str, str]:
    """
    Get information about available AI models
    
    Returns:
        Dictionary with model information for each provider
    """
    return {
        "anthropic": {
            "claude-3-sonnet-20240229": "Claude 3 Sonnet - Balanced performance and speed",
            "claude-3-opus-20240229": "Claude 3 Opus - Highest quality, slower",
            "claude-3-haiku-20240307": "Claude 3 Haiku - Fastest, lower cost"
        },
        "mistral": {
            "mistral-large-latest": "Mistral Large - Best performance",
            "mistral-medium-latest": "Mistral Medium - Balanced performance",
            "mistral-small-latest": "Mistral Small - Fastest, lower cost",
            "open-mistral-7b": "Open Mistral 7B - Open source model",
            "open-mixtral-8x7b": "Open Mixtral 8x7B - Mixture of experts"
        },
        "gemini": {
            "gemini-1.5-pro": "Gemini 1.5 Pro - Best performance, multimodal",
            "gemini-1.5-flash": "Gemini 1.5 Flash - Faster, cost-effective",
            "gemini-1.0-pro": "Gemini 1.0 Pro - Previous generation",
            "gemini-pro": "Gemini Pro - Standard model"
        }
    }

def log_ai_status():
    """Log current AI provider status for debugging"""
    status = get_ai_provider_status()
    
    logger.info("AI Provider Status Report",
               configured_provider=status["configured_provider"],
               ai_enabled=status["ai_enabled"],
               current_provider_available=status["current_provider_available"])
    
    for provider, info in status["providers"].items():
        logger.info(f"Provider {provider}",
                   available=info["available"],
                   model=info["model"],
                   api_key_configured=info["api_key_configured"])
    
    if not status["ai_enabled"]:
        logger.warning("No AI providers are available - using template fallback")
        logger.warning("Configure AI_ANTHROPIC_API_KEY, AI_MISTRAL_API_KEY, or AI_GEMINI_API_KEY and set AI_PROVIDER")

def get_recommended_settings() -> Dict[str, str]:
    """
    Get recommended environment variable settings
    
    Returns:
        Dictionary with recommended settings
    """
    return {
        "AI_PROVIDER": "anthropic",  # or mistral, gemini
        "AI_ANTHROPIC_API_KEY": "your-anthropic-api-key-here",
        "AI_ANTHROPIC_MODEL": "claude-3-sonnet-20240229", 
        "AI_ANTHROPIC_MAX_TOKENS": "2000",
        "AI_MISTRAL_API_KEY": "your-mistral-api-key-here",
        "AI_MISTRAL_MODEL": "mistral-large-latest",
        "AI_MISTRAL_MAX_TOKENS": "2000",
        "AI_GEMINI_API_KEY": "your-gemini-api-key-here",
        "AI_GEMINI_MODEL": "gemini-1.5-pro",
        "AI_GEMINI_MAX_TOKENS": "2000",
        "AI_GEMINI_TEMPERATURE": "0.7"
    }