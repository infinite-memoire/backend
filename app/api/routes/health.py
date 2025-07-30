from fastapi import APIRouter, HTTPException
from app.utils.logging_utils import get_logger
from datetime import datetime
import sys

router = APIRouter(tags=["health"])
logger = get_logger("health")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    
    try:
        # Basic health check
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "python_version": sys.version,
            "services": {
                "api": "operational"
            }
        }
        
        # Test database connections would go here
        # For MVP, we'll keep it simple
        
        logger.info("Health check passed")
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error_message=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")