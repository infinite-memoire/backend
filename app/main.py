from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import get_settings
from app.config.app_logging import setup_logging
from app.config.validation import validate_configuration
from app.utils.logging import get_logger
from app.middleware.logging import LoggingMiddleware

# Setup logging first
setup_logging()
logger = get_logger("main")

# Validate configuration
validate_configuration()

# Get settings
settings = get_settings()

def create_app() -> FastAPI:
    """Create FastAPI application with proper configuration"""
    
    logger.info("Starting application initialization",
               app_name=settings.app.app_name,
               version=settings.app.app_version,
               environment=settings.app.environment)
    
    app = FastAPI(
        title=settings.app.app_name,
        version=settings.app.app_version,
        debug=settings.app.debug,
        docs_url=settings.app.docs_url,
        redoc_url=settings.app.redoc_url
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=settings.app.cors_allow_credentials,
        allow_methods=settings.app.cors_allow_methods,
        allow_headers=settings.app.cors_allow_headers,
    )
    
    # Add request logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Include API routes
    from app.api.routes import upload, health, ai_processing
    app.include_router(upload.router, prefix=f"{settings.app.api_prefix}/upload")
    app.include_router(ai_processing.router, prefix=f"{settings.app.api_prefix}/ai")
    app.include_router(health.router, prefix=f"{settings.app.api_prefix}")
    
    logger.info("Application initialization completed",
               routes_count=len(app.routes),
               middleware_count=len(app.user_middleware))
    
    return app

app = create_app()

@app.on_event("startup")
async def startup_event():
    """Application startup event handler"""
    logger.info("Application startup initiated")
    
    # Initialize database connections
    from app.services.firestore import firestore_service
    from app.services.neo4j import neo4j_service
    
    # Test database connections
    await firestore_service.test_connection()
    await neo4j_service.test_connection()
    
    # Initialize AI processing system
    from app.services.orchestrator import orchestrator
    try:
        health = await orchestrator.health_check()
        logger.info("AI processing system initialized", status=health["orchestrator"])
    except Exception as e:
        logger.error("AI processing system initialization failed", error=str(e))
    
    logger.info("Application startup completed successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler"""
    logger.info("Application shutdown initiated")
    
    # Close database connections
    from app.services.neo4j import neo4j_service
    neo4j_service.close()
    
    # Close AI processing system
    from app.services.orchestrator import orchestrator
    try:
        orchestrator.close()
        logger.info("AI processing system shutdown completed")
    except Exception as e:
        logger.error("AI processing system shutdown failed", error=str(e))
    
    logger.info("Application shutdown completed")