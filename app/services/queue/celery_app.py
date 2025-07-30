from celery import Celery
from celery.signals import worker_ready, worker_shutdown
from app.config.settings_config import get_settings
from app.utils.logging_utils import get_logger
import os

logger = get_logger("celery_app")

# Initialize settings
settings = get_settings()

# Create Celery application
celery_app = Celery(
    "memoire_backend",
    broker=f"redis://localhost:6379/0",  # Default Redis broker
    backend=f"redis://localhost:6379/0",  # Result backend
    include=[
        "app.services.queue.tasks.audio_tasks",
        "app.services.queue.tasks.transcription_tasks",
        "app.services.queue.tasks.processing_tasks"
    ]
)

# Configure Celery
celery_app.conf.update(
    # Task routing
    task_routes={
        "app.services.queue.tasks.audio_tasks.*": {"queue": "audio_processing"},
        "app.services.queue.tasks.transcription_tasks.*": {"queue": "transcription"},
        "app.services.queue.tasks.processing_tasks.*": {"queue": "general_processing"},
        "app.services.queue.tasks.*": {"queue": "default"}
    },
    
    # Task execution settings
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    
    # Task result settings
    result_expires=3600,  # 1 hour
    result_backend_db=1,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Disable prefetching for better load balancing
    task_acks_late=True,
    worker_disable_rate_limits=True,
    
    # Retry settings
    task_default_retry_delay=60,  # 60 seconds
    task_max_retries=3,
    
    # Priority settings
    task_inherit_parent_priority=True,
    task_default_priority=5,
    worker_direct=True,
    
    # Memory and resource management
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,  # 10 minutes hard limit
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Queue configuration
    task_create_missing_queues=True,
    task_default_queue="default",
    task_queues={
        "high_priority": {"routing_key": "high_priority"},
        "transcription": {"routing_key": "transcription"},
        "audio_processing": {"routing_key": "audio_processing"},
        "general_processing": {"routing_key": "general_processing"},
        "default": {"routing_key": "default"}
    }
)

# Configure Redis URL from settings if available
redis_url = os.getenv("REDIS_URL")
if redis_url:
    celery_app.conf.broker_url = redis_url
    celery_app.conf.result_backend = redis_url

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal"""
    logger.info("Celery worker is ready", worker_name=sender.hostname if sender else "unknown")

@worker_shutdown.connect  
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown signal"""
    logger.info("Celery worker is shutting down", worker_name=sender.hostname if sender else "unknown")

# Task failure handler
@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery functionality"""
    logger.info("Debug task executed", task_id=self.request.id)
    return f"Request: {self.request!r}"