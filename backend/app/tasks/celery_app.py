"""Celery application configuration."""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from celery import Celery
from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "router_optimizer",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        'app.tasks.optimization_task',
        'app.tasks.map_processing_task',
        'app.tasks.llm_processing_task'
    ]
)

# Configure Celery
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timezone
    timezone='UTC',
    enable_utc=True,

    # Task tracking
    task_track_started=True,
    result_extended=True,

    # Timeouts
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # Soft limit at 55 minutes

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for heavy tasks
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks

    # Task result expiration
    result_expires=86400,  # Results expire after 24 hours

    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Configure task routes (optional)
# NOTE: Using default 'celery' queue for simplicity
# To use separate queues, start worker with: celery -A app.tasks.celery_app worker -Q optimization
# celery_app.conf.task_routes = {
#     'app.tasks.optimization_task.*': {'queue': 'optimization'},
#     'app.tasks.map_processing_task.*': {'queue': 'optimization'},
# }

# Configure beat schedule for periodic tasks (if needed)
celery_app.conf.beat_schedule = {
    # Example: Clean up old jobs every hour
    # 'cleanup-old-jobs': {
    #     'task': 'app.tasks.cleanup.cleanup_old_jobs',
    #     'schedule': 3600.0,
    # },
}
