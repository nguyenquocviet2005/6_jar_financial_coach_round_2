"""Celery application configuration for background tasks."""

from celery import Celery
from celery.schedules import crontab
from src.config import get_settings

settings = get_settings()

# Create Celery app
app = Celery(
    'mlops_tasks',
    broker=settings.celery_broker_url,
    backend=settings.redis_url,
    include=[
        'src.ai_coach.tasks',
        'src.classification.tasks',
        'src.prediction.tasks',
        'src.fine_tuning.tasks',
    ]
)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

# Scheduled tasks
app.conf.beat_schedule = {
    'daily-prediction-update': {
        'task': 'src.prediction.tasks.generate_daily_predictions',
        'schedule': crontab(hour=1, minute=0),  # Run at 1 AM daily
    },
    'weekly-model-retraining': {
        'task': 'src.classification.tasks.retrain_classification_model',
        'schedule': crontab(day_of_week=0, hour=2, minute=0),  # Run weekly on Sunday at 2 AM
    },
    'monthly-fine-tuning': {
        'task': 'src.fine_tuning.tasks.trigger_monthly_fine_tuning',
        'schedule': crontab(day_of_month=1, hour=3, minute=0),  # Run monthly on the 1st at 3 AM
    },
    'model-performance-monitoring': {
        'task': 'src.common.tasks.monitor_model_performance',
        'schedule': crontab(minute=0),  # Run hourly
    },
}

app.conf.timezone = 'UTC'

if __name__ == '__main__':
    app.start() 