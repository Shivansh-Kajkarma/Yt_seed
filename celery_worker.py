# celery_worker.py
import os
from celery import Celery

# Read Redis URL (falls back to localhost)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "kajkarma_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

@celery_app.task(bind=True)
def run_pipeline_task(self, run_tag: str):
    """
    Celery task: run the full pipeline for a single run_tag.
    Returns a small summary dict (also persisted to Mongo via wrapper).
    """
    from utils.pipeline_wrapper import full_pipeline
    try:
        result = full_pipeline(run_tag)
        return {"status": "success" if result.get("ok") else "paused", "details": result}
    except Exception as e:
        return {"status": "failed", "error": str(e)[:500]}
