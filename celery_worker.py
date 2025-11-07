import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("kajkarma_pipeline", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task(bind=True)
def run_pipeline_task(self, seed_channel_name: str):
    """
    Celery task to run the full Kajkarma YouTube competitor pipeline.
    Each run is fully isolated & Mongo-logged.
    """
    from utils.pipeline_wrapper import run_all_phases_for_seed

    print(f"🔥 Celery Task Started for seed: {seed_channel_name}")
    try:
        result = run_all_phases_for_seed(seed_channel_name)
        status = "success" if result.get("ok") else "paused"
        return {"status": status, "details": result}
    except Exception as e:
        print(f"❌ Celery pipeline failed for {seed_channel_name}: {e}")
        return {"status": "failed", "error": str(e)[:500]}
