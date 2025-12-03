import os
import sys
from pathlib import Path
from celery import Celery
from celery.signals import worker_process_init
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

def setup_python_path():
    """Add BASE_DIR to sys.path for imports"""
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    if str(BASE_DIR.parent) not in sys.path:
        sys.path.insert(0, str(BASE_DIR.parent))

setup_python_path()

@worker_process_init.connect
def configure_worker_process(**kwargs):
    setup_python_path()
    print(f"Worker subprocess initialized with sys.path: {sys.path[:3]}")

env_path = BASE_DIR / ".env"
load_dotenv(env_path)

REDIS_URL = os.getenv("REDIS_URL")

celery_app = Celery(
    "kajkarma_tasks",
    broker=REDIS_URL or "redis://localhost:6379/0",
    backend=REDIS_URL or "redis://localhost:6379/0",
)

# Optional: Configure routes here if you want defaults, 
# but we are handling it dynamically in main.py
celery_app.conf.task_routes = {
    'celery_worker.run_phase_pipeline': {'queue': 'default'} 
}

@celery_app.task(bind=True)
def run_phase_pipeline(
    self,
    sheet_url: str,
    seed_dict: dict = None,
    input_format: str = "General",
    clients_intent: str = "General",
    pipeline_execution_id: str = None, # <--- NEW ARGUMENT
):
    """
    Celery background task.
    Now accepts pipeline_execution_id to track the lineage of the run.
    """
    try:
        from utils.pipeline_wrapper import full_pipeline_from_sheet

        # We need to decide which API Key to use based on the Queue we are running in.
        # However, getting the Queue name inside the task is tricky.
        # BETTER STRATEGY: We read the API Key from the Environment Variable 
        # that we set in the tmux session (e.g. export YOUTUBE_API_KEY=...)
        
        # We pass this Env Var key to the wrapper
        api_key = os.getenv("YOUTUBE_API_KEY") 
        
        if not api_key:
            # Fallback for local testing or if env var is missing
            print("⚠️ No YOUTUBE_API_KEY found in env (Task specific). Checking .env fallback...")
            api_key = os.getenv("YOUTUBE_API_KEY_PODCAST") # Default or fail
            if not api_key:
                 return {"status": "failed", "error": "CRITICAL: No API Key available for this worker."}

        result = full_pipeline_from_sheet(
            self, 
            sheet_url, 
            seed_dict, 
            input_format, 
            clients_intent,
            pipeline_execution_id, # <--- PASS IT DOWN
            api_key # <--- PASS THE KEY
        )

        return {"status": "success", "details": result}

    except SystemExit as e:
        print(f"Task {self.request.id} is being paused and retried.")
        return {"status": "paused", "details": "Task paused due to quota."}

    except Exception as e:
        print(f"Pipeline error: {e}")
        return {"status": "failed", "error": str(e)[:500]}