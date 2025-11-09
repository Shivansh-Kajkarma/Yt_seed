# celery_worker.py
import os
import sys
from pathlib import Path
from celery import Celery
from celery.signals import worker_process_init
from dotenv import load_dotenv

# ------------------------------------------------------
# STEP 1: Force Python to see project root as importable
# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def setup_python_path():
    """Add BASE_DIR to sys.path for imports"""
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    if str(BASE_DIR.parent) not in sys.path:
        sys.path.insert(0, str(BASE_DIR.parent))


# Call it immediately for the main process
setup_python_path()


# ✅ CRITICAL FIX: Also call it when each worker subprocess starts
@worker_process_init.connect
def configure_worker_process(**kwargs):
    """Called when each Celery worker subprocess is initialized"""
    setup_python_path()
    print(f"🔧 Worker subprocess initialized with sys.path: {sys.path[:3]}")


# ------------------------------------------------------
# STEP 2: Load environment variables
# ------------------------------------------------------
env_path = BASE_DIR / ".env"
load_dotenv(env_path)
print(f"🧩 Loaded .env from: {env_path}")

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    print("❌ CRITICAL: REDIS_URL not found in environment!")
else:
    print(f"✅ REDIS_URL = {REDIS_URL}")

# ------------------------------------------------------
# STEP 3: Initialize Celery
# ------------------------------------------------------
celery_app = Celery(
    "kajkarma_tasks",
    broker=REDIS_URL or "redis://localhost:6379/0",
    backend=REDIS_URL or "redis://localhost:6379/0",
)


# ------------------------------------------------------
# STEP 4: Celery task entry point
# ------------------------------------------------------
@celery_app.task(bind=True)
def run_phase_pipeline(self, sheet_url: str):
    """
    Celery background task that triggers the entire Kajkarma pipeline.
    """
    try:
        # ✅ Force import after sys.path fix
        from utils.pipeline_wrapper import full_pipeline_from_sheet

        result = full_pipeline_from_sheet(sheet_url)
        return {"status": "success", "details": result}
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return {"status": "failed", "error": str(e)[:500]}
