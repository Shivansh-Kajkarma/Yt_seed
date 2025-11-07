# utils/pipeline_wrapper.py
import sys
from datetime import datetime
from pathlib import Path

# Make repo root importable if needed
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Import the phase "main" functions
from SCRIPTS.phase1_get_videos_and_keywords import main as phase1_main
from SCRIPTS.phase2_get_discovered_channels import main as phase2_main
from SCRIPTS.phase2_5_embeddings_filter import main as phase2_5_main
from SCRIPTS.phase3_tier_scoring import main as phase3_main

from utils.mongo_utils import save_json_blob

def normalize_run_tag(seed_channel_name: str) -> str:
    return seed_channel_name.upper().replace(" ", "_")

def record_run_status(run_tag: str, status: str, extra: dict | None = None):
    payload = {
        "run_tag": run_tag,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    # One document per run_tag, continuously overwritten
    save_json_blob(payload, "run_progress", "run_tag", run_tag)

def full_pipeline(run_tag: str):
    """
    Run all phases sequentially for a single run_tag (seed).
    Assumes each phase's main(run_tag: str) exists and reads/writes Mongo.
    """
    record_run_status(run_tag, "started")
    try:
        phase1_main(run_tag)     # pulls seed videos + saves to Mongo
        record_run_status(run_tag, "phase1_done")

        phase2_main(run_tag)     # discovery cached raw → Mongo
        record_run_status(run_tag, "phase2_done")

        phase2_5_main(run_tag)   # embeddings + scores → Mongo + CSV
        record_run_status(run_tag, "phase2_5_done")

        phase3_main(run_tag)     # GPT tiering → Mongo + CSV
        record_run_status(run_tag, "phase3_done")

        record_run_status(run_tag, "completed")
        return {"ok": True, "message": f"Completed pipeline for {run_tag}"}

    except SystemExit as e:
        # Used by quota-safe exit
        record_run_status(run_tag, "paused_due_to_quota", {"reason": str(e)})
        return {"ok": False, "paused": True, "message": str(e)}
    except Exception as e:
        record_run_status(run_tag, "failed", {"error": str(e)[:500]})
        raise
