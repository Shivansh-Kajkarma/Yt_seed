import sys, subprocess, os
from datetime import datetime
from utils.mongo_utils import save_json_blob

def record_run_status(run_tag: str, status: str, extra: dict | None = None):
    payload = {
        "run_tag": run_tag,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    save_json_blob(payload, "run_progress", "run_tag", run_tag)

def run_all_phases_for_seed(seed_channel_name: str):
    """
    Hybrid Orchestrator for full pipeline.
    ✅ Uses subprocess isolation (safe for Celery, memory)
    ✅ Logs Mongo progress after each phase
    ✅ Handles quota & crash pauses
    """
    run_tag = seed_channel_name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_run_tag = f"{run_tag}_{timestamp}"

    record_run_status(full_run_tag, "started", {"seed_name": seed_channel_name})

    scripts = [
        ("phase1", "SCRIPTS/phase1_get_videos_and_keywords.py"),
        ("phase2", "SCRIPTS/phase2_get_discovered_channels.py"),
        ("phase2_5", "SCRIPTS/phase2_5_embeddings_filter.py"),
        ("phase3", "SCRIPTS/phase3_tier_scoring.py"),
    ]

    try:
        for label, script in scripts:
            print(f"\n🚀 Running {label.upper()} for {full_run_tag}...")
            subprocess.run([sys.executable, script, full_run_tag], check=True)
            record_run_status(full_run_tag, f"{label}_done")

        record_run_status(full_run_tag, "completed")
        return {"ok": True, "run_tag": full_run_tag, "status": "completed"}

    except subprocess.CalledProcessError as e:
        record_run_status(full_run_tag, "failed", {"error": str(e)})
        return {"ok": False, "error": str(e)}

    except SystemExit as e:
        record_run_status(full_run_tag, "paused_due_to_quota", {"reason": str(e)})
        return {"ok": False, "paused": True, "message": str(e)}

    except Exception as e:
        record_run_status(full_run_tag, "failed", {"error": str(e)})
        raise
