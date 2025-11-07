import sys, subprocess, os
from datetime import datetime
from utils.mongo_utils import save_json_blob, check_quota_and_pause

def record_run_status(run_tag: str, status: str, extra: dict | None = None):
    # ... (this function is unchanged) ...
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
     Passes the simple 'run_tag' (e.g., 'moon') to all scripts.
     CORRECTLY handles Quota Pauses.
    """
    run_tag = seed_channel_name.lower().replace(" ", "_")

    record_run_status(run_tag, "started", {"seed_name": seed_channel_name})

    scripts = [
        ("phase1", "SCRIPTS/phase1_get_videos_and_keywords.py"),
        ("phase2", "SCRIPTS/phase2_get_discovered_channels.py"),
        ("phase2_5", "SCRIPTS/phase2_5_embeddings_filter.py"),
        ("phase3", "SCRIPTS/phase3_tier_scoring.py"),
    ]

    try:
        for label, script in scripts:
            print(f"\n🚀 Running {label.upper()} for {run_tag}...")
            # --- THIS IS THE KEY ---
            # We add capture_output=True to read the error message
            subprocess.run(
                [sys.executable, script, run_tag], 
                check=True, 
                capture_output=True, # <-- SOTA FIX 1
                text=True
            )
            record_run_status(run_tag, f"{label}_done")

        record_run_status(run_tag, "completed")
        return {"ok": True, "run_tag": run_tag, "status": "completed"}

    except subprocess.CalledProcessError as e:
        # --- SOTA FIX 2 ---
        # The script failed. NOW we check if it was our quota error.
        # The "SystemExit" message gets printed to stderr.
        if "YouTube quota exceeded" in e.stderr:
            print(f"🛑 PAUSE DETECTED: Quota limit hit during {label}.")
            # The check_quota_and_pause() function already logged to Mongo.
            # We just return the "paused" status to Celery.
            record_run_status(run_tag, "paused_due_to_quota", {"reason": "quotaExceeded"})
            return {"ok": False, "paused": True, "message": "YouTube quota exceeded."}
        else:
            # It was a *different* crash (a real Python error)
            print(f"❌ SCRIPT FAILED: A non-quota error occurred in {label}.")
            record_run_status(run_tag, "failed", {"error": str(e.stderr)})
            return {"ok": False, "error": str(e.stderr)}
    
    except Exception as e:
        # Catch any other weird errors
        record_run_status(run_tag, "failed", {"error": str(e)})
        raise