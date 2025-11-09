import sys
from datetime import datetime
from pathlib import Path
from utils.mongo_utils import save_json_blob
from SCRIPTS.phase1_get_videos_and_keywords import main as phase1_main
from SCRIPTS.phase2_get_discovered_channels import main as phase2_main
from SCRIPTS.phase2_5_embeddings_filter import main as phase2_5_main
from SCRIPTS.phase3_tier_scoring import main as phase3_main
from utils.youtube_utils import _load_from_google_sheet  # same one used in phase1
from dotenv import load_dotenv
load_dotenv()

def normalize_run_tag(seed_channel_name: str) -> str:
    """Convert seed name to safe run_tag."""
    return seed_channel_name.lower().replace(" ", "_")

def record_run_status(run_tag: str, status: str, extra: dict | None = None):
    payload = {
        "run_tag": run_tag,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    save_json_blob(payload, "run_progress", "run_tag", run_tag)
def full_pipeline_from_sheet(sheet_url: str):
    """
    Master pipeline that reads seed channels from Google Sheet
    and runs all phases for each one sequentially.
    """
    print(f"📄 Loading seeds from Google Sheet: {sheet_url}")
    df_seeds = _load_from_google_sheet(sheet_url)
    if df_seeds is None or df_seeds.empty:
        print("❌ No seed channels found in sheet!")
        return

    # --- CHANGED: Iterate over rows, not unique names ---
    for index, row in df_seeds.iterrows():
        seed_channel_name = row["Channel_Name"]
        seed_channel_url = row["Channel_URL"]
        
        if not seed_channel_name or not seed_channel_url:
            print(f"⚠️ Skipping row {index}: missing data")
            continue
            
        run_tag = normalize_run_tag(seed_channel_name)
        
        print(f"\n🚀 Starting pipeline for seed: {seed_channel_name} (tag: {run_tag})")
        record_run_status(run_tag, "started", {
            "seed_name": seed_channel_name, 
            "seed_url": seed_channel_url
        })

        try:
            # --- CHANGED: Pass all three args to phase1_main ---
            phase1_main(run_tag, seed_channel_name, seed_channel_url)
            record_run_status(run_tag, "phase1_done")

            phase2_main(run_tag)
            record_run_status(run_tag, "phase2_done")

            phase2_5_main(run_tag)
            record_run_status(run_tag, "phase2_5_done")

            phase3_main(run_tag)
            record_run_status(run_tag, "phase3_done")

            record_run_status(run_tag, "completed")
        except SystemExit as e:
            print(f"🛑 PAUSE DETECTED for {run_tag}: {e}")
            record_run_status(run_tag, "paused_due_to_quota", {"reason": str(e)})
        except Exception as e:
            print(f"❌ PIPELINE FAILED for {run_tag}: {e}")
            record_run_status(run_tag, "failed", {"error": str(e)[:300]})

    print("\n✅ All seeds from Google Sheet have been processed.")