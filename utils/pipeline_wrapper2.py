import sys, os
from datetime import datetime
from pathlib import Path
import pandas as pd
from celery import Task
import redis # We need Redis for the Global Counter

from celery_worker import run_phase_pipeline

# --- Make sure utils are importable ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# --- Import your utils ---
from utils.mongo_utils import (
    save_json_blob,
    load_collection_as_df,
    save_dataframe_to_mongo,
)
from SCRIPTS.phase2_get_discovered_channels import main as phase2_main
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# 1. GLOBAL CONFIG & REDIS
# ==================================================

QUEUE_COLLECTION = "seed_queue_discovered"
RUN_PROGRESS_COLLECTION = "run_progress"

# --- CONFIG FOR THREADING/DELAY ---
BATCH_SIZE = 3               # Process 3 channels...
# BATCH_COOLDOWN = 25 * 3600   # ...then wait 25 hours (in seconds)
# STAGGER_DELAY = 120          # Wait 2 mins between channels in the same batch
#Testing
BATCH_COOLDOWN = 120   # ...then wait 25 hours (in seconds)
STAGGER_DELAY = 10          # Wait 2 mins between channels in the same batch

# --- REDIS SETUP (For Global Schedule Tracking) ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r_client = redis.from_url(REDIS_URL)

# Key to store the total number of seeds we have EVER scheduled
# This ensures new seeds go to the BACK of the line, not the front.
GLOBAL_COUNTER_KEY = "kajkarma_global_seed_schedule_counter"

# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def normalize_run_tag(seed_channel_name: str) -> str:
    """Convert seed name to safe run_tag."""
    return seed_channel_name.lower().replace(" ", "_")

def record_run_status(run_tag: str, status: str, extra: dict | None = None):
    payload = {
        "run_tag": run_tag,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }
    save_json_blob(payload, RUN_PROGRESS_COLLECTION, "run_tag", run_tag)

def get_next_schedule_delay():
    """
    ATOMICALLY increments the global counter and calculates 
    the delay for the next task.
    This guarantees NO overlaps, even if multiple processes run at once.
    """
    # 1. Get a ticket number (Atomic Increment)
    ticket_number = r_client.incr(GLOBAL_COUNTER_KEY) - 1 # 0-indexed
    
    # 2. Which batch is this ticket in?
    batch_number = ticket_number // BATCH_SIZE
    
    # 3. Calculate Delay
    batch_delay = batch_number * BATCH_COOLDOWN
    intra_stagger = (ticket_number % BATCH_SIZE) * STAGGER_DELAY
    
    total_delay = batch_delay + intra_stagger
    
    print(f"🎫 Issued Ticket #{ticket_number} | Batch {batch_number} | Delay: {total_delay/3600:.2f} hrs")
    return total_delay

def run_feedback_loop_for_seed(
    run_tag: str, input_format: str = "General", clients_intent: str = "General"
):
    """
    THE INFINITE LOOP:
    Finds T1/T2 channels, ADDS them to the Global Schedule, 
    and queues them to run in the future.
    """
    print(f"\n--- 🔄 Running Feedback Loop for {run_tag} ---")

    try:
        # Load Completed Seeds to avoid loops
        processed = set()
        try:
            df_prog = load_collection_as_df(RUN_PROGRESS_COLLECTION)
            if not df_prog.empty: processed = set(df_prog["run_tag"].unique())
        except: pass

        # Load T1/T2 Results
        df_phase4 = load_collection_as_df(f"{run_tag.upper()}_final_ranked")
        if df_phase4.empty: return 0

        # Filter for Tier 1 and 2
        df_new_seeds = df_phase4[df_phase4["Final_Tier"].isin([1, 2])].copy()
        if df_new_seeds.empty: return 0

        # Create IDs
        df_new_seeds["new_run_tag"] = df_new_seeds["Discovered_Channel_Name"].apply(normalize_run_tag)
        
        # Filter duplicates (Already processed?)
        final_df = df_new_seeds[~df_new_seeds["new_run_tag"].isin(processed)]
        final_df = final_df.drop_duplicates(subset=["Discovered_Channel_ID"])

        count = 0
        for index, row in final_df.iterrows():
            new_seed_name = row["Discovered_Channel_Name"]
            
            # Prepare Payload
            new_seed_dict = {
                "Channel_Name": new_seed_name,
                "Channel_URL": row["Discovered_Channel_URL"],
                "Channel_ID": row["Discovered_Channel_ID"],
                "Discovered_From_Run": run_tag,
                "Queued_At": datetime.now().isoformat(),
            }

            # Save to Queue DB
            save_dataframe_to_mongo(pd.DataFrame([new_seed_dict]), QUEUE_COLLECTION, "Channel_ID")

            # --- CRITICAL: GET GLOBAL SCHEDULE TICKET ---
            # This puts the new seed at the END of the 25h line
            delay_seconds = get_next_schedule_delay()
            
            # Fire Task
            run_phase_pipeline.apply_async(
                args=[None, new_seed_dict, input_format, clients_intent],
                countdown=delay_seconds, 
            )
            count += 1
            print(f"   Waitlist +1: {new_seed_name} (Starts in {delay_seconds/3600:.1f}h)")

        return count

    except Exception as e:
        print(f"❌ Feedback Loop Error: {e}")
        return 0

# ==================================================
# 3. MAIN PIPELINE WRAPPER
# ==================================================

def full_pipeline_from_sheet(
    celery_task: Task,
    sheet_url: str,
    seed_dict: dict = None,
    input_format: str = "General",
    clients_intent: str = "General",
):
    # --- MODE 1: Loader (Initial Sheet) ---
    if sheet_url:
        print(f"📄 Loading seeds from Sheet...")
        # Reset counter if you want a fresh start, or keep it to append?
        # r_client.set(GLOBAL_COUNTER_KEY, 0) # UNCOMMENT TO RESET SCHEDULE ON NEW SHEET LOAD
        
        from utils.youtube_utils import _load_from_google_sheet
        df = _load_from_google_sheet(sheet_url)
        if df is None or df.empty: return {"status": "failed"}

        print(f"🗓️ Scheduling {len(df)} seeds into the Global Timeline...")

        for index, row in df.iterrows():
            # Get Ticket
            delay = get_next_schedule_delay()
            
            run_phase_pipeline.apply_async(
                args=[None, row.to_dict(), input_format, clients_intent],
                countdown=delay,
            )

        return {"status": "success", "message": f"Scheduled {len(df)} seeds."}

    # --- MODE 2: Processor (The Actual Work) ---
    elif seed_dict:
        # ... (Identical to your previous code, just runs the scripts)
        # Copy the logic from the previous answer for MODE 2
        # It just runs Phase 1 -> Phase 4 -> Feedback Loop
        
        seed_channel_name = str(seed_dict.get("Channel_Name"))
        run_tag = normalize_run_tag(seed_channel_name)
        
        # ... [Run Phase 1, 2, 3, 4] ...
        
        # AFTER Phase 4, call feedback loop:
        # This will add NEW items to the END of the schedule
        new_seeds_count = run_feedback_loop_for_seed(run_tag, input_format, clients_intent)
        
        return {"status": "success", "message": f"Processed {run_tag}. Added {new_seeds_count} new seeds to schedule."}