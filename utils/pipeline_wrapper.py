import sys, os  # <-- ADDED OS
from datetime import datetime
from pathlib import Path
import pandas as pd
from celery import Task
import redis  # <-- ADDED REDIS

# --- We need to import the task itself to re-queue it ---
from celery_worker import run_phase_pipeline 

# --- Make sure utils are importable ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# --- Import your utils ---
from utils.mongo_utils import save_json_blob, load_collection_as_df, save_dataframe_to_mongo
from SCRIPTS.phase1_get_videos_and_keywords import main as phase1_main
from SCRIPTS.phase2_get_discovered_channels import main as phase2_main
from SCRIPTS.phase2_5_embeddings_filter import main as phase2_5_main
from SCRIPTS.phase3_tier_scoring import main as phase3_main
from utils.youtube_utils import _load_from_google_sheet
from dotenv import load_dotenv
load_dotenv()

# ==================================================
# 1. GLOBAL CONFIG
# ==================================================

# --- Collection Names ---
QUEUE_COLLECTION = "seed_queue_discovered"
RUN_PROGRESS_COLLECTION = "run_progress"

# --- NEW: Redis Circuit Breaker Config ---
try:
    REDIS_URL = os.getenv("REDIS_URL")
    if not REDIS_URL:
        raise ValueError("REDIS_URL not found in .env")
    
    # This is our global "pause button"
    r = redis.from_url(REDIS_URL)
    YOUTUBE_QUOTA_FLAG_KEY = "youtube_quota_circuit_breaker"
    
    # How long to pause a *single task* when the flag is ON
    RETRY_DELAY_WHEN_PAUSED = 3600  # 1 hour
    
    # How long to pause the *failing task* AND the *global flag*
    RETRY_DELAY_ON_QUOTA_HIT = 90000 # 25 hours
    
    print("✅ Redis client initialized for circuit breaker.")
except Exception as e:
    print(f"❌ CRITICAL: Could not connect to Redis: {e}")
    r = None

# ==================================================
# 2. HELPER FUNCTIONS (Your existing, correct logic)
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


def run_feedback_loop_for_seed(run_tag: str):
    """
    Finds T1/T2 channels from a *single* run and adds
    them to the master queue as *new, individual Celery tasks*.
    """
    print(f"\n--- 🔄 Running Feedback Loop for {run_tag} ---")
    
    all_processed_seed_ids = set()
    try:
        df_all_progress = load_collection_as_df(RUN_PROGRESS_COLLECTION)
        if not df_all_progress.empty:
            all_processed_seed_ids = set(df_all_progress["run_tag"].unique())
    except Exception:
        pass 
        
    try:
        df_phase3 = load_collection_as_df(f"{run_tag.upper()}_phase3")
        if df_phase3.empty:
            print(f"...No Phase 3 results found for {run_tag}. Skipping feedback.")
            return 0 

        df_new_seeds = df_phase3[df_phase3["tier"].isin([1, 2])].copy()
        if df_new_seeds.empty:
            print(f"...No T1/T2 channels found for {run_tag}. Skipping feedback.")
            return 0 

        df_new_seeds["new_run_tag"] = df_new_seeds["Discovered_Channel_Name"].apply(normalize_run_tag)
        new_seed_run_tags = set(df_new_seeds["new_run_tag"].unique())
        
        final_new_run_tags = new_seed_run_tags - all_processed_seed_ids
        
        if not final_new_run_tags:
            print(f"...Found {len(new_seed_run_tags)} T1/T2 channels, but all are already processed/in queue.")
            return 0 

        print(f"...Found {len(final_new_run_tags)} brand new T1/T2 channels to queue.")
        
        final_df_to_queue = df_new_seeds[df_new_seeds["new_run_tag"].isin(final_new_run_tags)]
        final_df_to_queue = final_df_to_queue.drop_duplicates(subset=["Discovered_Channel_ID"])
        
        for index, row in final_df_to_queue.iterrows():
            new_seed_name = row["Discovered_Channel_Name"]
            new_seed_url = row["Discovered_Channel_URL"]
            new_seed_id = row["Discovered_Channel_ID"]

            new_seed_dict = {
                "Channel_Name": new_seed_name,
                "Channel_URL": new_seed_url,
                "Channel_ID": new_seed_id,
                "Discovered_From_Run": run_tag,
                "Queued_At": datetime.now().isoformat(),
                "Status": "queued"
            }
            
            save_dataframe_to_mongo(
                pd.DataFrame([new_seed_dict]),
                collection_name=QUEUE_COLLECTION,
                unique_key_column="Channel_ID" 
            )
            
            print(f"...Queueing new task for: {new_seed_name}")
            run_phase_pipeline.apply_async(
                args=[None, new_seed_dict], 
                countdown=10 
            )
            
        return len(final_df_to_queue)

    except Exception as e:
        print(f"❌ Feedback Loop FAILED for {run_tag}: {e}")
        return 0 

# ==================================================
# 3. MAIN PIPELINE WRAPPER (Updated with Circuit Breaker)
# ==================================================

def full_pipeline_from_sheet(celery_task: Task, sheet_url: str, seed_dict: dict = None):
    """
    Master pipeline controller.
    - If sheet_url is provided: Acts as a "Loader" and queues individual seed tasks.
    - If seed_dict is provided: Acts as a "Processor" for that one seed.
    """
    
    # --- Check for Redis connection ---
    if not r:
        print("❌ CRITICAL: No Redis client. Cannot run pipeline.")
        raise ConnectionError("Failed to connect to Redis for circuit breaker.")
    
    # --- MODE 1: "Loader" Task ---
    if sheet_url:
        print(f"📄 Loading seeds from Google Sheet: {sheet_url}")
        df_seeds_to_process = _load_from_google_sheet(sheet_url)
        
        if df_seeds_to_process is None or df_seeds_to_process.empty:
            print("❌ No seed channels to process in this batch!")
            return {"status": "failed", "reason": "No seeds found in sheet."}
        
        print(f"Found {len(df_seeds_to_process)} seeds. Queueing individual tasks...")
        
        for index, row in df_seeds_to_process.iterrows():
            seed_row_dict = row.to_dict()
            print(f"...Queueing task for {seed_row_dict.get('Channel_Name')}")
            
            run_phase_pipeline.apply_async(
                args=[None, seed_row_dict], # sheet_url is None
                countdown=index * 5 
            )
        
        return {"status": "success", "message": f"Queued {len(df_seeds_to_process)} individual seed tasks."}

    # --- MODE 2: "Processor" Task ---
    elif seed_dict:
        seed_channel_name = seed_dict.get("Channel_Name")
        seed_channel_url = seed_dict.get("Channel_URL")
        run_tag = normalize_run_tag(seed_channel_name)
        
        if not seed_channel_name or not seed_channel_url:
            print(f"❌ Skipping: seed dict is missing Name or URL.")
            return {"status": "failed", "reason": "Missing Name or URL"}

        # --- 1. NEW: CHECK THE CIRCUIT BREAKER ---
        # This is the "pause wall" fix.
        try:
            if r.exists(YOUTUBE_QUOTA_FLAG_KEY):
                print(f"🚦 Global Quota Pause is ON. Retrying {run_tag} in 1 hour.")
                # Pause for 1 hour, then check again
                celery_task.retry(countdown=RETRY_DELAY_WHEN_PAUSED)
                # We raise SystemExit to be consistent, though retry() does it.
                raise SystemExit("Global quota pause is active.")
        except Exception as redis_e:
            print(f"⚠️ WARNING: Could not check Redis quota flag: {redis_e}")
            # We'll proceed, but this is a risk.
        
        # --- 2. EXISTING: Smart Retry "Skip" Logic ---
        print(f"\n🚀🚀🚀 Starting pipeline for ONE seed: {seed_channel_name} (tag: {run_tag}) 🚀🚀🚀")
        try:
            progress = load_collection_as_df(RUN_PROGRESS_COLLECTION, {"run_tag": run_tag})
            if not progress.empty and progress.iloc[0]["status"] == "completed":
                print(f"✅ Seed '{seed_channel_name}' is already marked 'completed'. Skipping.")
                return {"status": "skipped", "reason": "Already completed."}
        except Exception as e:
            print(f"⚠️ Could not check run_progress: {e}")

        
        record_run_status(run_tag, "started", {
            "seed_name": seed_channel_name, 
            "seed_url": seed_channel_url
        })

        try:
            # --- This is your existing pipeline flow ---
            phase1_main(run_tag, seed_channel_name, seed_channel_url)
            record_run_status(run_tag, "phase1_done")

            phase2_main(run_tag)
            record_run_status(run_tag, "phase2_done")

            phase2_5_main(run_tag)
            record_run_status(run_tag, "phase2_5_done")

            phase3_main(run_tag)
            record_run_status(run_tag, "phase3_done")

            record_run_status(run_tag, "completed")
            
            # --- Feedback Loop ---
            new_seeds = run_feedback_loop_for_seed(run_tag)
            
            return {"status": "success", "message": f"Seed {run_tag} processed. Queued {new_seeds} new seeds."}

        # --- 3. UPDATED: AUTO-RESUME (SETS the circuit breaker) ---
        except SystemExit as e:
            # We check if the error message is *actually* our quota message
            if "YouTube quota exceeded" in str(e):
                # This is a REAL quota pause
                print(f"🛑 QUOTA PAUSE DETECTED for {run_tag}: {e}")
                record_run_status(run_tag, "paused_due_to_quota", {"reason": str(e)})
                
                # --- Set the global flag ---
                try:
                    if r:
                        print(f"🚦 SETTING Global Quota Pause flag for 25 hours...")
                        # Set the flag with a 25-hour expiration
                        r.set(YOUTUBE_QUOTA_FLAG_KEY, "true", ex=RETRY_DELAY_ON_QUOTA_HIT)
                    else:
                        print("❌ Cannot set quota flag: Redis client not found.")
                except Exception as redis_e:
                    print(f"❌ FAILED to set Redis quota flag: {redis_e}")
                
                # --- Retry the task ---
                print(f"...Telling Celery to retry THIS SEED in 25 hours...")
                celery_task.retry(countdown=RETRY_DELAY_ON_QUOTA_HIT, exc=e)
                
            else:
                # This is a DIFFERENT SystemExit (like Ctrl+C or a code bug)
                # We should NOT pause. We should let it fail gracefully.
                print(f"⚠️ A non-quota SystemExit was caught (e.g., Ctrl+C): {e}")
            
            # mUST raise the exception again to stop the wrapper
            raise e 
            
        except Exception as e:
            print(f"❌ PIPELINE FAILED for {run_tag}: {e}")
            record_run_status(run_tag, "failed", {"error": str(e)[:300]})
            return {"status": "failed", "error": str(e)[:300]}
            
    else:
        print("❌ ERROR: Task called with no sheet_url and no seed_dict. Nothing to do.")
        return {"status": "failed", "reason": "No input provided."}