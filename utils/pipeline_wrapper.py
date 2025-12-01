

import sys, os
from datetime import datetime
from pathlib import Path
import pandas as pd
from celery import Task
import redis  # We need Redis for the Global Counter

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
BATCH_SIZE = 3  # Process 3 channels...
# BATCH_COOLDOWN = 25 * 3600   # ...then wait 25 hours (in seconds)
# STAGGER_DELAY = 120          # Wait 2 mins between channels in the same batch
# Testing
BATCH_COOLDOWN = 90000  # ...then wait 25 hours (in seconds)
STAGGER_DELAY = 10  # Wait 2 mins between channels in the same batch

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
    # Merge extra fields into payload
    if extra:
        payload.update(extra)
    save_json_blob(payload, RUN_PROGRESS_COLLECTION, "run_tag", run_tag)


def get_next_schedule_delay():
    """
    ATOMICALLY increments the global counter and calculates
    the delay for the next task.
    This guarantees NO overlaps, even if multiple processes run at once.
    """
    # 1. Get a ticket number (Atomic Increment)
    ticket_number = r_client.incr(GLOBAL_COUNTER_KEY) - 1  # 0-indexed

    # 2. Which batch is this ticket in?
    batch_number = ticket_number // BATCH_SIZE

    # 3. Calculate Delay
    batch_delay = batch_number * BATCH_COOLDOWN
    intra_stagger = (ticket_number % BATCH_SIZE) * STAGGER_DELAY

    total_delay = batch_delay + intra_stagger

    print(
        f"🎫 Issued Ticket #{ticket_number} | Batch {batch_number} | Delay: {total_delay / 3600:.2f} hrs"
    )
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
            if not df_prog.empty:
                processed = set(df_prog["run_tag"].unique())
        except:
            pass

        # Load T1/T2 Results
        df_phase4 = load_collection_as_df(f"{run_tag.upper()}_final_ranked")
        if df_phase4.empty:
            return 0

        # Filter for Tier 1 and 2
        df_new_seeds = df_phase4[df_phase4["Final_Tier"].isin([1, 2])].copy()
        if df_new_seeds.empty:
            return 0

        # Create IDs
        df_new_seeds["new_run_tag"] = df_new_seeds["Discovered_Channel_Name"].apply(
            normalize_run_tag
        )

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
            save_dataframe_to_mongo(
                pd.DataFrame([new_seed_dict]), QUEUE_COLLECTION, "Channel_ID"
            )

            # --- CRITICAL: GET GLOBAL SCHEDULE TICKET ---
            # This puts the new seed at the END of the 25h line
            delay_seconds = get_next_schedule_delay()

            # Fire Task
            run_phase_pipeline.apply_async(
                args=[None, new_seed_dict, input_format, clients_intent],
                countdown=delay_seconds,
            )
            count += 1
            print(
                f"   Waitlist +1: {new_seed_name} (Starts in {delay_seconds / 3600:.1f}h)"
            )

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
        if df is None or df.empty:
            return {"status": "failed"}

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
        seed_channel_name = seed_dict.get("Channel_Name")
        seed_channel_url = seed_dict.get("Channel_URL")
        seed_channel_id = seed_dict.get("Channel_ID", "")
        run_tag = normalize_run_tag(seed_channel_name)

        if not seed_channel_name or not seed_channel_url:
            print(f"❌ Skipping: seed dict is missing Name or URL.")
            return {"status": "failed", "reason": "Missing Name or URL"}

        print(
            f"\n🚀🚀🚀 Starting SCHEDULED PIPELINE for: {seed_channel_name} (tag: {run_tag}) 🚀🚀🚀"
        )
        print(f"🎯 Configuration: Format='{input_format}' | Intent='{clients_intent}'")

        # Check if already completed
        try:
            progress = load_collection_as_df(
                RUN_PROGRESS_COLLECTION, {"run_tag": run_tag}
            )
            if not progress.empty and progress.iloc[0]["status"] == "completed":
                print(
                    f"✅ Seed '{seed_channel_name}' is already marked 'completed'. Skipping."
                )
                return {"status": "skipped", "reason": "Already completed."}
        except Exception as e:
            print(f"⚠️ Could not check run_progress: {e}")

        record_run_status(
            run_tag,
            "started",
            {
                "seed_name": seed_channel_name,
                "seed_url": seed_channel_url,
                "current_phase": "initializing",
            },
        )

        try:
            # ==================================================
            # PHASE 1: ytdlp_scripts/phase1_seed_processing.py
            # ==================================================
            print(f"\n--- [PHASE 1] Deep Seed Analysis (yt-dlp) ---")
            import subprocess

            phase1_result = subprocess.run(
                [
                    "python3",
                    os.path.join(
                        BASE_DIR, "ytdlp_scripts", "phase1_seed_processing.py"
                    ),
                    run_tag,
                    seed_channel_name,
                    seed_channel_url,
                    input_format,
                    clients_intent,
                ],
                capture_output=True,
                text=True,
            )

            if phase1_result.returncode != 0:
                print(f"⚠️ Phase 1 failed:")
                print(f"   stderr: {phase1_result.stderr}")
                print(f"   stdout: {phase1_result.stdout}")
                record_run_status(
                    run_tag, "skipped", {"reason": "Phase 1 failed - no videos found"}
                )
                return {"status": "skipped", "message": "Seed had no videos."}

            print(phase1_result.stdout)
            record_run_status(run_tag, "in_progress", {"current_phase": "phase1_done"})

            # ==================================================
            # PHASE 2: Channel Discovery
            # ==================================================
            print(f"\n--- [PHASE 2] Channel Discovery ---")
            phase2_main(run_tag)
            record_run_status(run_tag, "in_progress", {"current_phase": "phase2_done"})

            # ==================================================
            # PHASE 3: Multi-Step Filtering & Scoring
            # ==================================================

            # STEP 1: API-based filter
            print(f"\n--- [PHASE 3.1] API Metadata Filter ---")
            phase3_step1_result = subprocess.run(
                [
                    "python3",
                    os.path.join(
                        BASE_DIR, "ytdlp_scripts", "phase3_step1_api_filter.py"
                    ),
                    run_tag,
                ],
                capture_output=True,
                text=True,
            )

            if phase3_step1_result.returncode != 0:
                print(f"⚠️ Phase 3 Step 1 failed: {phase3_step1_result.stderr}")
            else:
                print(phase3_step1_result.stdout)

            record_run_status(
                run_tag, "in_progress", {"current_phase": "phase3_step1_done"}
            )

            # STEP 2: Deep scan with yt-dlp
            print(f"\n--- [PHASE 3.2] Deep Scan (yt-dlp) ---")
            phase3_step2_result = subprocess.run(
                [
                    "python3",
                    os.path.join(
                        BASE_DIR, "ytdlp_scripts", "phase3_step2_deep_scan.py"
                    ),
                    run_tag,
                ],
                capture_output=True,
                text=True,
            )

            if phase3_step2_result.returncode != 0:
                print(f"⚠️ Phase 3 Step 2 failed: {phase3_step2_result.stderr}")
            else:
                print(phase3_step2_result.stdout)

            record_run_status(
                run_tag, "in_progress", {"current_phase": "phase3_step2_done"}
            )

            # STEP 3A: LLM format verification
            print(f"\n--- [PHASE 3.3A] LLM Format Verification ---")
            phase3_step3a_result = subprocess.run(
                [
                    "python3",
                    os.path.join(
                        BASE_DIR, "ytdlp_scripts", "phase3_step3a_scoring_llm.py"
                    ),
                    run_tag,
                ],
                capture_output=True,
                text=True,
            )

            if phase3_step3a_result.returncode != 0:
                print(f"⚠️ Phase 3 Step 3A failed: {phase3_step3a_result.stderr}")
            else:
                print(phase3_step3a_result.stdout)

            record_run_status(
                run_tag, "in_progress", {"current_phase": "phase3_step3a_done"}
            )

            # STEP 3B: Embedding similarity
            print(f"\n--- [PHASE 3.3B] Embedding Similarity ---")
            phase3_step3b_result = subprocess.run(
                [
                    "python3",
                    os.path.join(
                        BASE_DIR, "ytdlp_scripts", "phase3_step3b_scoring_emb.py"
                    ),
                    run_tag,
                ],
                capture_output=True,
                text=True,
            )

            if phase3_step3b_result.returncode != 0:
                print(f"⚠️ Phase 3 Step 3B failed: {phase3_step3b_result.stderr}")
            else:
                print(phase3_step3b_result.stdout)

            record_run_status(
                run_tag, "in_progress", {"current_phase": "phase3_step3b_done"}
            )

            # ==================================================
            # PHASE 4: Final Ranking & Tiering
            # ==================================================
            print(f"\n--- [PHASE 4] Final Ranking ---")
            phase4_result = subprocess.run(
                [
                    "python3",
                    os.path.join(BASE_DIR, "ytdlp_scripts", "phase4_final_ranking.py"),
                    run_tag,
                ],
                capture_output=True,
                text=True,
            )

            if phase4_result.returncode != 0:
                print(f"⚠️ Phase 4 failed: {phase4_result.stderr}")
            else:
                print(phase4_result.stdout)

            record_run_status(run_tag, "in_progress", {"current_phase": "phase4_done"})

            record_run_status(
                run_tag, "completed", {"current_phase": "all_phases_complete"}
            )

            # --- Feedback Loop (This will add NEW items to the END of the schedule) ---
            new_seeds_count = run_feedback_loop_for_seed(
                run_tag, input_format, clients_intent
            )

            return {
                "status": "success",
                "message": f"Seed {run_tag} processed. Queued {new_seeds_count} new seeds to schedule.",
            }

        except Exception as e:
            print(f"❌ PIPELINE FAILED for {run_tag}: {e}")
            record_run_status(run_tag, "failed", {"error": str(e)[:300]})
            return {"status": "failed", "error": str(e)[:300]}

    else:
        print(
            "❌ ERROR: Task called with no sheet_url and no seed_dict. Nothing to do."
        )
        return {"status": "failed", "reason": "No input provided."}
