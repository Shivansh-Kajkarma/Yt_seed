import pandas as pd
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import re  # <-- ADDED for your new function
from typing import Optional  # <-- ADDED for your new function

# --- Make sure utils are importable ---
# This assumes your 'utils' folder is in the parent directory of 'SCRIPTS'
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from utils.youtube_utils import fetch_for_seed_channels, _load_from_google_sheet
    from utils.fingerprint_llm_utils import (
        get_channel_fingerprint_oneshot
    )
    from utils.mongo_utils import save_dataframe_to_mongo, save_json_blob
except ImportError:
    print("Error: Could not import from 'utils' directory.")
    print(f"Ensure 'utils' is at this path: {BASE_DIR / 'utils'}")
    sys.exit(1)

# ==================================================
# 1. CONFIGURATION
# ==================================================
#
# --- CHOOSE YOUR INPUT SOURCE ---
# Set this to "csv" or "sheet"
SEED_INPUT_SOURCE = "csv" 
#
# --- Set a memorable name for this run (e.g., "moon", "vox_analysis")
RUN_TAG = "moon"  # or dynamically from args/env later
MONGO_COLLECTION_PREFIX = f"{RUN_TAG.upper()}_phase1"
#
# ==================================================

# --- Input/Output Directories ---
SEED_CSV = BASE_DIR / "seed_channels.csv"
SEED_GOOGLE_SHEET_URL = "YOUR_PUBLIC_GOOGLE_SHEET_URL_HERE" # <-- ADD YOUR URL

OUTPUT_DIR_VIDEOS = BASE_DIR / "PHASE_1_OUTPUTS"
OUTPUT_DIR_FINGERPRINTS = BASE_DIR / "FINGERPRINTS_ONESHOT"

# --- Create a unique ID for this run ---
RUN_ID = f"{RUN_TAG}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- Define final output paths ---
FINAL_VIDEO_CSV_PATH = OUTPUT_DIR_VIDEOS / f"sample_videos_{RUN_ID}.csv"
FINAL_FINGERPRINT_JSON_PATH = OUTPUT_DIR_FINGERPRINTS / f"fingerprints_oneshot_{RUN_ID}.json"


def main():
    """
    Main pipeline script:
    1. Fetches video data from YouTube API.
    2. Saves video data to a timestamped CSV.
    3. Generates one-shot fingerprints for each channel.
    4. Saves fingerprints to a timestamped JSON.
    """
    print(f"--- 🚀 Starting New Pipeline Run ---")
    print(f"Run ID: {RUN_ID}")

    # --- 0. Create Output Directories ---
    try:
        OUTPUT_DIR_VIDEOS.mkdir(exist_ok=True)
        OUTPUT_DIR_FINGERPRINTS.mkdir(exist_ok=True)
        print(f"Ensured output directory exists: {OUTPUT_DIR_VIDEOS}")
        print(f"Ensured output directory exists: {OUTPUT_DIR_FINGERPRINTS}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not create output directories: {e}")
        return

    # --- 1. Load Seed Channels (NEW DYNAMIC LOGIC) ---
    print(f"\n--- [PHASE 1/3] Loading Seed Channels ---")
    
    seed_df = None # Initialize
    
    if SEED_INPUT_SOURCE.lower() == "csv":
        print(f"Loading from local CSV: {SEED_CSV.name}")
        try:
            seed_df = pd.read_csv(SEED_CSV)
        except FileNotFoundError:
            print(f"❌ ERROR: '{SEED_CSV.name}' not found in root directory.")
            return
        except Exception as e:
            print(f"❌ ERROR: Could not read seed CSV: {e}")
            return
            
    elif SEED_INPUT_SOURCE.lower() == "sheet":
        print(f"Loading from Google Sheet...")
        seed_df = _load_from_google_sheet(SEED_GOOGLE_SHEET_URL)
        
    else:
        print(f"❌ ERROR: Invalid SEED_INPUT_SOURCE: '{SEED_INPUT_SOURCE}'")
        print("   Please set it to 'csv' or 'sheet' at the top of the script.")
        return

    # --- Validation for the loaded DataFrame ---
    if seed_df is None or seed_df.empty:
        print(f"❌ FAILED to load any seed channels. Exiting.")
        return
        
    if 'Channel_Name' not in seed_df.columns or 'Channel_URL' not in seed_df.columns:
        print(f"❌ ERROR: Loaded data must have 'Channel_Name' and 'Channel_URL' columns.")
        return
        
    print(f"✅ Loaded {len(seed_df)} seed channels.")
    # --- END OF NEW LOADING LOGIC ---


    # --- 2. Fetch Video Data (from youtube_utils) ---
    print(f"\n--- [PHASE 2/3] Fetching Videos from YouTube API ---")
    try:
        df_videos = fetch_for_seed_channels(
            seed_df, 
            limit_per_channel=4,  # You can adjust this
            filter_shorts=True
        )

        if df_videos is None or df_videos.empty:
            print("❌ No videos fetched. Check API key and channel URLs.")
            return

        # --- Save Video CSV ---
        df_videos.to_csv(FINAL_VIDEO_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\n✅ SUCCESS: Saved {len(df_videos)} videos to:")
        print(f"   {FINAL_VIDEO_CSV_PATH}")
        try:
            print(f"\n📦 Pushing Phase 1 videos for '{RUN_TAG}' to MongoDB...")
            save_dataframe_to_mongo(
                df_videos,
                collection_name=f"{MONGO_COLLECTION_PREFIX}",
                unique_key_column="video_id"  # unique per video
            )
            print(f"✅ Mongo: {len(df_videos)} videos upserted to '{MONGO_COLLECTION_PREFIX}'.")
        except Exception as e:
            print(f"❌ Mongo push failed for phase1 videos: {e}")


    except Exception as e:
        print(f"❌ ERROR during video fetching or saving: {e}")
        return

    # --- 3. Generate One-Shot Fingerprints (from fingerprint_llm_utils) ---
    print(f"\n--- [PHASE 3/3] Generating One-Shot Fingerprints ---")
    
    all_fingerprints = {
        "metadata": {
            "run_id": RUN_ID,
            "run_tag": RUN_TAG,
            "created_at": datetime.now().isoformat(),
            "model_provider": "gpt-4o", # You can hardcode this as per our last chat
            "video_data_source": str(FINAL_VIDEO_CSV_PATH.name)
        },
        "channels": {}
    }
    
    grouped_channels = df_videos.groupby('Channel_ID')
    total_channels = len(grouped_channels)
    
    for i, (channel_id, channel_video_df) in enumerate(grouped_channels, 1):
        
        channel_name = channel_video_df['Channel_Name'].iloc[0]
        channel_desc = channel_video_df['channel_description'].iloc[0]
        
        print(f"\n[{i}/{total_channels}] Processing: {channel_name} (ID: {channel_id})")
        
        try:
            # --- This is the ONE-SHOT function call ---
            fingerprint_data = get_channel_fingerprint_oneshot(
                channel_name=channel_name,
                channel_description=channel_desc,
                video_df=channel_video_df,
                model_provider="gpt-4o" # Using gpt-4o as default
            )

            if not fingerprint_data:
                 print(f"  ⚠️  WARNING: One-shot function returned empty data for {channel_name}.")
                 all_fingerprints["channels"][channel_id] = {
                    "channel_name": channel_name,
                    "status": "failed_empty_response",
                    "fingerprint": {}
                 }
                 continue

            all_fingerprints["channels"][channel_id] = {
                "channel_name": channel_name,
                "status": "success",
                "fingerprint": fingerprint_data 
            }
            print(f"  ✅ Success for {channel_name}.")
            
            if "profile" in fingerprint_data:
                print(f"     -> Profile: {fingerprint_data.get('profile')}")

        except Exception as e:
            print(f"  ❌ ERROR processing fingerprint for {channel_name}: {e}")
            all_fingerprints["channels"][channel_id] = {
                "channel_name": channel_name,
                "status": "error",
                "error_message": str(e),
                "fingerprint": {}
            }
        
        time.sleep(1) # Be kind to the LLM API

    # --- Save Final Fingerprint JSON ---
    print("\n--- Pipeline Complete. Saving final JSON. ---")
    try:
        with open(FINAL_FINGERPRINT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_fingerprints, f, indent=2, ensure_ascii=False)
        
        print(f"✅ SUCCESS: All fingerprints saved to:")
        print(f"   {FINAL_FINGERPRINT_JSON_PATH}")
        
    except Exception as e:
        print(f"❌ ERROR saving final JSON: {e}")

    try:
        print(f"\n📦 Pushing fingerprints to MongoDB for '{RUN_TAG}'...")
        save_json_blob(
            data=all_fingerprints,
            collection_name=f"{MONGO_COLLECTION_PREFIX}_fingerprints",
            unique_key="run_id",
            key_value=RUN_ID
        )
        print(f"✅ Mongo: Fingerprint blob saved to '{MONGO_COLLECTION_PREFIX}_fingerprints' (run_id={RUN_ID})")
    except Exception as e:
        print(f"❌ Mongo push failed for fingerprints: {e}")


if __name__ == "__main__":
    main()