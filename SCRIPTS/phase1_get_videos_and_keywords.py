import pandas as pd
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# --- Make sure utils are importable ---
# This assumes your 'utils' folder is in the parent directory of 'SCRIPTS'
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from utils.youtube_utils import fetch_for_seed_channels
    from utils.fingerprint_llm_utils import (
        get_channel_fingerprint_oneshot # <-- This is 'gpt' in your util file
    )
except ImportError:
    print("Error: Could not import from 'utils' directory.")
    print(f"Ensure 'utils' is at this path: {BASE_DIR / 'utils'}")
    sys.exit(1)

# ==================================================
# 1. CONFIGURATION
# ==================================================
#
# --- EDIT THIS TAG ---
# Set a memorable name for this run (e.g., "moon", "vox_analysis", "test_run")
# This will be part of the output filenames.
RUN_TAG = "moon"
#
# ==================================================

# --- Input/Output Directories ---
SEED_CSV = BASE_DIR / "seed_channels.csv"
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

    # --- 1. Load Seed Channels ---
    print(f"\n--- [PHASE 1/3] Loading Seed Channels ---")
    try:
        seed_df = pd.read_csv(SEED_CSV)
        if 'Channel_Name' not in seed_df.columns or 'Channel_URL' not in seed_df.columns:
            print(f"❌ ERROR: '{SEED_CSV}' must have 'Channel_Name' and 'Channel_URL' columns.")
            return
        print(f"✅ Loaded {len(seed_df)} seed channels from '{SEED_CSV.name}'.")
    except FileNotFoundError:
        print(f"❌ ERROR: '{SEED_CSV.name}' not found in root directory.")
        return
    except Exception as e:
        print(f"❌ ERROR: Could not read seed CSV: {e}")
        return

    # --- 2. Fetch Video Data (from youtube_utils) ---
    print(f"\n--- [PHASE 2/3] Fetching Videos from YouTube API ---")
    try:
        df_videos = fetch_for_seed_channels(
            seed_df, 
            limit_per_channel=30,  # You can adjust this
            filter_shorts=True
        )

        if df_videos is None or df_videos.empty:
            print("❌ No videos fetched. Check API key and channel URLs.")
            return

        # --- Save Video CSV ---
        df_videos.to_csv(FINAL_VIDEO_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\n✅ SUCCESS: Saved {len(df_videos)} videos to:")
        print(f"   {FINAL_VIDEO_CSV_PATH}")

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
            "model_provider": "gpt",
            "video_data_source": str(FINAL_VIDEO_CSV_PATH.name)
        },
        "channels": {}
    }
    
    # Group by Channel_ID to process each channel
    # Grouping by ID is safer than grouping by name
    grouped_channels = df_videos.groupby('Channel_ID')
    total_channels = len(grouped_channels)
    
    for i, (channel_id, channel_video_df) in enumerate(grouped_channels, 1):
        
        # Get consistent channel info from the first row
        channel_name = channel_video_df['Channel_Name'].iloc[0]
        channel_desc = channel_video_df['channel_description'].iloc[0]
        
        print(f"\n[{i}/{total_channels}] Processing: {channel_name} (ID: {channel_id})")
        
        try:
            # --- This is the ONE-SHOT function call ---
            fingerprint_data = get_channel_fingerprint_oneshot(
                channel_name=channel_name,
                channel_description=channel_desc,
                video_df=channel_video_df,
                model_provider="gpt-4o"
            )

            if not fingerprint_data:
                 print(f"  ⚠️  WARNING: One-shot function returned empty data for {channel_name}.")
                 all_fingerprints["channels"][channel_id] = {
                    "channel_name": channel_name,
                    "status": "failed_empty_response",
                    "fingerprint": {}
                 }
                 continue

            # Store the full JSON (profile + keywords)
            all_fingerprints["channels"][channel_id] = {
                "channel_name": channel_name,
                "status": "success",
                "fingerprint": fingerprint_data 
            }
            print(f"  ✅ Success for {channel_name}.")
            
            # Optional: Pretty print the profile for logging
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
        
        # Add a small delay to be kind to the LLM API
        time.sleep(1) 

    # --- Save Final Fingerprint JSON ---
    print("\n--- Pipeline Complete. Saving final JSON. ---")
    try:
        with open(FINAL_FINGERPRINT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_fingerprints, f, indent=2, ensure_ascii=False)
        
        print(f"✅ SUCCESS: All fingerprints saved to:")
        print(f"   {FINAL_FINGERPRINT_JSON_PATH}")
        
    except Exception as e:
        print(f"❌ ERROR saving final JSON: {e}")


if __name__ == "__main__":
    main()