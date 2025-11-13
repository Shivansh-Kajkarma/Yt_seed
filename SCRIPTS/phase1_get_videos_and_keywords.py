import pandas as pd
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import re
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from utils.youtube_utils import fetch_for_seed_channels, _load_from_google_sheet
    from utils.fingerprint_llm_utils import (
        get_channel_fingerprint_oneshot
    )
    from utils.mongo_utils import save_dataframe_to_mongo, save_json_blob
except ImportError as e: 
    print("Error: Could not import from 'utils' directory.")
    print(f"Ensure 'utils' is at this path: {BASE_DIR / 'utils'}")
    raise e 

def main(run_tag: str, seed_channel_name: str, seed_channel_url: str): 
    """
    Main pipeline script:
    1. Fetches video data for ONE seed channel from YouTube API.
    2. Saves video data to MongoDB.
    3. Generates one-shot fingerprint for that channel.
    4. Saves fingerprint to MongoDB.
    """
    MONGO_COLLECTION_PREFIX = f"{run_tag.upper()}_phase1"
    RUN_ID = f"{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


    print(f"--- 🚀 Starting New Pipeline Run ---")
    print(f"Run ID: {RUN_ID}")
    print(f"Run Tag: {run_tag}")
    print(f"Seed Channel: {seed_channel_name}")

    print(f"\n--- [PHASE 1/3] Preparing Seed Channel ---")
 
    seed_df = pd.DataFrame([
        {"Channel_Name": seed_channel_name, "Channel_URL": seed_channel_url}
    ])
        
    print(f"✅ Loaded 1 seed channel for processing: {seed_channel_name}")

    # --- 2. Fetch Video Data (from youtube_utils) ---
    print(f"\n--- [PHASE 2/3] Fetching Videos from YouTube API ---")
    try:
        df_videos = fetch_for_seed_channels(
            seed_df, 
            limit_per_channel=30,  # can adjust this
            filter_shorts=True,
            run_tag=run_tag 
        )

        if df_videos is None or df_videos.empty:
            print("❌ No videos fetched. Check API key and channel URLs.")
            return False # Stop this seed's run
        
        try:
            print(f"\n📦 Pushing Phase 1 videos for '{run_tag}' to MongoDB...")
            df_videos["run_tag"] = run_tag
            df_videos["run_id"] = RUN_ID
            
            save_dataframe_to_mongo(
                df_videos,
                collection_name=f"{MONGO_COLLECTION_PREFIX}",
                unique_key_column="video_id"
            )
            print(f"✅ Mongo: {len(df_videos)} videos upserted to '{MONGO_COLLECTION_PREFIX}'.")
        except Exception as e:
            print(f"❌ Mongo push failed for phase1 videos: {e}")
            pass


    except Exception as e:
        print(f"❌ ERROR during video fetching or saving: {e}")
        raise e

    # --- 3. Generate One-Shot Fingerprints (from fingerprint_llm_utils) ---
    print(f"\n--- [PHASE 3/3] Generating One-Shot Fingerprints ---")
    
    all_fingerprints = {
        "metadata": {
            "run_id": RUN_ID,
            "run_tag": run_tag,
            "created_at": datetime.now().isoformat(),
            "model_provider": "gpt-4o",
            "video_data_source": f"mongo_collection:{MONGO_COLLECTION_PREFIX}" 
        },
        "channels": {}
    }
    
    # This will now only group the videos for the single seed channel
    grouped_channels = df_videos.groupby('Channel_ID')
    total_channels = len(grouped_channels) # Should be 1
    
    for i, (channel_id, channel_video_df) in enumerate(grouped_channels, 1):
        
        channel_name = channel_video_df['Channel_Name'].iloc[0]
        channel_desc = channel_video_df['channel_description'].iloc[0]
        
        print(f"\n[{i}/{total_channels}] Processing: {channel_name} (ID: {channel_id})")
        
        try:
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

            all_fingerprints["channels"][channel_id] = {
                "channel_name": channel_name,
                "status": "success",
                "fingerprint": fingerprint_data 
            }
            print(f"  ✅ Success for {channel_name}.")
            
            if "profile" in fingerprint_data:
                # Just print a snippet
                profile_snippet = str(fingerprint_data.get('profile', {})).split(',')[0]
                print(f"     -> Profile: {profile_snippet}...")


        except Exception as e:
            print(f"  ❌ ERROR processing fingerprint for {channel_name}: {e}")
            all_fingerprints["channels"][channel_id] = {
                "channel_name": channel_name,
                "status": "error",
                "error_message": str(e),
                "fingerprint": {}
            }
        
        time.sleep(2) 

    try:
        print(f"\n📦 Pushing fingerprints to MongoDB for '{run_tag}'...")
        save_json_blob(
            data=all_fingerprints,
            collection_name=f"{MONGO_COLLECTION_PREFIX}_fingerprints",
            unique_key="run_id", 
            key_value=RUN_ID
        )
        print(f"✅ Mongo: Fingerprint blob saved to '{MONGO_COLLECTION_PREFIX}_fingerprints' (run_id={RUN_ID})")
    except Exception as e:
        print(f"❌ Mongo push failed for fingerprints: {e}")
        raise e # This is a critical failure, stop the run

    print(f"\n✅ Phase 1 complete for {run_tag}.")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python phase1_get_videos_and_keywords.py <run_tag> <channel_name> <channel_url>")
        print("Example: python ...py moon Moon 'https://www.youtube.com/@moon-real'")
        sys.exit(1) 
    
    tag = sys.argv[1]
    name = sys.argv[2]
    url = sys.argv[3]
    
    print(f"Running in __main__ test mode for {name}...")
    main(tag, name, url)