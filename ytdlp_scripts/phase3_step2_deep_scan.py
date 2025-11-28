import sys
import os
import json
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo
    from utils.yt_dlp_utils import fetch_video_data_ytdlp
except ImportError as e:
    print("❌ Error importing utils.")
    raise e

# CONFIG
VIDEOS_TO_SCAN = 5  # We scan 3 to get a representative sample
# REMOVED: MIN_PODCAST_DURATION = 600 (Client says duration doesn't matter)

def main():
    if len(sys.argv) < 2:
        print("Usage: python ytdlp_scripts/phase3_step2_deep_scan.py <run_tag>")
        sys.exit(1)
        
    run_tag = sys.argv[1]
    print(f"🚀 PHASE 3 STEP B: Deep Scan (yt-dlp) | Tag: {run_tag}")
    print("ℹ️  Constraint Update: Duration Filter REMOVED (Format > Duration).")
    
    # 1. Load Step 1 Survivors
    collection_in = f"{run_tag.upper()}_phase3_step1"
    try:
        df_candidates = load_collection_as_df(collection_in)
        if df_candidates.empty:
            print("❌ No candidates found in Step 1.")
            return
        print(f"📥 Loaded {len(df_candidates)} candidates to scan.")
    except Exception as e:
        print(f"❌ Error loading candidates: {e}")
        return

    scanned_results = []
    
    for index, row in df_candidates.iterrows():
        channel_name = row.get('Discovered_Channel_Name')
        
        # Parse videos from Phase 2 (API data) to get IDs
        videos_json = row.get('Discovered_Videos_JSON', '[]')
        try:
            video_list = json.loads(videos_json)
        except:
            video_list = []
            
        if not video_list:
            print(f"⚠️ Skipping {channel_name} (No video IDs)")
            continue
            
        print(f"\n[{index+1}/{len(df_candidates)}] Deep Scanning: {channel_name}")
        
        # 3. Stealth Scan (Top 3)
        # We need this deep data to prove the "100% Format Match" in the next step
        target_videos = video_list[:VIDEOS_TO_SCAN]
        deep_data_list = []
        
        for vid in target_videos:
            vid_id = vid.get('video_id')
            
            # Call yt-dlp utils (Stealth Mode)
            data = fetch_video_data_ytdlp(vid_id)
            
            if data:
                deep_data_list.append(data)
                print(f"    ✅ Fetched: {data['title'][:30]}... (Has Trans: {len(data.get('caption_tracks', '') or '') > 0})")
            else:
                print(f"    ❌ Failed to fetch: {vid_id}")
            
            # Sleep to stay safe
            time.sleep(random.uniform(4.0, 6.0))

        if not deep_data_list:
            print(f"    ⚠️ All scans failed for {channel_name}. Skipping.")
            continue

        # 4. METRICS (For Context only, not Filtering)
        durations = [d.get('duration', 0) for d in deep_data_list]
        avg_duration = np.mean(durations) if durations else 0
        has_chapters_count = sum(1 for d in deep_data_list if d.get('has_chapters'))
        
        # 5. STATUS UPDATE
        # We pass EVERYONE who successfully scanned. 
        # Step C (LLM) will decide if the *Content* matches the Format.
        print(f"    ✅ DATA ACQUIRED. Avg Duration: {int(avg_duration)}s | Chapters: {has_chapters_count}/{len(deep_data_list)}")

        # 6. Prepare Output
        row_dict = row.to_dict()
        row_dict['Deep_Scan_Data'] = json.dumps(deep_data_list) # Save massive JSON string
        row_dict['step2_avg_duration'] = avg_duration
        row_dict['step2_has_chapters'] = has_chapters_count > 0
        row_dict['step2_status'] = "scanned"
        
        scanned_results.append(row_dict)
        
        # Save-as-you-go
        save_dataframe_to_mongo(
            pd.DataFrame([row_dict]), 
            f"{run_tag.upper()}_phase3_step2", 
            "Discovered_Channel_ID"
        )

    print("\n🏁 Phase 3 Step B Complete.")

if __name__ == "__main__":
    main()