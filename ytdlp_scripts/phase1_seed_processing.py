import sys
import os
import json
import time
import random
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- 1. SETUP PATHS ---
# Add project root to sys.path to allow importing from 'utils'
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import Utils
try:
    from utils.youtube_utils import _load_from_google_sheet, fetch_recent_videos, extract_channel_id
    from utils.yt_dlp_utils import fetch_video_data_ytdlp
    from utils.mongo_utils import save_dataframe_to_mongo, save_json_blob
    from utils.fingerprint_llm_utils import get_channel_fingerprint_oneshot
except ImportError as e:
    print("❌ Error: Could not import from 'utils'. Run this script from the project root.")
    print("   Example: python ytdlp_scripts/phase1_seed_processing.py <args>")
    raise e

# Config
OUTPUT_DIR = os.path.join(BASE_DIR, "ytdlp_scripts", "output", "phase1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_single_seed(run_tag, channel_name, channel_url, max_videos=10):
    """
    1. Fetch basic list via API (Fast, filter shorts).
    2. Enrich details via yt-dlp (Stealth, Transcripts).
    3. Save to Disk & Mongo.
    4. Generate Fingerprint.
    """
    print(f"\nStarted processing seed: {channel_name}...")

    # --- STEP 1: Resolve ID & Basic API Fetch ---
    channel_id = extract_channel_id(channel_url)
    if not channel_id:
        print(f"❌ Could not resolve Channel ID for {channel_url}")
        return False

    print(f"   ID: {channel_id} | Fetching recent {max_videos} videos via API...")
    
    # We use the API to get the list quickly and filter out Shorts immediately
    # This saves us from running yt-dlp on irrelevant content.
    api_videos, channel_desc = fetch_recent_videos(
        channel_id, 
        max_results=max_videos, 
        filter_shorts=True, # Important: Filter shorts here
        run_tag=run_tag,
        seed_name=channel_name
    )

    if not api_videos:
        print("❌ No long-form videos found via API.")
        return False

    print(f"   ✅ API found {len(api_videos)} long-form videos. Starting Deep Scan...")

    # --- STEP 2: Deep Enrich with yt-dlp ---
    enriched_videos = []
    
    for i, vid in enumerate(api_videos):
        vid_id = vid['video_id']
        print(f"      [{i+1}/{len(api_videos)}] Deep scanning: {vid['title'][:30]}...")

        # CALL YOUR NEW UTILITY
        deep_data = fetch_video_data_ytdlp(vid_id)

        if deep_data:
            # Merge API data (reliable published_at) with Deep Data (Transcripts/Chapters)
            # deep_data keys overwrite api_videos keys if duplicates exist
            merged_data = {**vid, **deep_data}
            
            # Ensure we have the Channel info in every row
            merged_data['Channel_Name'] = channel_name
            merged_data['Channel_ID'] = channel_id
            merged_data['channel_description'] = channel_desc
            merged_data['run_tag'] = run_tag
            
            enriched_videos.append(merged_data)
            print(f"         -> Success. Transcript len: {len(deep_data.get('caption_tracks', '') or '')}")
        else:
            print("         -> Failed to fetch deep data. Skipping.")

        # RATE LIMITING (Crucial for Stealth)
        time.sleep(random.uniform(2.0, 5.0))

    if not enriched_videos:
        print("❌ All deep scans failed.")
        return False

    # --- STEP 3: Save Output ---
    df_enriched = pd.DataFrame(enriched_videos)
    
    # 3a. Save Local JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = channel_name.replace(" ", "_").lower()
    filename = f"{run_tag}_{safe_name}_deep_data.json"
    local_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(local_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_videos, f, indent=2, default=str)
    print(f"\n   💾 Saved local backup: {local_path}")

    # 3b. Save to MongoDB (Phase 1 Collection)
    try:
        save_dataframe_to_mongo(
            df_enriched, 
            f"{run_tag.upper()}_phase1", 
            unique_key_column="video_id"
        )
        print(f"   💾 Saved {len(df_enriched)} records to Mongo.")
    except Exception as e:
        print(f"   ⚠️ Mongo Save Error: {e}")

    # --- STEP 4: Generate Fingerprint (With Transcripts!) ---
    print("\n   🧠 Generating Content Fingerprint (using Transcripts)...")
    
    # Create a "Virtual DataFrame" for the LLM that prioritizes the transcript
    # The LLM utils look for 'description', so we append transcript to description 
    # to give it full context without rewriting the LLM function.
    df_for_llm = df_enriched.copy()
    
    def combine_text(row):
        desc = row.get('description', '')
        transcript = row.get('caption_tracks', '')
        chapters = row.get('chapters', [])
        
        chapter_text = "\n".join([c.get('title', '') for c in chapters]) if chapters else ""
        
        # Feed the LLM the "Super Context"
        return f"DESCRIPTION:\n{desc}\n\nCHAPTERS:\n{chapter_text}\n\nTRANSCRIPT SAMPLE:\n{transcript[:8000]}" # Limit to avoid token overflow

    df_for_llm['description'] = df_for_llm.apply(combine_text, axis=1)

    fingerprint = get_channel_fingerprint_oneshot(
        channel_name=channel_name,
        channel_description=channel_desc,
        video_df=df_for_llm, # Passing the transcript-enriched DF
        model_provider="gpt-4o"
    )

    # Save Fingerprint to Mongo
    fp_wrapper = {
        "metadata": {
            "run_tag": run_tag,
            "created_at": datetime.now().isoformat(),
            "data_source": "ytdlp_stealth"
        },
        "channels": {
            channel_id: {
                "channel_name": channel_name,
                "fingerprint": "fingerprint"
            }
        }
    }
    save_json_blob(fp_wrapper, f"{run_tag.upper()}_phase1_fingerprints", "run_tag", run_tag)
    
    print("   ✅ Fingerprint generated and saved.")
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python ytdlp_scripts/phase1_seed_processing.py <run_tag> <sheet_url_or_csv>")
        sys.exit(1)

    run_tag = sys.argv[1]
    input_source = sys.argv[2]

    print(f"🚀 STARTING PHASE 1 (Stealth Mode) | Tag: {run_tag}")
    
    # Load Inputs
    df_seeds = _load_from_google_sheet(input_source)
    if df_seeds is None or df_seeds.empty:
        print("❌ No seeds found in input.")
        return

    # Process Seeds
    for _, row in df_seeds.iterrows():
        name = row['Channel_Name']
        url = row['Channel_URL']
        
        try:
            process_single_seed(run_tag, name, url)
        except Exception as e:
            print(f"❌ Critical Error processing {name}: {e}")
            continue

    print("\n🏁 Phase 1 Complete.")

if __name__ == "__main__":
    main()


