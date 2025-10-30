import pandas as pd
import json
import re
import time
from pathlib import Path
import os
import hashlib
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# --- Import necessary functions from YOUR utils ---
# Import youtube utils for fetching videos and API Key
from utils.youtube_utils import fetch_recent_videos, API_KEY, _safe_get_json, YT_BASE # Import fetch_recent_videos
# Import LLM utils for niche extraction and helpers
from utils.fingerprint_llm_utils import extract_niche_llm, preprocess_text_for_llm, gemini_model, gpt_client # Import extract_niche_llm

# === Configuration ===
INPUT_CSV_PATH = Path("./manual_keyword_test_results.csv") # Output from manual_test.py
OUTPUT_CSV_PATH = Path("./manual_keyword_test_results_with_niches.csv")
SEED_CHANNEL_NAME = "Ali Abdaal" # For reference logging

VIDEOS_PER_CANDIDATE = 10 # Fetch 10 non-short videos per candidate
FILTER_SHORTS = True

# --- LLM API Settings ---
MAX_RETRIES = 3
RETRY_DELAY = 5

# --- Cache Setup ---
# Define BASE directory relative to this script file
BASE_DIR = Path(__file__).resolve().parent
MANUAL_OUTPUTS_DIR = BASE_DIR / "manual_outputs"
YT_VIDEO_CACHE_DIR = MANUAL_OUTPUTS_DIR / "cache_videos" # Cache for fetch_recent_videos
LLM_NICHE_CACHE_DIR = MANUAL_OUTPUTS_DIR / "cache_niche" # Cache for extract_niche_llm
YT_VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LLM_NICHE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === Caching Helper Functions (Defined within this script) ===
def _get_cache_key(*args) -> str:
    """Creates a simple hash key from function arguments."""
    s = str(args)
    return hashlib.md5(s.encode()).hexdigest()

def load_from_cache(cache_dir: Path, filename: str):
    """Loads data from a JSON cache file in a specific directory."""
    cache_file = cache_dir / f"{filename}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                print(f"   CACHE HIT: Loading {cache_dir.name}/{filename}.json")
                return json.load(f)
        except Exception as e:
            print(f"   ⚠️ Cache load error for {cache_dir.name}/{filename}.json: {e}")
    print(f"   CACHE MISS: {cache_dir.name}/{filename}.json")
    return None

def save_to_cache(cache_dir: Path, filename: str, data):
    """Saves data to a JSON cache file in a specific directory."""
    cache_file = cache_dir / f"{filename}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"   CACHE SAVE: Saved to {cache_dir.name}/{filename}.json")
    except Exception as e:
        print(f"   ⚠️ Cache save error for {cache_dir.name}/{filename}.json: {e}")

# === Cached Function Wrappers ===

# Wrapper for fetch_recent_videos using YT_VIDEO_CACHE_DIR
def fetch_recent_videos_cached(channel_id: str, max_results: int, filter_shorts: bool) -> Tuple[List[Dict], str]:
    cache_filename = f"fetch_recent_videos_{channel_id}_max{max_results}_filter{filter_shorts}"
    # Use the correct cache directory
    cached_data = load_from_cache(YT_VIDEO_CACHE_DIR, cache_filename)
    if cached_data is not None:
        if isinstance(cached_data, list) and len(cached_data) == 2:
             return cached_data[0], cached_data[1]
        else: print(f"   ⚠️ Video Cache format error, refetching...")

    # Call the original function imported from youtube_utils
    # Make sure youtube_utils.fetch_recent_videos uses the corrected logic!
    videos, description = fetch_recent_videos(channel_id, max_results, filter_shorts) #

    # Use the correct cache directory
    save_to_cache(YT_VIDEO_CACHE_DIR, cache_filename, [videos, description])
    return videos, description

# Wrapper for extract_niche_llm using LLM_NICHE_CACHE_DIR
def extract_niche_llm_cached(channel_id: str, channel_description: str, video_titles: List[str], channel_name: str) -> str:
    titles_hash = _get_cache_key(video_titles[:10])
    desc_hash = _get_cache_key(channel_description[:500])
    cache_filename = f"niche_{channel_id}_{desc_hash}_{titles_hash}"
    # Use the correct cache directory
    cached_data = load_from_cache(LLM_NICHE_CACHE_DIR, cache_filename)
    if cached_data is not None:
        # Assuming niche is stored as {"niche": "..."} or just string
        if isinstance(cached_data, dict): return cached_data.get("niche", "General")
        if isinstance(cached_data, str): return cached_data
        else: print(f"   ⚠️ Niche Cache format error, refetching...")

    # Call the original function imported from fingerprint_llm_utils
    # Make sure fingerprint_llm_utils.extract_niche_llm is the one you want
    niche = extract_niche_llm(channel_description, video_titles, channel_name, model_type="gpt") #

    # Use the correct cache directory
    save_to_cache(LLM_NICHE_CACHE_DIR, cache_filename, {"niche": niche}) # Save as dict
    return niche


# === Main Execution ===
def main():
    start_time = time.time()
    print("="*70)
    print("EXTRACTING NICHES FOR CANDIDATE CHANNELS")
    print(f"Input: {INPUT_CSV_PATH.name}")
    print("="*70)

    # --- Load Input ---
    if not INPUT_CSV_PATH.exists(): print(f"❌ Input CSV not found: {INPUT_CSV_PATH}"); return
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        required = ['id', 'name', 'description']
        if not all(col in df.columns for col in required):
             missing = [c for c in required if c not in df.columns]
             print(f"❌ Input CSV missing required columns: {missing}"); return
        df['description'] = df['description'].fillna('')
        print(f"✅ Loaded {len(df)} channels from {INPUT_CSV_PATH.name}")
    except Exception as e: print(f"❌ Error loading CSV: {e}"); return

    # --- Check API Key & LLM Clients ---
    if not API_KEY: print("❌ YouTube API Key missing!"); return
   
    # --- Add/Reset Niche Columns ---
    df['Discovered_Niche'] = "Pending"
    df['Video_Fetch_Status'] = "Pending"

    processed_count = 0
    # --- Process Each Channel ---
    for index, row in df.iterrows():
        # Ensure channel_id is treated as string, handle potential NaN/float issues from CSV read
        channel_id = str(row['id']).strip()
        if not channel_id or channel_id == 'nan':
             print(f"\nSkipping row {index+1} due to invalid Channel ID.")
             df.loc[index, 'Video_Fetch_Status'] = "Invalid ID"
             df.loc[index, 'Discovered_Niche'] = "Skipped"
             continue

        channel_name = str(row['name']).strip()
        print(f"\nProcessing {index + 1}/{len(df)}: {channel_name} ({channel_id})")

        # 1. Fetch Videos & Use Channel Desc (Cached)
        try:
            print("   Fetching videos...")
            videos, channel_desc_fetched = fetch_recent_videos_cached(
                channel_id=channel_id,
                max_results=VIDEOS_PER_CANDIDATE,
                filter_shorts=FILTER_SHORTS
            )
            df.loc[index, 'Video_Fetch_Status'] = f"Success ({len(videos)} videos)" if videos else "Success (0 videos)" # More accurate status

            # Use fetched description if available, otherwise fallback to description from input CSV
            # Ensure the description from CSV is also treated as string
            channel_desc_from_csv = str(row.get('description', '')).strip()
            channel_desc_to_use = channel_desc_fetched if channel_desc_fetched else channel_desc_from_csv

        except Exception as e:
            print(f"   ❌ Error fetching videos for {channel_name}: {e}")
            df.loc[index, 'Video_Fetch_Status'] = f"Error: {str(e)[:50]}"
            videos = []
            channel_desc_to_use = str(row.get('description', '')).strip() # Fallback

        # 2. Extract Niche (Cached)
        video_titles = [v.get('title', '') for v in videos] # Get titles from fetched videos
        try:
            print("   Extracting niche...")
            discovered_niche = extract_niche_llm_cached(
                channel_id=channel_id, # Pass ID for better caching
                channel_description=channel_desc_to_use,
                video_titles=video_titles,
                channel_name=channel_name
            )
            df.loc[index, 'Discovered_Niche'] = discovered_niche
        except Exception as e:
            print(f"   ❌ Error during niche extraction: {e}")
            df.loc[index, 'Discovered_Niche'] = "Error"

        processed_count += 1

        # Save progress periodically
        if processed_count % 10 == 0:
            try:
                # Make sure to save the entire df, not just the slice being processed
                df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
                print(f"--- Saved progress ({processed_count} processed) ---")
            except Exception as e: print(f"   ⚠️ Error saving intermediate progress: {e}")

        time.sleep(1) # Small delay

    # --- Final Save ---
    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\n✅ Successfully processed {processed_count} channels.")
        print(f"💾 Final results with niches saved to: {OUTPUT_CSV_PATH.name}")
    except Exception as e: print(f"❌ Error saving final CSV: {e}")

    end_time = time.time()
    print(f"⏱️ Total time: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()