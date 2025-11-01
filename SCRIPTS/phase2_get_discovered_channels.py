import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime
import csv
import sys
import os

# --- Make sure utils are importable ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    # --- Import ONLY YouTube Helper Functions ---
    from utils.youtube_utils import (
        search_videos_multi_focused,
        get_channel_metadata_batch,
        fetch_recent_videos,
    )
except ImportError:
    print("Error: Could not import from 'utils' directory.")
    print(f"Ensure 'utils' is at this path: {BASE_DIR / 'utils'}")
    sys.exit(1)


# === CONFIGURATION ===
# --- EDIT THIS TAG ---
# Set a memorable name for this run (e.g., "moon", "vox_analysis")
RUN_TAG = "moon"
#
# ==================================================

# --- Input Files (from Phase 1) ---
# We need to find the LATEST Phase 1 files to use as our input
try:
    FINGERPRINTS_DIR = BASE_DIR / "FINGERPRINTS_ONESHOT"
    SEED_VIDEOS_DIR = BASE_DIR / "PHASE_1_OUTPUTS"
    
    # Find the most recent fingerprint file for our run_tag
    latest_fingerprint_file = max(
        FINGERPRINTS_DIR.glob(f"fingerprints_oneshot_{RUN_TAG}*.json"), 
        key=os.path.getctime
    )
    # Find the most recent video file for our run_tag
    latest_seed_video_file = max(
        SEED_VIDEOS_DIR.glob(f"sample_videos_{RUN_TAG}*.csv"), 
        key=os.path.getctime
    )
    
    print(f"Using Seed Fingerprints: {latest_fingerprint_file.name}")
    print(f"Using Seed Video Data: {latest_seed_video_file.name}")

except Exception as e:
    print(f"❌ ERROR: Could not find Phase 1 output files for RUN_TAG='{RUN_TAG}'.")
    print(f"Make sure 'FINGERPRINTS_ONESHOT/' and 'PHASE_1_OUTPUTS/' contain files for this tag.")
    print(f"Error details: {e}")
    sys.exit(1)


# --- Output Files (for Phase 2) ---
OUTPUT_CACHE_DIR = BASE_DIR / "PHASE_2_DISCOVERY_CACHE"
OUTPUT_CACHE_DIR.mkdir(exist_ok=True) # Ensure directory exists

# The cache file that this script WRITES TO
CACHE_DATA_PATH = OUTPUT_CACHE_DIR / f"phase2_discovered_raw_data_{RUN_TAG}.csv"

# The high-level log file this script UPDATES
SEEN_CHANNELS_PATH = BASE_DIR / f"seen_channels_{RUN_TAG}.csv"


# --- Define ALL columns for the new cache file ---
CACHE_COLUMN_ORDER = [
    "Seed_Channel_Name", "Seed_Channel_ID",
    "Discovered_Channel_Name", "Discovered_Channel_ID", "Discovered_Channel_URL",
    "Discovered_Subs", "Discovered_Video_Count", "Discovered_Country",
    "Discovered_Channel_Description", "Discovered_Videos_JSON", # <-- CRITICAL: Raw data for LLM
    "Discovery_Level", "Timestamp",
]

AUTO_KEEP_COUNTRIES = [
    'US', 'GB', 'CA', 'AU', 'NZ', 'NG', 'Unknown'
]

# --- Settings ---
SEED_CHANNELS = ["Moon"] # Which seeds from the fingerprint file to use
MIN_SUBSCRIBERS = 10000
MIN_VIDEOS = 6
VIDEOS_PER_CANDIDATE = 20 # How many videos to fetch for LLM analysis

# --- Rate Limiting ---
DELAY_BETWEEN_CANDIDATES = 2 # Shorter delay, no LLM call
DELAY_BETWEEN_SEEDS = 10


# === HELPER FUNCTIONS ===

def load_cached_channels(file_path):
    """Loads the Phase 2 cache and returns a set of processed channel IDs."""
    if not file_path.exists():
        return set()
    try:
        df = pd.read_csv(file_path, usecols=["Discovered_Channel_ID"])
        return set(df["Discovered_Channel_ID"].astype(str).tolist())
    except Exception as e:
        print(f"  ⚠️  Could not read cache file {file_path.name}: {e}")
        return set()

def append_to_cache_csv(data_dict, file_path):
    """Appends a single row (dict) to the Phase 2 cache CSV file."""
    file_exists = file_path.exists()
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_COLUMN_ORDER, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)
    except Exception as e:
        print(f"  ❌ ERROR appending to {file_path.name}: {e}")

def save_seen_channels(seen_data_list, file_path):
    """Saves seen channels to CSV with deduplication"""
    if not seen_data_list: return
    try:
        df = pd.DataFrame(seen_data_list)
        all_cols = ["Channel_ID", "Channel_Name", "Date_Added", "Processing_Status"]
        for col in all_cols:
            if col not in df.columns:
                df[col] = None
        df = df[all_cols]
        df = df.sort_values(by="Date_Added").drop_duplicates(subset=["Channel_ID"], keep="last")
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"  ❌ Error saving {file_path.name}: {e}")

def load_seen_channels(file_path):
    """Loads seen channels from CSV"""
    seen_data = []
    seen_ids = set()
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            required_cols = ["Channel_ID", "Channel_Name", "Date_Added", "Processing_Status"]
            if all(col in df.columns for col in required_cols):
                seen_data = df.to_dict("records")
                seen_ids = set(df["Channel_ID"].astype(str).tolist())
                print(f"  📂 Loaded {len(seen_ids)} previously seen channels (from log)")
            else:
                print("  ⚠️  Seen channels CSV has wrong columns. Starting fresh log.")
        except Exception as e:
            print(f"  ⚠️  Could not read seen channels: {e}")
    else:
        print("  📂 No existing seen channels file. Starting fresh log.")
    return seen_data, seen_ids

def _update_status(seen_data, channel_id, status):
    """Update processing status for a channel in the high-level log"""
    for entry in seen_data:
        if entry["Channel_ID"] == channel_id:
            entry["Processing_Status"] = status
            break

# --- Main processing function ---
def process_seed_channel(
    seed_channel,
    seed_channel_id,
    seed_keywords,  # This is a DICTIONARY
    seen_channels_data,
    seen_ids,
    cached_ids,     # Set of already cached IDs
    cache_file_path # Path to save data
):
    """
    Process a single seed channel:
    1. Search YouTube using seed keywords.
    2. Filter out seen/cached channels.
    3. Get metadata for new channels.
    4. Filter by subs/videos/country.
    5. Fetch video data for qualified channels.
    6. Save all raw data to the Phase 2 cache file.
    Returns: (number_of_new_channels_cached)
    """

    print("\n" + "=" * 70)
    print(f"PROCESSING SEED: {seed_channel}")
    print(f"  (Will skip {len(cached_ids)} channels already in {cache_file_path.name})")
    print("=" * 70)

    current_time = datetime.now().isoformat()
    new_channels_cached_count = 0

    # --- Flatten seed keywords for searching ---
    seed_keywords_list = []
    if isinstance(seed_keywords, dict):
        for k, v in seed_keywords.items():
            if isinstance(v, list): seed_keywords_list.extend(v)
    else:
        print(f"❌ Seed keywords for {seed_channel} are not a dict. Skipping seed.")
        return 0
    
    if not seed_keywords_list:
        print(f"❌ No seed keywords found for {seed_channel}. Skipping seed.")
        return 0

    # --- STEP 1: Multi-Focused Search ---
    print(f"\n🔍 STEP 1: Searching YouTube with {len(seed_keywords_list)} keywords...")
    try:
        candidate_ids = search_videos_multi_focused(
            seed_keywords_list,
            max_results_per_search=30,
            max_keywords=len(seed_keywords_list)
        )
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return 0

    candidate_ids = candidate_ids - seen_ids - {seed_channel_id}
    if not candidate_ids:
        print("  ⚠️  No new candidates found after filtering seen log")
        return 0
    print(f"  ✅ {len(candidate_ids)} new candidates to evaluate")

    # --- STEP 2: Get Metadata ---
    print(f"\n📊 STEP 2: Fetching channel metadata...")
    metadata = get_channel_metadata_batch(list(candidate_ids))
    for meta in metadata:
        if meta["id"] not in seen_ids:
            seen_channels_data.append({
                "Channel_ID": meta["id"], "Channel_Name": meta["name"],
                "Date_Added": current_time, "Processing_Status": "discovered",
            })
            seen_ids.add(meta["id"])
    save_seen_channels(seen_channels_data, SEEN_CHANNELS_PATH) # Save new discoveries to log

    # --- STEP 3: Pre-Filter ---
    print(f"\n🔍 STEP 3: Pre-filtering by subscribers and videos...")
    qualified = []
    for meta in metadata:
        if meta["subscribers"] == -1:
            print(f"  - Filtering {meta['name']} (subscribers hidden)")
            _update_status(seen_channels_data, meta["id"], "filtered_hidden_subs")
            continue
        if meta["subscribers"] < MIN_SUBSCRIBERS:
            print(f"  - Filtering {meta['name']} (subs: {meta['subscribers']:,})")
            _update_status(seen_channels_data, meta["id"], "filtered_subs")
            continue
        if meta["video_count"] < MIN_VIDEOS:
            print(f"  - Filtering {meta['name']} (videos: {meta['video_count']})")
            _update_status(seen_channels_data, meta["id"], "filtered_videos")
            continue
        country = meta.get('country', 'Unknown')
        if country not in AUTO_KEEP_COUNTRIES:
            print(f"  - Filtering {meta['name']} (Country: {country})")
            _update_status(seen_channels_data, meta["id"], f"filtered_country_{country}")
            continue
        qualified.append(meta)
    save_seen_channels(seen_channels_data, SEEN_CHANNELS_PATH) # Save filter status to log

    # --- STEP 4: Filter against *already cached* channels ---
    print(f"\n🔍 STEP 4: Filtering against {len(cached_ids)} already cached channels...")
    candidates_to_process = []
    for c in qualified:
        if c['id'] not in cached_ids:
            candidates_to_process.append(c)
            
    if not candidates_to_process:
        print("  ✅ No new qualified candidates to cache for this seed.")
        return 0
    print(f"  🎯 {len(candidates_to_process)} new candidates to fetch and cache (out of {len(qualified)} qualified).")

    # --- STEP 5: Fetch Video Data & Cache (Save-as-you-go) ---
    print(f"\n🎯 STEP 5: Fetching and Caching {len(candidates_to_process)} candidates...")

    for i, candidate in enumerate(candidates_to_process, 1):
        print(f"\n  [{i}/{len(candidates_to_process)}] {candidate['name']}")
        print(f"     Subs: {candidate['subscribers']:,} | Videos: {candidate['video_count']:,}")

        try:
            # --- 5a. Fetch recent videos and channel description ---
            print(f"     Fetching {VIDEOS_PER_CANDIDATE} videos...")
            videos, cand_desc = fetch_recent_videos(
                candidate["id"], max_results=VIDEOS_PER_CANDIDATE, filter_shorts=True
            )
            
            if len(videos) < 3:
                print(f"     ⚠️  Only {len(videos)} videos found, logging and skipping")
                _update_status(seen_channels_data, candidate["id"], "skipped_few_videos")
                continue

            # --- 5b. Build the "Raw Data" Dictionary ---
            candidate_raw_data = {
                "Seed_Channel_Name": seed_channel,
                "Seed_Channel_ID": seed_channel_id,
                "Discovered_Channel_Name": candidate["name"],
                "Discovered_Channel_ID": candidate["id"],
                "Discovered_Channel_URL": candidate["url"],
                "Discovered_Subs": candidate["subscribers"],
                "Discovered_Video_Count": candidate["video_count"],
                "Discovered_Country": candidate.get("country", "Unknown"),
                "Discovered_Channel_Description": cand_desc,
                "Discovered_Videos_JSON": json.dumps(videos), # <-- Save video list as JSON string
                "Discovery_Level": 1,
                "Timestamp": datetime.now().isoformat(),
            }

            # --- 5c. SAVE TO DISK (This is the cache!) ---
            append_to_cache_csv(candidate_raw_data, cache_file_path)
            new_channels_cached_count += 1
            print(f"     ✅ Cached raw data for {candidate['name']} to {cache_file_path.name}")

            # --- 5d. Update high-level log ---
            _update_status(seen_channels_data, candidate["id"], "cached_for_llm")
            save_seen_channels(seen_channels_data, SEEN_CHANNELS_PATH)
            
            time.sleep(DELAY_BETWEEN_CANDIDATES)

        except Exception as e:
            print(f"     ❌ Error on {candidate['name']}: {str(e)[:100]}")
            _update_status(seen_channels_data, candidate["id"], "error_caching")

    return new_channels_cached_count


# === MAIN FUNCTION ===
def main():
    start_time = time.time()
    print("=" * 70)
    print(f"CHANNEL DISCOVERY PIPELINE (PHASE 2)")
    print(f"Run Tag: {RUN_TAG}")
    print(f"Seeds: {', '.join(SEED_CHANNELS)}")
    print(f"Output Cache: {CACHE_DATA_PATH.name}")
    print("=" * 70)

    # --- 1. Load Seed Keywords (from Phase 1 JSON) ---
    print(f"\n📖 Loading seed keywords from {latest_fingerprint_file.name}...")
    seed_keywords_map = {}
    try:
        with open(latest_fingerprint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Find the keywords for the channels we care about
            for channel_id, details in data.get("channels", {}).items():
                channel_name = details.get("channel_name")
                if channel_name in SEED_CHANNELS:
                    keywords_data = details.get("fingerprint", {}).get("keywords", {})
                    if keywords_data:
                        seed_keywords_map[channel_name] = keywords_data
                        print(f"  📌 Found keywords for seed: {channel_name}")
                    else:
                        print(f"  ⚠️  No keywords found in fingerprint for {channel_name}")

        if not seed_keywords_map:
            print(f"❌ ERROR: Could not find keywords for any seeds in {SEED_CHANNELS}.")
            return

    except Exception as e:
        print(f"❌ ERROR loading keywords: {e}")
        return

    # --- 2. Load Seed Channel ID Mapping (from Phase 1 CSV) ---
    print(f"\n🗺️  Loading channel IDs from {latest_seed_video_file.name}...")
    try:
        df_videos = pd.read_csv(latest_seed_video_file)
        seed_id_map = (
            df_videos.drop_duplicates(subset=["Channel_Name"])[
                ["Channel_Name", "Channel_ID"]
            ]
            .set_index("Channel_Name")["Channel_ID"]
            .to_dict()
        )
        print(f"  ✅ Loaded {len(seed_id_map)} channel mappings")
    except Exception as e:
        print(f"❌ ERROR loading channel IDs: {e}")
        return

    # --- 3. Load Seen Channels (High-level log) ---
    print(f"\n📂 Loading seen channels log from {SEEN_CHANNELS_PATH.name}...")
    seen_channels_data, seen_ids = load_seen_channels(SEEN_CHANNELS_PATH)
    current_time = datetime.now().isoformat()
    for seed_channel in SEED_CHANNELS:
        seed_channel_id = seed_id_map.get(seed_channel)
        if seed_channel_id and seed_channel_id not in seen_ids:
            seen_channels_data.append({
                "Channel_ID": seed_channel_id, "Channel_Name": seed_channel,
                "Date_Added": current_time, "Processing_Status": "seed",
            })
            seen_ids.add(seed_channel_id)
            print(f"  ✅ Added seed '{seed_channel}' to seen list")
    save_seen_channels(seen_channels_data, SEEN_CHANNELS_PATH)

    # --- 4. Load ALREADY CACHED channels (from Phase 2 cache) ---
    print(f"\n🔄 Loading already cached channels from {CACHE_DATA_PATH.name}...")
    cached_channel_ids = load_cached_channels(CACHE_DATA_PATH)
    print(f"  ✅ Found {len(cached_channel_ids)} channels in cache to skip.")

    # --- 5. Process Each Seed Channel ---
    total_new_channels_cached = 0
    for seed_idx, seed_channel in enumerate(SEED_CHANNELS, 1):
        print(f"\n{'=' * 70}")
        print(f"SEED {seed_idx}/{len(SEED_CHANNELS)}: {seed_channel}")
        print(f"{'=' * 70}")

        # Get all required seed info
        seed_keywords = seed_keywords_map.get(seed_channel)
        seed_channel_id = seed_id_map.get(seed_channel)
        
        if not seed_keywords or not seed_channel_id:
            print(f"⚠️  WARNING: Missing data for '{seed_channel}', skipping")
            continue

        new_channels_this_seed = process_seed_channel(
            seed_channel,
            seed_channel_id,
            seed_keywords, # Pass the DICT
            seen_channels_data,
            seen_ids,
            cached_channel_ids,  # <-- PASS THE SET
            CACHE_DATA_PATH      # <-- PASS THE PATH
        )

        total_new_channels_cached += new_channels_this_seed
        print(f"\n✅ Seed '{seed_channel}' complete:")
        print(f"   Cached {new_channels_this_seed} new channels this run.")

        if seed_idx < len(SEED_CHANNELS):
            print(f"\n⏸️  Waiting {DELAY_BETWEEN_SEEDS}s before next seed...")
            time.sleep(DELAY_BETWEEN_SEEDS)

    # === FINAL SUMMARY ===
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 2 DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"Seeds processed: {len(SEED_CHANNELS)}")
    print(f"Total new channels cached this run: {total_new_channels_cached}")
    print(f"Total channels in cache: {len(cached_channel_ids) + total_new_channels_cached}")
    print(f"⏱️  Total runtime: {elapsed / 60:.1f} minutes")
    print(f"\n✅ Next step: Run 'phase3_llm_scoring.py' (or similar) to process:")
    print(f"   {CACHE_DATA_PATH.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()