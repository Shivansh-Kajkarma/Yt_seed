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
    from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo

    print("✅ Successfully imported YouTube utils.")
except ImportError:
    print("Error: Could not import from 'utils' directory.")
    print(f"Ensure 'utils' is at this path: {BASE_DIR / 'utils'}")
    sys.exit(1)


# === CONFIGURATION ===
# --- EDIT THIS TAG ---
# Set a memorable name for this run (e.g., "moon", "vox")
RUN_TAG = "moon"
MONGO_COLLECTION_PREFIX = f"{RUN_TAG.upper()}_phase2"
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
OUTPUT_SEARCH_CACHE_DIR = BASE_DIR / "PHASE_2_SEARCH_CACHE" # <-- NEW: Search Cache
OUTPUT_CACHE_DIR.mkdir(exist_ok=True) 
OUTPUT_SEARCH_CACHE_DIR.mkdir(exist_ok=True) # <-- NEW: Create Dir

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
MAX_VIDEOS = 2500 # Your filter for news orgs
VIDEOS_PER_CANDIDATE = 20 # How many videos to fetch for LLM analysis

# --- Rate Limiting ---
DELAY_BETWEEN_CANDIDATES = 2 # Shorter delay, no LLM call
DELAY_BETWEEN_SEEDS = 10


# === HELPER FUNCTIONS (Unchanged) ===

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

# --- Main processing function (UPDATED WITH CACHE) ---
def process_seed_channel(
    seed_channel,
    seed_channel_id,
    seed_keywords,  # This is a DICTIONARY
    seen_channels_data,
    seen_ids,
    cached_ids,     # Set of already cached IDs
    cache_file_path, # Path to save data
    search_cache_path: Path # <-- NEW: Path for the search cache
):
    """
    Process a single seed channel.
    NOW INCLUDES a cache for Step 1 (the expensive search).
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

    # --- STEP 1: Multi-Focused Search (NOW WITH CACHING) ---
    print(f"\n🔍 STEP 1: Finding candidate channels...")

    # Check if a search cache for this fingerprint already exists
    if search_cache_path.exists():
        print(f"  ✅ Found existing search cache: {search_cache_path.name}")
        try:
            with open(search_cache_path, 'r') as f:
                # Load the list of IDs and convert to a set
                candidate_ids = set(json.load(f))
            print(f"  Loaded {len(candidate_ids)} candidates from cache. (0 quota units used)")
        except Exception as e:
            print(f"  ⚠️ Error loading cache: {e}. Forcing a new search.")
            candidate_ids = None # Set to None to trigger search
    else:
        print(f"  ℹ️ No search cache found. Running new search (This will use quota)...")
        candidate_ids = None # Set to None to trigger search

    if candidate_ids is None:
        # This is the "run the search" block
        print(f"   Searching YouTube with {len(seed_keywords_list)} keywords...")
        try:
            candidate_ids = search_videos_multi_focused(
                seed_keywords_list,
                max_results_per_search=30,
                max_keywords=len(seed_keywords_list)
            )
            # --- SAVE TO CACHE ---
            try:
                with open(search_cache_path, 'w') as f:
                    # Convert set to list for JSON serialization
                    json.dump(list(candidate_ids), f)
                print(f"  ✅ Saved {len(candidate_ids)} found candidates to cache: {search_cache_path.name}")
            except Exception as e:
                print(f"  ⚠️ Error saving to search cache: {e}")
            # --- END SAVE ---
        except Exception as e:
            print(f"❌ Search failed: {e}")
            # If we fail the search (e.g., quota), we must stop.
            # Raise the exception to be caught by main()
            raise e 


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
        # --- Your Max Video Count Filter ---
        if meta["video_count"] > MAX_VIDEOS:
            print(f"  - Filtering {meta['name']} (videos: {meta['video_count']:,}) - LIKELY A NEWS ORG")
            _update_status(seen_channels_data, meta["id"], "filtered_max_videos")
            continue
        # --- End Filter ---
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
                candidate["id"], max_results=VIDEOS_PER_CANDIDATE, filter_shorts=True, min_videos_in_first_batch=5, max_items_to_scan=500
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
                "Discovered_Videos_JSON": json.dumps(videos),
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
            error_str = str(e)
            print(f"     ❌ Error on {candidate['name']}: {error_str[:150]}")
            
            # --- This is the quota fix from last time ---
            if "quotaExceeded" in error_str or "403" in error_str:
                print("\n" + "="*50)
                print("     🛑 QUOTA EXCEEDED. Stopping gracefully.")
                print("     All data saved so far is safe in the CSV.")
                print("     Re-run this script tomorrow to continue.")
                print("="*50 + "\n")
                _update_status(seen_channels_data, candidate["id"], "error_quota_limit")
                save_seen_channels(seen_channels_data, SEEN_CHANNELS_PATH) # Save log one last time
                break # <-- This exits the loop
            # --- END FIX ---
            
            _update_status(seen_channels_data, candidate["id"], "error_caching")

    return new_channels_cached_count


# === MAIN FUNCTION (UPDATED WITH CACHE) ===
def main():
    start_time = time.time()
    print("=" * 70)
    print(f"CHANNEL DISCOVERY PIPELINE (PHASE 2)")
    print(f"Run Tag: {RUN_TAG}")
    print(f"Seeds: {', '.join(SEED_CHANNELS)}")
    print(f"Output Cache: {CACHE_DATA_PATH.name}")
    print("=" * 70)

    # --- 1. Load Seed Keywords (from Phase 1 JSON) ---
    print(f"\n📖 Loading seed keywords for '{RUN_TAG}' from MongoDB...")
    seed_keywords_map = {}
    try:
        # Phase-1 fingerprints are stored in <RUN_TAG>_phase1_fingerprints
        df_fp = load_collection_as_df(f"{RUN_TAG.upper()}_phase1_fingerprints", {"metadata.run_tag": RUN_TAG})
        if df_fp.empty:
            raise ValueError(f"No fingerprints in Mongo for run_tag={RUN_TAG}")

        # The run-level blob is one document; reconstruct dict like the file structure you used before
        fp_blob = df_fp.iloc[0].to_dict()
        channels_dict = fp_blob.get("channels", {})

        # Keep using your existing SEED_CHANNELS filter
        for channel_id, details in channels_dict.items():
            channel_name = details.get("channel_name")
            if channel_name in SEED_CHANNELS:
                kws = details.get("fingerprint", {}).get("keywords", {})
                if kws:
                    seed_keywords_map[channel_name] = kws
                    print(f"  📌 Found keywords for seed: {channel_name}")
                else:
                    print(f"  ⚠️  No keywords in fingerprint for {channel_name}")

        if not seed_keywords_map:
            print(f"❌ ERROR: Could not find keywords for any seeds in {SEED_CHANNELS}.")
            return
    except Exception as e:
        print(f"❌ ERROR loading fingerprints from Mongo: {e}")
        return


    # --- 2. Load Seed Channel ID Mapping (from Phase 1 CSV) ---
    print(f"\n🗺️  Loading channel IDs for '{RUN_TAG}' from MongoDB...")
    try:
        # Phase-1 videos live in <RUN_TAG>_phase1 (one doc per video)
        df_videos = load_collection_as_df(f"{RUN_TAG.upper()}_phase1")
        if df_videos.empty:
            raise ValueError("Phase-1 videos collection is empty in Mongo")

        # Your original logic, but on df_videos from Mongo
        seed_id_map = (
            df_videos.drop_duplicates(subset=["Channel_Name"])[["Channel_Name", "Channel_ID"]]
            .set_index("Channel_Name")["Channel_ID"]
            .to_dict()
        )
        print(f"  ✅ Loaded {len(seed_id_map)} channel mappings")
    except Exception as e:
        print(f"❌ ERROR loading Channel_ID map from Mongo: {e}")
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
            
        try:
            # --- NEW: Define the cache path based on the fingerprint file ---
            fingerprint_filename = latest_fingerprint_file.stem # Gets name without .json
            # Creates a cache name like: search_cache_vox_20251101_192102.json
            search_cache_filename = f"search_cache_{fingerprint_filename.replace('fingerprints_oneshot_', '')}.json"
            search_cache_path = OUTPUT_SEARCH_CACHE_DIR / search_cache_filename

            new_channels_this_seed = process_seed_channel(
                seed_channel,
                seed_channel_id,
                seed_keywords, # Pass the DICT
                seen_channels_data,
                seen_ids,
                cached_channel_ids,  # <-- PASS THE SET
                CACHE_DATA_PATH,     # <-- PASS THE PATH
                search_cache_path    # <-- NEW: PASS THE SEARCH CACHE PATH
            )
            
            total_new_channels_cached += new_channels_this_seed

        except Exception as e:
            # This will catch the "quotaExceeded" error from Step 1
            if "quotaExceeded" in str(e) or "403" in str(e):
                print(f"\n🛑 CRITICAL: Quota exceeded during search for seed '{seed_channel}'.")
                print("   Stopping the entire script. Please re-run tomorrow.")
                break # Exit the loop over seeds
            else:
                print(f"\n❌ UNEXPECTED ERROR processing seed '{seed_channel}': {e}")
                print("   Skipping this seed and continuing...")
                continue # Go to the next seed

        print(f"\n✅ Seed '{seed_channel}' complete:")
        print(f"   Cached {new_channels_this_seed} new channels this run.")

        if seed_idx < len(SEED_CHANNELS):
            print(f"\n⏸️  Waiting {DELAY_BETWEEN_SEEDS}s before next seed...")
            time.sleep(DELAY_BETWEEN_SEEDS)

    try:
        if CACHE_DATA_PATH.exists():
            df_cache_all = pd.read_csv(CACHE_DATA_PATH)
            if not df_cache_all.empty:
                # composite unique key so a discovered channel can exist per seed
                df_cache_all["SeedDiscoveredKey"] = (
                    df_cache_all["Seed_Channel_ID"].astype(str)
                    + "::"
                    + df_cache_all["Discovered_Channel_ID"].astype(str)
                )
                # optional audit columns
                df_cache_all["run_tag"] = RUN_TAG
                df_cache_all["mirrored_at"] = datetime.now().isoformat()

                save_dataframe_to_mongo(
                    df_cache_all,
                    collection_name=f"{RUN_TAG.upper()}_phase2",
                    unique_key_column="SeedDiscoveredKey"
                )
                print(f"✅ Mongo: Upserted {len(df_cache_all)} discovered rows into '{RUN_TAG.upper()}_phase2'.")
            else:
                print("⚠️ Cache CSV exists but is empty; nothing to push to Mongo.")
        else:
            print("⚠️ No cache CSV found; nothing to push to Mongo.")
    except Exception as e:
        print(f"❌ Mongo push failed for Phase 2 final discoveries: {e}")
    # === FINAL SUMMARY ===
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 2 DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"Seeds processed: {len(SEED_CHANNELS)}")
    print(f"Total new channels cached this run: {total_new_channels_cached}")
    print(f"Total channels in cache: {len(cached_channel_ids) + total_new_channels_cached}")
    print(f"⏱️  Total runtime: {elapsed / 60:.1f} minutes")
    print(f"\n✅ Next step: Run 'phase2_5_embedding_triage.py' to process:")
    print(f"   {CACHE_DATA_PATH.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()