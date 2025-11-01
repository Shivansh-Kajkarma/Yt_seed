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
    from utils.youtube_utils import (
        search_videos_multi_focused,
        get_channel_metadata_batch,
        fetch_recent_videos,
    )
except ImportError:
    print("Error: Could not import from 'utils' directory.")
    print(f"Ensure 'utils' is at this path: {BASE_DIR / 'utils'}")
    sys.exit(1)


# ==================================================
# 1. CONFIGURATION
# ==================================================
#
# --- EDIT THIS TAG FOR EACH TEST ---
# Change this for each test to create a new, separate output file.
RUN_TAG = "test_society_destroyed"
#
# --- PASTE YOUR KEYWORDS HERE ---
MANUAL_KEYWORDS = [
    "society destroyed"
    # "exposed as a monster",
    # "everything wrong with",
    # "brain rot exposed"
]
#
# ==================================================


# --- "Dummy" Seed Info (For logging purposes) ---
SEED_NAME_FOR_LOGS = "manual_test"
SEED_ID_FOR_LOGS = "manual_test_id"


# --- Output Files (for this test run) ---
OUTPUT_CACHE_DIR = BASE_DIR / "PHASE_2_DISCOVERY_CACHE"
OUTPUT_CACHE_DIR.mkdir(exist_ok=True) # Ensure directory exists

# The cache file that this script WRITES TO
CACHE_DATA_PATH = OUTPUT_CACHE_DIR / f"phase2_discovered_raw_data_{RUN_TAG}.csv"

# The high-level log file this script UPDATES
SEEN_CHANNELS_PATH = BASE_DIR / f"seen_channels_MANUAL_TESTS.csv" # A separate log for tests


# --- Define ALL columns for the new cache file ---
CACHE_COLUMN_ORDER = [
    "Seed_Channel_Name", "Seed_Channel_ID",
    "Discovered_Channel_Name", "Discovered_Channel_ID", "Discovered_Channel_URL",
    "Discovered_Subs", "Discovered_Video_Count", "Discovered_Country",
    "Discovered_Channel_Description", "Discovered_Videos_JSON",
    "Discovery_Level", "Timestamp",
]

AUTO_KEEP_COUNTRIES = [
    'US', 'GB', 'CA', 'AU', 'NZ', 'NG', 'Unknown'
]

# --- Settings ---
MIN_SUBSCRIBERS = 10000
MIN_VIDEOS = 6
VIDEOS_PER_CANDIDATE = 20

# --- Rate Limiting ---
DELAY_BETWEEN_CANDIDATES = 2
DELAY_BETWEEN_SEEDS = 10


# === HELPER FUNCTIONS (Identical to Phase 2) ===

def load_cached_channels(file_path):
    if not file_path.exists():
        return set()
    try:
        df = pd.read_csv(file_path, usecols=["Discovered_Channel_ID"])
        return set(df["Discovered_Channel_ID"].astype(str).tolist())
    except Exception as e:
        print(f"  ⚠️  Could not read cache file {file_path.name}: {e}")
        return set()

def append_to_cache_csv(data_dict, file_path):
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
    for entry in seen_data:
        if entry["Channel_ID"] == channel_id:
            entry["Processing_Status"] = status
            break

# --- MODIFIED processing function ---
def process_manual_keywords(
    seed_keywords_list, # <-- TAKES THE LIST DIRECTLY
    seen_channels_data,
    seen_ids,
    cached_ids,
    cache_file_path
):
    """
    MODIFIED version of the Phase 2 processor.
    Takes a flat list of keywords and runs the discovery.
    """
    print("\n" + "=" * 70)
    print(f"PROCESSING MANUAL TEST: {RUN_TAG}")
    print(f"  (Will skip {len(cached_ids)} channels already in {cache_file_path.name})")
    print("=" * 70)

    current_time = datetime.now().isoformat()
    new_channels_cached_count = 0

    if not seed_keywords_list:
        print(f"❌ No keywords in MANUAL_KEYWORDS list. Stopping.")
        return 0

    # --- STEP 1: Multi-Focused Search ---
    print(f"\n🔍 STEP 1: Searching YouTube with {len(seed_keywords_list)} keywords...")
    print(f"   Keywords: {seed_keywords_list}")
    try:
        candidate_ids = search_videos_multi_focused(
            seed_keywords_list,
            max_results_per_search=50, # Get more results per keyword
            max_keywords=len(seed_keywords_list)
        )
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return 0

    candidate_ids = candidate_ids - seen_ids - {SEED_ID_FOR_LOGS}
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
    save_seen_channels(seen_channels_data, SEEN_CHANNELS_PATH)

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
    save_seen_channels(seen_channels_data, SEEN_CHANNELS_PATH)

    # --- STEP 4: Filter against *already cached* channels ---
    print(f"\n🔍 STEP 4: Filtering against {len(cached_ids)} already cached channels...")
    candidates_to_process = []
    for c in qualified:
        if c['id'] not in cached_ids:
            candidates_to_process.append(c)
            
    if not candidates_to_process:
        print("  ✅ No new qualified candidates to cache for this test.")
        return 0
    print(f"  🎯 {len(candidates_to_process)} new candidates to fetch and cache (out of {len(qualified)} qualified).")

    # --- STEP 5: Fetch Video Data & Cache (Save-as-you-go) ---
    print(f"\n🎯 STEP 5: Fetching and Caching {len(candidates_to_process)} candidates...")

    for i, candidate in enumerate(candidates_to_process, 1):
        print(f"\n  [{i}/{len(candidates_to_process)}] {candidate['name']}")
        print(f"     Subs: {candidate['subscribers']:,} | Videos: {candidate['video_count']:,}")

        try:
            print(f"     Fetching {VIDEOS_PER_CANDIDATE} videos...")
            videos, cand_desc = fetch_recent_videos(
                candidate["id"], max_results=VIDEOS_PER_CANDIDATE, filter_shorts=True
            )
            
            if len(videos) < 3:
                print(f"     ⚠️  Only {len(videos)} videos found, logging and skipping")
                _update_status(seen_channels_data, candidate["id"], "skipped_few_videos")
                continue

            candidate_raw_data = {
                "Seed_Channel_Name": SEED_NAME_FOR_LOGS,
                "Seed_Channel_ID": SEED_ID_FOR_LOGS,
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

            append_to_cache_csv(candidate_raw_data, cache_file_path)
            new_channels_cached_count += 1
            print(f"     ✅ Cached raw data for {candidate['name']} to {cache_file_path.name}")

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
    print(f"MANUAL KEYWORD TEST (PHASE 2)")
    print(f"Run Tag: {RUN_TAG}")
    print(f"Output Cache: {CACHE_DATA_PATH.name}")
    print("=" * 70)

    # --- 1. Load Seen Channels Log ---
    print(f"\n📂 Loading seen channels log from {SEEN_CHANNELS_PATH.name}...")
    seen_channels_data, seen_ids = load_seen_channels(SEEN_CHANNELS_PATH)

    # --- 2. Load ALREADY CACHED channels (from this test's cache file) ---
    print(f"\n🔄 Loading already cached channels from {CACHE_DATA_PATH.name}...")
    cached_channel_ids = load_cached_channels(CACHE_DATA_PATH)
    print(f"  ✅ Found {len(cached_channel_ids)} channels in cache to skip.")

    # --- 3. Process The Manual Keywords ---
    total_new_channels_cached = process_manual_keywords(
        MANUAL_KEYWORDS,
        seen_channels_data,
        seen_ids,
        cached_channel_ids,
        CACHE_DATA_PATH
    )

    # === FINAL SUMMARY ===
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("MANUAL TEST COMPLETE")
    print("=" * 70)
    print(f"Keywords tested: {', '.join(MANUAL_KEYWORDS)}")
    print(f"Total new channels cached this run: {total_new_channels_cached}")
    print(f"Total channels in this test's cache: {len(cached_channel_ids) + total_new_channels_cached}")
    print(f"⏱️  Total runtime: {elapsed / 60:.1f} minutes")
    print(f"\n✅ You can now inspect the results in:")
    print(f"   {CACHE_DATA_PATH.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()