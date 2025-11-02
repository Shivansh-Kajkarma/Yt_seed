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
        # fetch_recent_videos, # <-- We don't need this anymore
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
RUN_TAG = "test_society_destroyed_channels_only"
#
# --- PASTE YOUR KEYWORDS HERE ---
MANUAL_KEYWORDS = [
    "how billionaires exploit the poor"
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

# --- NEW: This is now the FINAL report, not a cache ---
FINAL_REPORT_PATH = OUTPUT_CACHE_DIR / f"phase2_CHANNELS_ONLY_report_{RUN_TAG}.csv"

# The high-level log file this script UPDATES
SEEN_CHANNELS_PATH = BASE_DIR / f"seen_channels_MANUAL_TESTS.csv" # A separate log for tests


# --- NEW: Simplified columns for the metadata-only report ---
REPORT_COLUMN_ORDER = [
    "Seed_Keywords", # <-- Added to see what found this
    "Discovered_Channel_Name", "Discovered_Channel_ID", "Discovered_Channel_URL",
    "Discovered_Subs", "Discovered_Video_Count", "Discovered_Country",
    "Timestamp",
]

AUTO_KEEP_COUNTRIES = [
    'US', 'GB', 'CA', 'AU', 'NZ', 'NG', 'Unknown'
]

# --- Settings ---
MIN_SUBSCRIBERS = 10000
MIN_VIDEOS = 6
# VIDEOS_PER_CANDIDATE = 20 # <-- Not needed


# === HELPER FUNCTIONS (Modified) ===

def load_processed_channels(file_path):
    """Loads the report file and returns a set of processed channel IDs."""
    if not file_path.exists():
        return set()
    try:
        df = pd.read_csv(file_path, usecols=["Discovered_Channel_ID"])
        return set(df["Discovered_Channel_ID"].astype(str).tolist())
    except Exception as e:
        print(f"  ⚠️  Could not read report file {file_path.name}: {e}")
        return set()

def append_to_report_csv(data_dict, file_path):
    """Appends a single row (dict) to the final report CSV file."""
    file_exists = file_path.exists()
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_COLUMN_ORDER, extrasaction='ignore')
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
    seed_keywords_list,
    seen_channels_data,
    seen_ids,
    processed_ids,  # <-- Set of already processed IDs
    report_file_path # <-- Path to save data
):
    """
    MODIFIED version. Stops after Step 4 and saves only metadata.
    """
    print("\n" + "=" * 70)
    print(f"PROCESSING MANUAL TEST: {RUN_TAG}")
    print(f"  (Will skip {len(processed_ids)} channels already in {report_file_path.name})")
    print("=" * 70)

    current_time = datetime.now().isoformat()
    new_channels_saved_count = 0

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

    # --- STEP 4: Filter against *already processed* channels ---
    print(f"\n🔍 STEP 4: Filtering against {len(processed_ids)} already processed channels...")
    candidates_to_process = []
    for c in qualified:
        if c['id'] not in processed_ids:
            candidates_to_process.append(c)
            
    if not candidates_to_process:
        print("  ✅ No new qualified candidates to save for this test.")
        return 0
    print(f"  🎯 {len(candidates_to_process)} new candidates to save (out of {len(qualified)} qualified).")


    # --- STEP 5: SAVE METADATA (NO VIDEO FETCH) ---
    print(f"\n🎯 STEP 5: Saving metadata for {len(candidates_to_process)} candidates...")

    for i, candidate in enumerate(candidates_to_process, 1):
        try:
            # Build the simple report dictionary
            report_data = {
                "Seed_Keywords": ", ".join(seed_keywords_list),
                "Discovered_Channel_Name": candidate["name"],
                "Discovered_Channel_ID": candidate["id"],
                "Discovered_Channel_URL": candidate["url"],
                "Discovered_Subs": candidate["subscribers"],
                "Discovered_Video_Count": candidate["video_count"],
                "Discovered_Country": candidate.get("country", "Unknown"),
                "Timestamp": datetime.now().isoformat(),
            }

            # Save the metadata row
            append_to_report_csv(report_data, report_file_path)
            new_channels_saved_count += 1
            print(f"     ✅ Saved metadata for {candidate['name']} ({i}/{len(candidates_to_process)})")

            # Update high-level log
            _update_status(seen_channels_data, candidate["id"], "saved_metadata_only")
            save_seen_channels(seen_channels_data, SEEN_CHANNELS_PATH)
            
        except Exception as e:
            print(f"     ❌ Error saving {candidate['name']}: {str(e)[:100]}")
            _update_status(seen_channels_data, candidate["id"], "error_saving_metadata")

    return new_channels_saved_count


# === MAIN FUNCTION ===
def main():
    start_time = time.time()
    print("=" * 70)
    print(f"MANUAL KEYWORD TEST (METADATA ONLY)")
    print(f"Run Tag: {RUN_TAG}")
    print(f"Output Report: {FINAL_REPORT_PATH.name}")
    print("=" * 70)

    # --- 1. Load Seen Channels Log ---
    print(f"\n📂 Loading seen channels log from {SEEN_CHANNELS_PATH.name}...")
    seen_channels_data, seen_ids = load_seen_channels(SEEN_CHANNELS_PATH)

    # --- 2. Load ALREADY PROCESSED channels (from this test's report file) ---
    print(f"\n🔄 Loading already processed channels from {FINAL_REPORT_PATH.name}...")
    processed_channel_ids = load_processed_channels(FINAL_REPORT_PATH)
    print(f"  ✅ Found {len(processed_channel_ids)} channels in report to skip.")

    # --- 3. Process The Manual Keywords ---
    total_new_channels_saved = process_manual_keywords(
        MANUAL_KEYWORDS,
        seen_channels_data,
        seen_ids,
        processed_channel_ids,
        FINAL_REPORT_PATH
    )

    # === FINAL SUMMARY ===
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("MANUAL TEST COMPLETE (METADATA ONLY)")
    print("=" * 70)
    print(f"Keywords tested: {', '.join(MANUAL_KEYWORDS)}")
    print(f"Total new channels saved this run: {total_new_channels_saved}")
    print(f"Total channels in this test's report: {len(processed_channel_ids) + total_new_channels_saved}")
    print(f"⏱️  Total runtime: {elapsed / 60:.1f} minutes")
    print(f"\n✅ You can now inspect the results in:")
    print(f"   {FINAL_REPORT_PATH.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()