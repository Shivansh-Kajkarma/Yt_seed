import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime
import sys
from typing import Tuple

# --- Make sure utils are importable ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.mongo_utils import load_seen_channels_from_mongo, save_seen_channels_to_mongo, load_cached_ids_from_mongo, save_search_cache_to_mongo, load_search_cache_from_mongo
try:
    # --- Import YouTube & Mongo Utils ---
    from utils.youtube_utils import (
        search_videos_multi_focused,
        get_channel_metadata_batch,
        fetch_recent_videos,
    )
    from utils.mongo_utils import (
        load_collection_as_df, 
        save_dataframe_to_mongo
    )

    print("✅ Successfully imported YouTube & Mongo utils.")
except ImportError as e: 
    print("Error: Could not import from 'utils' directory.")
    print(f"Ensure 'utils' is at this path: {BASE_DIR / 'utils'}")
    raise e # <-- CHANGED: Raise error, don't exit

# ==================================================
# 1. CONFIGURATION
# ==================================================

# --- Mongo Collection Names ---
# We use a single log for all seen channels, partitioned by run_tag
MONGO_SEEN_CHANNELS_LOG = "run_progress_seen_channels"
# We use a single cache for all YouTube search results
MONGO_SEARCH_CACHE = "cache_youtube_searches"

# --- Static Filters (Unchanged) ---
AUTO_KEEP_COUNTRIES = [
    'US', 'GB', 'CA', 'AU', 'NZ', 'NG', 'Unknown'
]
MIN_SUBSCRIBERS = 10000
MIN_VIDEOS = 6
MAX_VIDEOS = 5000 # Your filter for news orgs
VIDEOS_PER_CANDIDATE = 3 # How many videos to fetch for LLM analysis #changed from 10 to 3
# --- Rate Limiting (Unchanged) ---
DELAY_BETWEEN_CANDIDATES = 2 # Shorter delay, no LLM call
DELAY_BETWEEN_SEEDS = 10


# ==================================================
# 3. MAIN PROCESSING FUNCTION (REFACTORED)
# ==================================================

def process_seed_channel(
    seed_channel: str,
    seed_channel_id: str,
    seed_keywords: dict,  # This is a DICTIONARY
    seen_channels_dict: dict, # <-- CHANGED: Now a dict
    seen_ids: set,
    cached_ids: set,      # Set of already cached IDs
    run_tag: str,
    mongo_phase2_collection: str # <-- CHANGED: Pass in collection name
):
    """
    Process a single seed channel.
    All I/O is now with MongoDB.
    """

    print("\n" + "=" * 70)
    print(f"PROCESSING SEED: {seed_channel}")
    print(f"  (Will skip {len(cached_ids)} channels already in {mongo_phase2_collection})")
    print("=" * 70)

    current_time = datetime.now().isoformat()
    new_channels_cached_count = 0

    # --- Flatten seed keywords (Unchanged) ---
    seed_keywords_list = []
    
    # Case A: New Format (List)
    if isinstance(seed_keywords, list):
        seed_keywords_list = seed_keywords
        
    # Case B: Old Format (Dict) - Backward Compatibility
    elif isinstance(seed_keywords, dict):
        for k, v in seed_keywords.items():
            if isinstance(v, list): seed_keywords_list.extend(v)
            
    else:
        print(f"❌ Seed keywords for {seed_channel} are invalid format. Skipping.")
        return 0
    
    # Deduplicate and Limit
    seed_keywords_list = list(set(seed_keywords_list))
    
    if not seed_keywords_list:
        print(f"❌ No seed keywords found for {seed_channel}. Skipping.")
        return 0

    # --- STEP 1: Multi-Focused Search (REFACTORED FOR MONGO) ---
    print("\n🔍 STEP 1: Finding candidate channels...")

    # Create a unique cache key based on the seed ID and top 3 keywords
    search_cache_key = f"{seed_channel_id}::{'|'.join(sorted(seed_keywords_list[:3]))}"

    # Check Mongo cache
    candidate_ids = load_search_cache_from_mongo(search_cache_key, MONGO_SEARCH_CACHE)

    if candidate_ids is None:
        # This is the "run the search" block
        print(f"   Searching YouTube with {len(seed_keywords_list)} keywords...")
        try:
            candidate_ids = search_videos_multi_focused(
                seed_keywords_list[:3],  #changes from complete->3
                max_results_per_search=3,  #changes from 20->2
                max_keywords=len(seed_keywords_list),
                run_tag=run_tag,
                seed_name=seed_channel
            )
            # --- SAVE TO MONGO CACHE ---
            save_search_cache_to_mongo(search_cache_key, candidate_ids, run_tag, MONGO_SEARCH_CACHE)
            # --- END SAVE ---
        except Exception as e:
            print(f"❌ Search failed: {e}")
            raise e # Raise to be caught by wrapper (e.g., quota)

    candidate_ids = candidate_ids - seen_ids - {seed_channel_id}
    if not candidate_ids:
        print("  ⚠️  No new candidates found after filtering seen log")
        return 0
    print(f"  ✅ {len(candidate_ids)} new candidates to evaluate")

    # --- STEP 2: Get Metadata (REFACTORED FOR MONGO) ---
    print("\n📊 STEP 2: Fetching channel metadata...")
    metadata = get_channel_metadata_batch(list(candidate_ids), run_tag=run_tag, seed_name=seed_channel)
    
    # Update the seen log in memory
    new_seen_count = 0
    for meta in metadata:
        if meta["id"] not in seen_ids:
            seen_channels_dict[meta["id"]] = {
                "Channel_ID": meta["id"], 
                "Channel_Name": meta["name"],
                "Date_Added": current_time, 
                "Processing_Status": "discovered",
                "run_tag": run_tag # Partition key
            }
            seen_ids.add(meta["id"])
            new_seen_count += 1
    
    if new_seen_count > 0:
        print(f"  ...found {new_seen_count} new channels. Saving seen log to Mongo.")
        save_seen_channels_to_mongo(seen_channels_dict, MONGO_SEEN_CHANNELS_LOG) # Save new discoveries to log
    else:
        print("  ...no new channels found in this batch.")

    # --- STEP 3: Pre-Filter (REFACTORED FOR MONGO) ---
    print("\n🔍 STEP 3: Pre-filtering by subscribers and videos...")
    qualified = []
    log_updated = False
    for meta in metadata:
        channel_id = meta["id"]
        status = ""
        if meta["subscribers"] == -1:
            status = "filtered_hidden_subs"
        elif meta["subscribers"] < MIN_SUBSCRIBERS:
            status = "filtered_subs"
        elif meta["video_count"] < MIN_VIDEOS:
            status = "filtered_videos"
        elif meta["video_count"] > MAX_VIDEOS:
            status = "filtered_max_videos"
        elif meta.get('country', 'Unknown') not in AUTO_KEEP_COUNTRIES:
            status = f"filtered_country_{meta.get('country', 'Unknown')}"
        
        if status:
            print(f"  - Filtering {meta['name']} ({status})")
            if channel_id in seen_channels_dict:
                seen_channels_dict[channel_id]["Processing_Status"] = status
                log_updated = True
        else:
            qualified.append(meta) # It passed all filters

    if log_updated:
        print("  ...saving filter status to seen log in Mongo.")
        save_seen_channels_to_mongo(seen_channels_dict, MONGO_SEEN_CHANNELS_LOG) # Save filter status to log

    # --- STEP 4: Filter against *already cached* channels (Unchanged) ---
    print(f"\n🔍 STEP 4: Filtering against {len(cached_ids)} already cached channels...")
    candidates_to_process = []
    for c in qualified:
        if c['id'] not in cached_ids:
            candidates_to_process.append(c)
            
    if not candidates_to_process:
        print("  ✅ No new qualified candidates to cache for this seed.")
        return 0
    print(f"  🎯 {len(candidates_to_process)} new candidates to fetch and cache (out of {len(qualified)} qualified).")

    # --- STEP 5: Fetch Video Data & Cache (REFACTORED FOR MONGO) ---
    print(f"\n🎯 STEP 5: Fetching and Caching {len(candidates_to_process)} candidates...")
    for i, candidate in enumerate(candidates_to_process, 1):
        print(f"\n  [{i}/{len(candidates_to_process)}] {candidate['name']}")
        print(f"     Subs: {candidate['subscribers']:,} | Videos: {candidate['video_count']:,}")

        try:
            # --- 5a. Fetch recent videos (Unchanged) ---
            print(f"     Fetching {VIDEOS_PER_CANDIDATE} videos...")
            videos, cand_desc = fetch_recent_videos(
                candidate["id"], 
                max_results=VIDEOS_PER_CANDIDATE, 
                filter_shorts=True, 
                min_videos_in_first_batch=1,  #changed 3->1 
                max_items_to_scan=500, 
                run_tag=run_tag, 
                seed_name=seed_channel
            )
            
            if len(videos) < 1: #changed 3->1 
                print(f"     ⚠️  Only {len(videos)} videos found, logging and skipping")
                if candidate["id"] in seen_channels_dict:
                    seen_channels_dict[candidate["id"]]["Processing_Status"] = "skipped_few_videos"
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
                "run_tag": run_tag, # <-- Add run_tag
                # Composite key for Mongo upsert
                "SeedDiscoveredKey": f"{seed_channel_id}::{candidate['id']}"
            }

            # --- 5c. SAVE-AS-YOU-GO TO MONGO ---
            # We save a 1-row DataFrame. Mongo utils handles the upsert.
            df_to_save = pd.DataFrame([candidate_raw_data])
            save_dataframe_to_mongo(
                df_to_save,
                collection_name=mongo_phase2_collection,
                unique_key_column="SeedDiscoveredKey"
            )
            new_channels_cached_count += 1
            print(f"     ✅ Cached raw data for {candidate['name']} to Mongo.")

            # --- 5d. Update high-level log ---
            if candidate["id"] in seen_channels_dict:
                seen_channels_dict[candidate["id"]]["Processing_Status"] = "cached_for_llm"
            
            # Note: We don't save the seen log *every* time, only on breaks/errors.
            
            time.sleep(DELAY_BETWEEN_CANDIDATES)

        except Exception as e:
            error_str = str(e)
            print(f"     ❌ Error on {candidate['name']}: {error_str[:150]}")
            
            if candidate["id"] in seen_channels_dict:
                seen_channels_dict[candidate["id"]]["Processing_Status"] = "error_caching"

            # --- Quota fix (Unchanged) ---
            if "quotaExceeded" in error_str or "403" in error_str:
                print("\n" + "="*50)
                print("     🛑 QUOTA EXCEEDED. Stopping gracefully.")
                print("     All data saved so far is safe in MongoDB.")
                print("     Re-run this script tomorrow to continue.")
                print("="*50 + "\n")
                if candidate["id"] in seen_channels_dict:
                   seen_channels_dict[candidate["id"]]["Processing_Status"] = "error_quota_limit"
                # Save log one last time before exiting
                save_seen_channels_to_mongo(seen_channels_dict, MONGO_SEEN_CHANNELS_LOG) 
                raise e # Re-raise the quota error to be caught by wrapper
            # --- END FIX ---

    # --- 5e. Final save of the seen log for this seed ---
    print("  ...saving final seen log status for this seed.")
    save_seen_channels_to_mongo(seen_channels_dict, MONGO_SEEN_CHANNELS_LOG)
    
    return new_channels_cached_count


# ==================================================
# 4. MAIN FUNCTION (REFACTORED FOR MONGO)
# ==================================================
def main(run_tag: str):
    start_time = time.time()

    # --- Define Mongo Collection Names ---
    MONGO_FP_COLLECTION = f"{run_tag.upper()}_phase1_fingerprints"
    MONGO_VIDEOS_COLLECTION = f"{run_tag.upper()}_phase1"
    MONGO_PHASE2_COLLECTION = f"{run_tag.upper()}_phase2" # This is our main output

    print("=" * 70)
    print("CHANNEL DISCOVERY PIPELINE (PHASE 2)")
    print(f"Run Tag: {run_tag}")
    print(f"Output Collection: {MONGO_PHASE2_COLLECTION}")
    print("=" * 70)
    
    # --- 1. Load Seed Keywords (from Mongo) ---
    print(f"\n📖 Loading seed keywords for '{run_tag}' from {MONGO_FP_COLLECTION}...")
    seed_keywords_map = {}
    seed_channel_names = []
    try:
        # Load the latest fingerprint for this run_tag
        df_fp_all = load_collection_as_df(MONGO_FP_COLLECTION, {"metadata.run_tag": run_tag})
        if df_fp_all.empty:
            raise ValueError(f"No fingerprints in Mongo for run_tag={run_tag}")
        
        print(df_fp_all.head())
        meta_df = pd.json_normalize(df_fp_all['metadata'])
        
        # --- FIX FOR MISSING metadata.created_at ---
        df_fp_all = pd.concat([
            df_fp_all.drop(columns=['metadata']), 
            meta_df
        ], axis=1)
        
        # 3. NOW we can sort by 'created_at' (it's no longer 'metadata.created_at')
        df_fp_all = df_fp_all.sort_values("created_at", ascending=False)
        fp_blob = df_fp_all.iloc[0].to_dict()
        
        channels_dict = fp_blob.get("channels", {})
        if not channels_dict:
             raise ValueError(f"Fingerprint blob for {run_tag} has no 'channels' key.")

        # --- REMOVED HARDCODED SEED_CHANNELS FILTER ---
        # Process *all* channels found in this fingerprint file
        for channel_id, details in channels_dict.items():
            channel_name = details.get("channel_name")
            if not channel_name:
                continue
            
            fp = details.get("fingerprint", {})
            
            # PRIORITY 1: Check for new "search_keywords" list
            kws = fp.get("search_keywords")
            
            # PRIORITY 2: Fallback to old "keywords" dict
            if not kws:
                kws = fp.get("keywords")
                
            if kws:
                seed_keywords_map[channel_name] = kws
                seed_channel_names.append(channel_name)
                # Check list length or dict size for logging
                count = len(kws) if isinstance(kws, list) else sum(len(v) for v in kws.values())
                print(f"  📌 Found {count} keywords for seed: {channel_name}")
            else:
                print(f"  ⚠️  No keywords in fingerprint for {channel_name}, skipping.")

        if not seed_keywords_map:
            raise ValueError("Could not find valid keywords for any seeds.")
            
    except Exception as e:
        print(f"❌ ERROR loading fingerprints from Mongo: {e}")
        raise e # Stop script


    # --- 2. Load Seed Channel ID Mapping (from Mongo) ---
    print(f"\n🗺️  Loading channel IDs from {MONGO_VIDEOS_COLLECTION}...")
    try:
        df_videos = load_collection_as_df(MONGO_VIDEOS_COLLECTION)
        if df_videos.empty:
            raise ValueError("Phase-1 videos collection is empty in Mongo")

        seed_id_map = (
            df_videos.drop_duplicates(subset=["Channel_Name"])[["Channel_Name", "Channel_ID"]]
            .set_index("Channel_Name")["Channel_ID"]
            .to_dict()
        )
        print(f"  ✅ Loaded {len(seed_id_map)} channel mappings")
    except Exception as e:
        print(f"❌ ERROR loading Channel_ID map from Mongo: {e}")
        raise e # Stop script


    # --- 3. Load Seen Channels (from Mongo) ---
    seen_channels_dict, seen_ids = load_seen_channels_from_mongo(run_tag, MONGO_SEEN_CHANNELS_LOG)
    current_time = datetime.now().isoformat()
    log_updated = False
    for seed_channel in seed_channel_names:
        seed_channel_id = seed_id_map.get(seed_channel)
        if seed_channel_id and seed_channel_id not in seen_ids:
            seen_channels_dict[seed_channel_id] = {
                "Channel_ID": seed_channel_id, 
                "Channel_Name": seed_channel,
                "Date_Added": current_time, 
                "Processing_Status": "seed",
                "run_tag": run_tag
            }
            seen_ids.add(seed_channel_id)
            log_updated = True
            print(f"  ✅ Added seed '{seed_channel}' to seen list")
            
    if log_updated:
        save_seen_channels_to_mongo(seen_channels_dict,MONGO_SEEN_CHANNELS_LOG)

    # --- 4. Load ALREADY CACHED channels (from Mongo) ---
    cached_channel_ids = load_cached_ids_from_mongo(MONGO_PHASE2_COLLECTION)

    # --- 5. Process Each Seed Channel ---
    total_new_channels_cached = 0
    for seed_idx, seed_channel in enumerate(seed_channel_names, 1):
        print(f"\n{'=' * 70}")
        print(f"SEED {seed_idx}/{len(seed_channel_names)}: {seed_channel}")
        print(f"{'=' * 70}")

        # Get all required seed info
        seed_keywords = seed_keywords_map.get(seed_channel)
        seed_channel_id = seed_id_map.get(seed_channel)
        
        if not seed_keywords or not seed_channel_id:
            print(f"⚠️  WARNING: Missing data for '{seed_channel}', skipping")
            continue
            
        try:
            new_channels_this_seed = process_seed_channel(
                seed_channel,
                seed_channel_id,
                seed_keywords,
                seen_channels_dict, # Pass the dict
                seen_ids,
                cached_channel_ids,
                run_tag,
                MONGO_PHASE2_COLLECTION # Pass output collection name
            )
            
            total_new_channels_cached += new_channels_this_seed

        except Exception as e:
            # This catches quota errors raised from process_seed_channel
            if "quotaExceeded" in str(e) or "403" in str(e):
                print(f"\n🛑 CRITICAL: Quota exceeded during processing for seed '{seed_channel}'.")
                print("   Stopping the entire script. Please re-run tomorrow.")
                raise e # Re-raise to be caught by wrapper
            else:
                print(f"\n❌ UNEXPECTED ERROR processing seed '{seed_channel}': {e}")
                print("   Skipping this seed and continuing...")
                continue # Go to the next seed

        print(f"\n✅ Seed '{seed_channel}' complete:")
        print(f"   Cached {new_channels_this_seed} new channels this run.")

        if seed_idx < len(seed_channel_names):
            print(f"\n⏸  Waiting {DELAY_BETWEEN_SEEDS}s before next seed...")
            time.sleep(DELAY_BETWEEN_SEEDS)

    # --- 6. FINAL SUMMARY ---
    # No final Mongo push is needed because we saved as-we-went.
    elapsed = time.time() - start_time
    # Re-load the cached IDs to get the final count
    final_cached_ids = load_cached_ids_from_mongo(MONGO_PHASE2_COLLECTION)

    print("\n" + "=" * 70)
    print("PHASE 2 DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"Seeds processed: {len(seed_channel_names)}")
    print(f"Total new channels cached this run: {total_new_channels_cached}")
    print(f"Total channels in cache: {len(final_cached_ids)}")
    print(f"⏱️  Total runtime: {elapsed / 60:.1f} minutes")
    print("\n✅ Next step: Run 'phase2_5_embedding_triage.py'")
    print("=" * 70)

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "DEFAULT"
    main(tag)