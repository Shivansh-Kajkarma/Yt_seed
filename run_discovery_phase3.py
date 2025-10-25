# /run_discover_score_log.py (Debug MagnatesMedia + Incremental Save)

import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime
import sys

# --- Import Helper Functions ---
from utils.youtube_utils import (
    search_videos_focused, # Now uses OR
    get_channel_metadata_batch,
    fetch_recent_videos
)
from utils.fingerprint_llm_utils import (
    create_channel_fingerprint_llm,
    calculate_jaccard_similarity
)

# --- Configuration ---
base_dir = Path(__file__).resolve().parent
# Input Files
keywords_file_path = base_dir / "channel_keywords_simple_gpt.json"
seed_video_data_path = base_dir / "sample_videos_new.csv"
seen_channels_path = base_dir / "seen_channels.csv"
# Output Files
all_results_path = base_dir / "all_discovered_channels.csv"
high_similarity_path = base_dir / "high_similarity_channels.csv"
# LLM Choice & Target Channel for Debugging
MODEL_PROVIDER = "gpt" # Ensure this matches your keywords file
TARGET_SEED_CHANNEL = "MagnatesMedia" # <<<--- FOCUS ON THIS CHANNEL
# Thresholds & Limits
SIMILARITY_THRESHOLD = 0.3 # <<<--- LOWER THRESHOLD FOR DEBUGGING JACCARD
MIN_SUBSCRIBERS = 10000
MIN_VIDEOS = 10
MAX_CANDIDATES_TO_SCORE = 5
VIDEOS_PER_CANDIDATE = 10
# Delays (in seconds) - Keep them reasonable
DELAY_BETWEEN_SEEDS = 5
DELAY_BETWEEN_CANDIDATES = 3 # Slightly increased for LLM pacing

# --- Helper Function for Incremental Save ---
def save_seen_channels(seen_data_list, file_path):
    """Saves the seen channels data list to CSV."""
    if not seen_data_list:
        print("  (No seen channel data to save yet.)")
        return
    try:
        df_seen_updated = pd.DataFrame(seen_data_list)
        # Sort and remove duplicates just before saving
        df_seen_updated = df_seen_updated.sort_values(by="Date_Added").drop_duplicates(subset=['Channel_ID'], keep='last')
        df_seen_updated.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"  💾 Saved {len(df_seen_updated)} seen channel entries to {file_path.name}")
    except Exception as e:
        print(f"  ❌ Error saving {file_path.name}: {e}")
# --- End Helper ---


# --- Main Logic ---
def main():
    start_time = time.time()
    print(f"--- Starting Discovery & Scoring Process (DEBUG MODE: {TARGET_SEED_CHANNEL}) ---")
    print(f"Using LLM Provider: {MODEL_PROVIDER.upper()}")

    # 1. Load Seed Keywords & Filter for Target
    # ... (loading keywords_file_path - same as before) ...
    try:
        with open(keywords_file_path, 'r', encoding='utf-8') as f:
            all_seed_keywords = json.load(f)
        # --- FILTER FOR TARGET CHANNEL ---
        seed_channel_keywords = {TARGET_SEED_CHANNEL: all_seed_keywords.get(TARGET_SEED_CHANNEL)}
        if not seed_channel_keywords[TARGET_SEED_CHANNEL]:
            print(f"❌ ERROR: Target channel '{TARGET_SEED_CHANNEL}' not found in keyword file or has no keywords."); return
        print(f"  Loaded keywords for target seed channel: {TARGET_SEED_CHANNEL}.")
    except Exception as e: print(f"❌ ERROR loading keywords: {e}"); return


    # 2. Load Seed Channel Name -> ID Mapping (same as before)
    # ... (loading seed_video_data_path - same as before) ...
    seed_name_to_id = {};
    try:
        df_videos = pd.read_csv(seed_video_data_path)
        if not all(col in df_videos.columns for col in ['Channel_Name', 'Channel_ID']): raise ValueError("CSV needs 'Channel_Name', 'Channel_ID'")
        seed_name_to_id = df_videos.drop_duplicates(subset=['Channel_Name'])[['Channel_Name', 'Channel_ID']].set_index('Channel_Name')['Channel_ID'].to_dict()
        print(f"  Created mapping for {len(seed_name_to_id)} unique seed channels found in video data.")
        if TARGET_SEED_CHANNEL not in seed_name_to_id:
             print(f"⚠️ WARNING: Target channel '{TARGET_SEED_CHANNEL}' ID not found in mapping CSV.")
             # Attempt to extract ID from seed_channels.csv maybe? For now, we'll let it fail later if ID is needed.
    except Exception as e: print(f"❌ ERROR processing seed video CSV: {e}"); return


    # 3. Load or Initialize Seen Channels (same as before)
    # ... (loading seen_channels_path - same as before) ...
    seen_channels_data = []; seen_ids = set()
    if seen_channels_path.exists():
        try:
            df_seen = pd.read_csv(seen_channels_path)
            if all(col in df_seen.columns for col in ['Channel_ID', 'Channel_Name', 'Date_Added', 'Processing_Status']):
                 seen_channels_data = df_seen.to_dict('records'); seen_ids = set(df_seen['Channel_ID'].astype(str).tolist())
                 print(f"  Loaded {len(seen_ids)} previously seen channel IDs.")
            else: print("⚠️ WARNING: seen_channels.csv columns incorrect. Starting fresh.")
        except Exception as e: print(f"⚠️ WARNING: Could not read seen_channels.csv: {e}. Starting fresh.")
    else: print("  seen_channels.csv not found. Starting fresh.")
    # Add target seed if not seen (same logic)
    current_time_str = datetime.now().isoformat()
    target_seed_id = seed_name_to_id.get(TARGET_SEED_CHANNEL)
    if target_seed_id and target_seed_id not in seen_ids:
         seen_channels_data.append({"Channel_ID": target_seed_id, "Channel_Name": TARGET_SEED_CHANNEL, "Date_Added": current_time_str, "Processing_Status": "seed"})
         seen_ids.add(target_seed_id)
         print(f"  Added target seed '{TARGET_SEED_CHANNEL}' to seen list.")
    print(f"  Total seen IDs before processing: {len(seen_ids)}")
    # --- INCREMENTAL SAVE 1: Save seen list after initialization ---
    save_seen_channels(seen_channels_data, seen_channels_path)


    # 4. Initialize Results Lists (same as before)
    all_results_list = []
    high_similarity_results_list = []

    # --- Main Loop (Now only runs for TARGET_SEED_CHANNEL) ---
    print("\n--- Starting Discovery & Scoring Loop ---")
    seed_channel_name = TARGET_SEED_CHANNEL
    seed_keywords = seed_channel_keywords[seed_channel_name]
    seed_channel_id = seed_name_to_id.get(seed_channel_name)

    if not seed_channel_id:
        print(f"❌ ERROR: Cannot proceed without Channel ID for '{seed_channel_name}'.")
        return # Exit if target seed ID is missing

    # Validate keywords (same as before)
    if not seed_keywords or seed_keywords == ["ERROR_PROCESSING"] or len(seed_keywords) < 3:
         print(f"  Skipping - Insufficient or invalid keywords for {seed_channel_name}.")
         return # Exit if target seed has bad keywords

    loop_start_time = time.time()
    print(f"\n[1/1] Processing Seed: {seed_channel_name}")

    # --- Step 1: Single Smart Search (Uses OR function now) ---
    try:
        # Calls the function which now uses " OR "
        candidate_ids_set = search_videos_focused(seed_keywords, max_results=20)
    except Exception as search_err:
         print(f"  ❌ Search failed for {seed_channel_name}: {search_err}. Stopping.")
         return

    # --- Step 2: Filter Seen & Get Metadata (same as before) ---
    candidate_ids_to_fetch = list(candidate_ids_set - seen_ids - {seed_channel_id})
    if not candidate_ids_to_fetch:
        print("  No new candidate channels found after filtering seen IDs.")
        # Save seen_channels again even if no candidates found for this seed
        save_seen_channels(seen_channels_data, seen_channels_path)
        return # Exit after processing the single target seed

    print(f"  Fetching metadata for {len(candidate_ids_to_fetch)} new candidates...")
    candidate_metadata_list = get_channel_metadata_batch(candidate_ids_to_fetch)

    # Update seen_channels_data (same as before)
    current_time_str = datetime.now().isoformat()
    new_discovered_count = 0
    for meta in candidate_metadata_list:
         if meta['id'] not in seen_ids:
              seen_channels_data.append({"Channel_ID": meta['id'], "Channel_Name": meta['name'], "Date_Added": current_time_str, "Processing_Status": "discovered"})
              seen_ids.add(meta['id'])
              new_discovered_count += 1
    if new_discovered_count > 0:
        print(f"  Added {new_discovered_count} newly discovered channels to seen list.")
        # --- INCREMENTAL SAVE 2: Save seen list after adding discovered ---
        save_seen_channels(seen_channels_data, seen_channels_path)

    # --- Step 3: Pre-Filter (same as before) ---
    qualified_channels = []
    # ... (filtering logic same as before, updating seen_channels_data status) ...
    for ch_meta in candidate_metadata_list:
        filtered = False
        status_update = 'discovered' # Keep default status if qualified
        if ch_meta['subscribers'] != -1 and ch_meta['subscribers'] < MIN_SUBSCRIBERS:
            print(f"    - Filtering {ch_meta['name']} (Subs: {ch_meta['subscribers']} < {MIN_SUBSCRIBERS})")
            status_update = 'filtered_subs'; filtered = True
        elif ch_meta['video_count'] < MIN_VIDEOS:
            print(f"    - Filtering {ch_meta['name']} (Videos: {ch_meta['video_count']} < {MIN_VIDEOS})")
            status_update = 'filtered_videos'; filtered = True

        # Update status in seen_channels_data for ALL discovered channels (filtered or not)
        for entry in seen_channels_data:
            if entry['Channel_ID'] == ch_meta['id'] and entry['Processing_Status'] == 'discovered':
                 entry['Processing_Status'] = status_update; break

        if not filtered:
             qualified_channels.append(ch_meta)

    # --- INCREMENTAL SAVE 3: Save seen list after filtering ---
    save_seen_channels(seen_channels_data, seen_channels_path)

    if not qualified_channels:
         print("  No candidates passed pre-filtering.")
         return # Exit after processing the single target seed

    # --- Step 4: Selective Scoring (same as before) ---
    qualified_channels.sort(key=lambda x: x["subscribers"], reverse=True)
    candidates_to_score = qualified_channels[:MAX_CANDIDATES_TO_SCORE]
    print(f"\n  Scoring top {len(candidates_to_score)} qualified candidates (by subs)...")

    for idx_c, candidate in enumerate(candidates_to_score, 1):
        # ... (scoring logic: fetch_recent_videos, create_channel_fingerprint_llm, calculate_jaccard_similarity - same as before) ...
        candidate_start_time = time.time(); candidate_id = candidate["id"]; candidate_name = candidate["name"]
        print(f"\n    [{idx_c}/{len(candidates_to_score)}] Scoring: {candidate_name} ({candidate_id})")
        current_candidate_status = "failed_scoring" # Default
        score = 0.0 # Default score

        try:
            print(f"      Fetching {VIDEOS_PER_CANDIDATE} videos for scoring...")
            videos = fetch_recent_videos(candidate_id, max_results=VIDEOS_PER_CANDIDATE, filter_shorts=True)
            if not videos or len(videos) < 3: print(f"      ⚠️ Skipping scoring - Not enough videos ({len(videos)})."); current_candidate_status = "skipped_few_videos"; continue

            print(f"      Generating fingerprint using {MODEL_PROVIDER.upper()}..."); video_df_cand = pd.DataFrame(videos)
            candidate_keywords = create_channel_fingerprint_llm(video_df_cand, channel_name=candidate_name, model_type=MODEL_PROVIDER)
            if not candidate_keywords: print(f"      ⚠️ Skipping scoring - Failed fingerprint generation."); current_candidate_status = "failed_fingerprint"; continue

            # score = calculate_jaccard_similarity(seed_keywords, candidate_keywords) # Calculate score
            # print(f"      📊 Jaccard Similarity Score: {score:.3f}")

            # # Log results (same as before)
            # current_time_str = datetime.now().isoformat()
            # result_row = {"Seed_Channel_Name": seed_channel_name, "Seed_Channel_ID": seed_channel_id, "Discovered_Channel_Name": candidate_name, "Discovered_Channel_ID": candidate_id, "Discovered_Channel_URL": candidate["url"], "Discovered_Subs": candidate["subscribers"], "Similarity_Score": score, "Level": 1, "Timestamp": current_time_str}
            # all_results_list.append(result_row)
            # current_candidate_status = "scored"

            # if score >= SIMILARITY_THRESHOLD:
            #     print(f"      🎯 HIGH SIMILARITY! (Score >= {SIMILARITY_THRESHOLD})")
            #     high_similarity_results_list.append(result_row)
            #     current_candidate_status = "scored_high_similarity"

            # Calculate Score
            score = calculate_jaccard_similarity(seed_keywords, candidate_keywords)
            print(f"      📊 Jaccard Similarity Score: {score:.3f}")

            # Log Result
            current_time_str = datetime.now().isoformat()
            result_row = {
                "Seed_Channel_Name": seed_channel_name,
                "Seed_Channel_ID": seed_channel_id,
                "Discovered_Channel_Name": candidate_name,
                "Discovered_Channel_ID": candidate_id,
                "Discovered_Channel_URL": candidate["url"],
                "Discovered_Subs": candidate["subscribers"],
                "Similarity_Score": score,
                "Level": 1,
                "Timestamp": current_time_str,
                # --- ADDED LINE ---
                "Discovered_Keywords": candidate_keywords # Store the list
                # --- END ADDED LINE ---
            }
            all_results_list.append(result_row)
            current_candidate_status = "scored"

            # Check High Similarity
            if score >= SIMILARITY_THRESHOLD:
                    print(f"      🎯 HIGH SIMILARITY! (Score >= {SIMILARITY_THRESHOLD})")
                    # Also add keywords to the high similarity list if needed
                    high_similarity_results_list.append(result_row) # result_row now includes keywords
                    current_candidate_status = "scored_high_similarity"

            time.sleep(DELAY_BETWEEN_CANDIDATES) # Pace candidate scoring

        except Exception as score_err: print(f"      ❌ ERROR during scoring process: {score_err}"); current_candidate_status = "error_scoring"
        finally:
             # Update status in seen_channels_data after scoring attempt
             for entry in seen_channels_data:
                  if entry['Channel_ID'] == candidate_id:
                       entry['Processing_Status'] = current_candidate_status
                       entry['Last_Score'] = score # Add score info
                       break
             # --- INCREMENTAL SAVE 4: Save seen list after each candidate scored ---
             save_seen_channels(seen_channels_data, seen_channels_path)


    loop_end_time = time.time()
    print(f"\n  Finished processing seed '{seed_channel_name}' in {loop_end_time - loop_start_time:.2f} seconds.")

    # --- Saving Final Results (Only need to save all/high lists now) ---
    print("\n--- Saving Final Results ---")
    total_run_time = time.time() - start_time

    # Save All Scored Results
    if all_results_list:
        df_all = pd.DataFrame(all_results_list)
        try: df_all.to_csv(all_results_path, index=False, encoding='utf-8-sig'); print(f"✅ Saved {len(df_all)} scored results to {all_results_path.name}")
        except Exception as e: print(f"❌ Error saving {all_results_path.name}: {e}")
    else: print(f"ℹ️ No channels were successfully scored. '{all_results_path.name}' not created/updated.")

    # Save High Similarity Results
    if high_similarity_results_list:
        df_high = pd.DataFrame(high_similarity_results_list)
        try: df_high.to_csv(high_similarity_path, index=False, encoding='utf-8-sig'); print(f"✅ Saved {len(df_high)} high similarity channels to {high_similarity_path.name}")
        except Exception as e: print(f"❌ Error saving {high_similarity_path.name}: {e}")
    else: print(f"ℹ️ No high similarity channels found. '{high_similarity_path.name}' not created/updated.")

    # Final save of seen_channels is already handled incrementally

    print(f"\n--- Process Complete ---")
    print(f"Total Run Time: {total_run_time:.2f} seconds")
    # Updated summary for single channel run
    print(f"Processed 1 seed channel: {TARGET_SEED_CHANNEL}")
    print(f"Total candidates scored: {len(all_results_list)}")
    print(f"Total high similarity candidates: {len(high_similarity_results_list)}")


if __name__ == "__main__":
    main()