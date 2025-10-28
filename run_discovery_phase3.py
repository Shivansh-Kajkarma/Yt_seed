# # # /run_discover_score_log.py (Debug MagnatesMedia + Incremental Save)

# # import pandas as pd
# # import json
# # from pathlib import Path
# # import time
# # from datetime import datetime
# # import sys

# # # --- Import Helper Functions ---
# # from utils.youtube_utils import (
# #     # search_videos_focused, # Now uses OR
# #     search_videos_multi_focused,
# #     get_channel_metadata_batch,
# #     fetch_recent_videos
# # )
# # from utils.fingerprint_llm_utils import (
# #     create_channel_fingerprint_llm,
# #     # calculate_jaccard_similarity
# #     # calculate_phrase_similarity
# #     calculate_embedding_similarity_hybrid
# # )

# # # --- Configuration ---
# # base_dir = Path(__file__).resolve().parent
# # # Input Files
# # keywords_file_path = base_dir / "channel_keywords_simple_gemini_phrases.json"
# # seed_video_data_path = base_dir / "sample_videos_new.csv"
# # seen_channels_path = base_dir / "seen_channels.csv"
# # # Output Files
# # all_results_path = base_dir / "all_discovered_channels_phrases.csv"
# # high_similarity_path = base_dir / "high_similarity_channels_phrases.csv"
# # # LLM Choice & Target Channel for Debugging
# # MODEL_PROVIDER = "gemini" # Ensure this matches your keywords file
# # TARGET_SEED_CHANNEL = "MagnatesMedia" # <<<--- FOCUS ON THIS CHANNEL
# # # Thresholds & Limits
# # SIMILARITY_THRESHOLD = 0.3 # <<<--- LOWER THRESHOLD FOR DEBUGGING JACCARD
# # MIN_SUBSCRIBERS = 10000
# # MIN_VIDEOS = 10
# # MAX_CANDIDATES_TO_SCORE = 5
# # VIDEOS_PER_CANDIDATE = 10
# # # Delays (in seconds) - Keep them reasonable
# # DELAY_BETWEEN_SEEDS = 5
# # DELAY_BETWEEN_CANDIDATES = 3 # Slightly increased for LLM pacing

# # # --- Helper Function for Incremental Save ---
# # def save_seen_channels(seen_data_list, file_path):
# #     """Saves the seen channels data list to CSV."""
# #     if not seen_data_list:
# #         print("  (No seen channel data to save yet.)")
# #         return
# #     try:
# #         df_seen_updated = pd.DataFrame(seen_data_list)
# #         # Sort and remove duplicates just before saving
# #         df_seen_updated = df_seen_updated.sort_values(by="Date_Added").drop_duplicates(subset=['Channel_ID'], keep='last')
# #         df_seen_updated.to_csv(file_path, index=False, encoding='utf-8-sig')
# #         print(f"  💾 Saved {len(df_seen_updated)} seen channel entries to {file_path.name}")
# #     except Exception as e:
# #         print(f"  ❌ Error saving {file_path.name}: {e}")
# # # --- End Helper ---


# # # --- Main Logic ---
# # def main():
# #     start_time = time.time()
# #     print(f"--- Starting Discovery & Scoring Process (DEBUG MODE: {TARGET_SEED_CHANNEL}) ---")
# #     print(f"Using LLM Provider: {MODEL_PROVIDER.upper()}")

# #     # 1. Load Seed Keywords & Filter for Target
# #     # ... (loading keywords_file_path - same as before) ...
# #     try:
# #         with open(keywords_file_path, 'r', encoding='utf-8') as f:
# #             all_seed_keywords = json.load(f)
# #         # --- FILTER FOR TARGET CHANNEL ---
# #         seed_channel_keywords = {TARGET_SEED_CHANNEL: all_seed_keywords.get(TARGET_SEED_CHANNEL)}
# #         if not seed_channel_keywords[TARGET_SEED_CHANNEL]:
# #             print(f"❌ ERROR: Target channel '{TARGET_SEED_CHANNEL}' not found in keyword file or has no keywords."); return
# #         print(f"  Loaded keywords for target seed channel: {TARGET_SEED_CHANNEL}.")
# #     except Exception as e: print(f"❌ ERROR loading keywords: {e}"); return


# #     # 2. Load Seed Channel Name -> ID Mapping (same as before)
# #     # ... (loading seed_video_data_path - same as before) ...
# #     seed_name_to_id = {};
# #     try:
# #         df_videos = pd.read_csv(seed_video_data_path)
# #         if not all(col in df_videos.columns for col in ['Channel_Name', 'Channel_ID']): raise ValueError("CSV needs 'Channel_Name', 'Channel_ID'")
# #         seed_name_to_id = df_videos.drop_duplicates(subset=['Channel_Name'])[['Channel_Name', 'Channel_ID']].set_index('Channel_Name')['Channel_ID'].to_dict()
# #         print(f"  Created mapping for {len(seed_name_to_id)} unique seed channels found in video data.")
# #         if TARGET_SEED_CHANNEL not in seed_name_to_id:
# #              print(f"⚠️ WARNING: Target channel '{TARGET_SEED_CHANNEL}' ID not found in mapping CSV.")
# #              # Attempt to extract ID from seed_channels.csv maybe? For now, we'll let it fail later if ID is needed.
# #     except Exception as e: print(f"❌ ERROR processing seed video CSV: {e}"); return


# #     # 3. Load or Initialize Seen Channels (same as before)
# #     # ... (loading seen_channels_path - same as before) ...
# #     seen_channels_data = []; seen_ids = set()
# #     if seen_channels_path.exists():
# #         try:
# #             df_seen = pd.read_csv(seen_channels_path)
# #             if all(col in df_seen.columns for col in ['Channel_ID', 'Channel_Name', 'Date_Added', 'Processing_Status']):
# #                  seen_channels_data = df_seen.to_dict('records'); seen_ids = set(df_seen['Channel_ID'].astype(str).tolist())
# #                  print(f"  Loaded {len(seen_ids)} previously seen channel IDs.")
# #             else: print("⚠️ WARNING: seen_channels.csv columns incorrect. Starting fresh.")
# #         except Exception as e: print(f"⚠️ WARNING: Could not read seen_channels.csv: {e}. Starting fresh.")
# #     else: print("  seen_channels.csv not found. Starting fresh.")
# #     # Add target seed if not seen (same logic)
# #     current_time_str = datetime.now().isoformat()
# #     target_seed_id = seed_name_to_id.get(TARGET_SEED_CHANNEL)
# #     if target_seed_id and target_seed_id not in seen_ids:
# #          seen_channels_data.append({"Channel_ID": target_seed_id, "Channel_Name": TARGET_SEED_CHANNEL, "Date_Added": current_time_str, "Processing_Status": "seed"})
# #          seen_ids.add(target_seed_id)
# #          print(f"  Added target seed '{TARGET_SEED_CHANNEL}' to seen list.")
# #     print(f"  Total seen IDs before processing: {len(seen_ids)}")
# #     # --- INCREMENTAL SAVE 1: Save seen list after initialization ---
# #     save_seen_channels(seen_channels_data, seen_channels_path)


# #     # 4. Initialize Results Lists (same as before)
# #     all_results_list = []
# #     high_similarity_results_list = []

# #     # --- Main Loop (Now only runs for TARGET_SEED_CHANNEL) ---
# #     print("\n--- Starting Discovery & Scoring Loop ---")
# #     seed_channel_name = TARGET_SEED_CHANNEL
# #     seed_keywords = seed_channel_keywords[seed_channel_name]
# #     seed_channel_id = seed_name_to_id.get(seed_channel_name)

# #     if not seed_channel_id:
# #         print(f"❌ ERROR: Cannot proceed without Channel ID for '{seed_channel_name}'.")
# #         return # Exit if target seed ID is missing

# #     # Validate keywords (same as before)
# #     if not seed_keywords or seed_keywords == ["ERROR_PROCESSING"] or len(seed_keywords) < 3:
# #          print(f"  Skipping - Insufficient or invalid keywords for {seed_channel_name}.")
# #          return # Exit if target seed has bad keywords

# #     loop_start_time = time.time()
# #     print(f"\n[1/1] Processing Seed: {seed_channel_name}")

# #     # --- Step 1: Single Smart Search (Uses OR function now) ---
# #     try:
# #         # Calls the function which now uses " OR "
# #         candidate_ids_set = search_videos_multi_focused(seed_keywords, max_results_per_search=20)
# #         # candidate_ids_set = search_videos_focused(seed_keywords, max_results=20)
# #     except Exception as search_err:
# #          print(f"  ❌ Search failed for {seed_channel_name}: {search_err}. Stopping.")
# #          return

# #     # --- Step 2: Filter Seen & Get Metadata (same as before) ---
# #     candidate_ids_to_fetch = list(candidate_ids_set - seen_ids - {seed_channel_id})
# #     if not candidate_ids_to_fetch:
# #         print("  No new candidate channels found after filtering seen IDs.")
# #         # Save seen_channels again even if no candidates found for this seed
# #         save_seen_channels(seen_channels_data, seen_channels_path)
# #         return # Exit after processing the single target seed

# #     print(f"  Fetching metadata for {len(candidate_ids_to_fetch)} new candidates...")
# #     candidate_metadata_list = get_channel_metadata_batch(candidate_ids_to_fetch)

# #     # Update seen_channels_data (same as before)
# #     current_time_str = datetime.now().isoformat()
# #     new_discovered_count = 0
# #     for meta in candidate_metadata_list:
# #          if meta['id'] not in seen_ids:
# #               seen_channels_data.append({"Channel_ID": meta['id'], "Channel_Name": meta['name'], "Date_Added": current_time_str, "Processing_Status": "discovered"})
# #               seen_ids.add(meta['id'])
# #               new_discovered_count += 1
# #     if new_discovered_count > 0:
# #         print(f"  Added {new_discovered_count} newly discovered channels to seen list.")
# #         # --- INCREMENTAL SAVE 2: Save seen list after adding discovered ---
# #         save_seen_channels(seen_channels_data, seen_channels_path)

# #     # --- Step 3: Pre-Filter (same as before) ---
# #     qualified_channels = []
# #     # ... (filtering logic same as before, updating seen_channels_data status) ...
# #     for ch_meta in candidate_metadata_list:
# #         filtered = False
# #         status_update = 'discovered' # Keep default status if qualified
# #         if ch_meta['subscribers'] != -1 and ch_meta['subscribers'] < MIN_SUBSCRIBERS:
# #             print(f"    - Filtering {ch_meta['name']} (Subs: {ch_meta['subscribers']} < {MIN_SUBSCRIBERS})")
# #             status_update = 'filtered_subs'; filtered = True
# #         elif ch_meta['video_count'] < MIN_VIDEOS:
# #             print(f"    - Filtering {ch_meta['name']} (Videos: {ch_meta['video_count']} < {MIN_VIDEOS})")
# #             status_update = 'filtered_videos'; filtered = True

# #         # Update status in seen_channels_data for ALL discovered channels (filtered or not)
# #         for entry in seen_channels_data:
# #             if entry['Channel_ID'] == ch_meta['id'] and entry['Processing_Status'] == 'discovered':
# #                  entry['Processing_Status'] = status_update; break

# #         if not filtered:
# #              qualified_channels.append(ch_meta)

# #     # --- INCREMENTAL SAVE 3: Save seen list after filtering ---
# #     save_seen_channels(seen_channels_data, seen_channels_path)

# #     if not qualified_channels:
# #          print("  No candidates passed pre-filtering.")
# #          return # Exit after processing the single target seed

# #     # --- Step 4: Selective Scoring (same as before) ---
# #     qualified_channels.sort(key=lambda x: x["subscribers"], reverse=True)
# #     candidates_to_score = qualified_channels[:MAX_CANDIDATES_TO_SCORE]
# #     print(f"\n  Scoring top {len(candidates_to_score)} qualified candidates (by subs)...")

# #     for idx_c, candidate in enumerate(candidates_to_score, 1):
# #         # ... (scoring logic: fetch_recent_videos, create_channel_fingerprint_llm, calculate_jaccard_similarity - same as before) ...
# #         candidate_start_time = time.time(); candidate_id = candidate["id"]; candidate_name = candidate["name"]
# #         print(f"\n    [{idx_c}/{len(candidates_to_score)}] Scoring: {candidate_name} ({candidate_id})")
# #         current_candidate_status = "failed_scoring" # Default
# #         score = 0.0 # Default score

# #         try:
# #             print(f"      Fetching {VIDEOS_PER_CANDIDATE} videos for scoring...")
# #             videos = fetch_recent_videos(candidate_id, max_results=VIDEOS_PER_CANDIDATE, filter_shorts=True)
# #             if not videos or len(videos) < 3: print(f"      ⚠️ Skipping scoring - Not enough videos ({len(videos)})."); current_candidate_status = "skipped_few_videos"; continue

# #             print(f"      Generating fingerprint using {MODEL_PROVIDER.upper()}..."); video_df_cand = pd.DataFrame(videos)
# #             candidate_keywords = create_channel_fingerprint_llm(video_df_cand, channel_name=candidate_name, model_type=MODEL_PROVIDER)
# #             if not candidate_keywords: print(f"      ⚠️ Skipping scoring - Failed fingerprint generation."); current_candidate_status = "failed_fingerprint"; continue

# #             # score = calculate_jaccard_similarity(seed_keywords, candidate_keywords) # Calculate score
# #             # print(f"      📊 Jaccard Similarity Score: {score:.3f}")

# #             # # Log results (same as before)
# #             # current_time_str = datetime.now().isoformat()
# #             # result_row = {"Seed_Channel_Name": seed_channel_name, "Seed_Channel_ID": seed_channel_id, "Discovered_Channel_Name": candidate_name, "Discovered_Channel_ID": candidate_id, "Discovered_Channel_URL": candidate["url"], "Discovered_Subs": candidate["subscribers"], "Similarity_Score": score, "Level": 1, "Timestamp": current_time_str}
# #             # all_results_list.append(result_row)
# #             # current_candidate_status = "scored"

# #             # if score >= SIMILARITY_THRESHOLD:
# #             #     print(f"      🎯 HIGH SIMILARITY! (Score >= {SIMILARITY_THRESHOLD})")
# #             #     high_similarity_results_list.append(result_row)
# #             #     current_candidate_status = "scored_high_similarity"

# #             # Calculate Score
# #             # score = calculate_jaccard_similarity(seed_keywords, candidate_keywords)
# #             # score = calculate_phrase_similarity(seed_keywords, candidate_keywords)
# #             score = calculate_embedding_similarity_hybrid(seed_keywords, candidate_keywords)
# #             print(f"      📊 Jaccard Similarity Score: {score:.3f}")

# #             # Log Result
# #             current_time_str = datetime.now().isoformat()
# #             result_row = {
# #                 "Seed_Channel_Name": seed_channel_name,
# #                 "Seed_Channel_ID": seed_channel_id,
# #                 "Discovered_Channel_Name": candidate_name,
# #                 "Discovered_Channel_ID": candidate_id,
# #                 "Discovered_Channel_URL": candidate["url"],
# #                 "Discovered_Subs": candidate["subscribers"],
# #                 "Similarity_Score": score,
# #                 "Level": 1,
# #                 "Timestamp": current_time_str,
# #                 # --- ADDED LINE ---
# #                 "Discovered_Keywords": candidate_keywords # Store the list
# #                 # --- END ADDED LINE ---
# #             }
# #             all_results_list.append(result_row)
# #             current_candidate_status = "scored"

# #             # Check High Similarity
# #             if score >= SIMILARITY_THRESHOLD:
# #                     print(f"      🎯 HIGH SIMILARITY! (Score >= {SIMILARITY_THRESHOLD})")
# #                     # Also add keywords to the high similarity list if needed
# #                     high_similarity_results_list.append(result_row) # result_row now includes keywords
# #                     current_candidate_status = "scored_high_similarity"

# #             time.sleep(DELAY_BETWEEN_CANDIDATES) # Pace candidate scoring

# #         except Exception as score_err: print(f"      ❌ ERROR during scoring process: {score_err}"); current_candidate_status = "error_scoring"
# #         finally:
# #              # Update status in seen_channels_data after scoring attempt
# #              for entry in seen_channels_data:
# #                   if entry['Channel_ID'] == candidate_id:
# #                        entry['Processing_Status'] = current_candidate_status
# #                        entry['Last_Score'] = score # Add score info
# #                        break
# #              # --- INCREMENTAL SAVE 4: Save seen list after each candidate scored ---
# #              save_seen_channels(seen_channels_data, seen_channels_path)


# #     loop_end_time = time.time()
# #     print(f"\n  Finished processing seed '{seed_channel_name}' in {loop_end_time - loop_start_time:.2f} seconds.")

# #     # --- Saving Final Results (Only need to save all/high lists now) ---
# #     print("\n--- Saving Final Results ---")
# #     total_run_time = time.time() - start_time

# #     # Save All Scored Results
# #     if all_results_list:
# #         df_all = pd.DataFrame(all_results_list)
# #         try: df_all.to_csv(all_results_path, index=False, encoding='utf-8-sig'); print(f"✅ Saved {len(df_all)} scored results to {all_results_path.name}")
# #         except Exception as e: print(f"❌ Error saving {all_results_path.name}: {e}")
# #     else: print(f"ℹ️ No channels were successfully scored. '{all_results_path.name}' not created/updated.")

# #     # Save High Similarity Results
# #     if high_similarity_results_list:
# #         df_high = pd.DataFrame(high_similarity_results_list)
# #         try: df_high.to_csv(high_similarity_path, index=False, encoding='utf-8-sig'); print(f"✅ Saved {len(df_high)} high similarity channels to {high_similarity_path.name}")
# #         except Exception as e: print(f"❌ Error saving {high_similarity_path.name}: {e}")
# #     else: print(f"ℹ️ No high similarity channels found. '{high_similarity_path.name}' not created/updated.")

# #     # Final save of seen_channels is already handled incrementally

# #     print(f"\n--- Process Complete ---")
# #     print(f"Total Run Time: {total_run_time:.2f} seconds")
# #     # Updated summary for single channel run
# #     print(f"Processed 1 seed channel: {TARGET_SEED_CHANNEL}")
# #     print(f"Total candidates scored: {len(all_results_list)}")
# #     print(f"Total high similarity candidates: {len(high_similarity_results_list)}")


# # if __name__ == "__main__":
# #     main()



# # /run_discover_score_log.py (Production Ready - Multi-Search + Enhanced CSV)

# import pandas as pd
# import json
# from pathlib import Path
# import time
# from datetime import datetime

# # --- Import Helper Functions ---
# from utils.youtube_utils import (
#     search_videos_multi_focused,  # Optimized 3-search strategy
#     get_channel_metadata_batch,
#     fetch_recent_videos
# )
# from utils.fingerprint_llm_utils import (
#     create_channel_fingerprint_llm,
#     calculate_embedding_similarity_hybrid
# )

# # === CONFIGURATION ===
# base_dir = Path(__file__).resolve().parent

# # Input Files
# keywords_file_path = base_dir / "channel_keywords_simple_gemini_phrases.json"
# seed_video_data_path = base_dir / "sample_videos_new.csv"
# seen_channels_path = base_dir / "seen_channels.csv"

# # Output Files  
# all_results_path = base_dir / "all_discovered_channels_new_per.csv"
# high_similarity_path = base_dir / "high_similarity_channels_new_per.csv"

# # Settings
# MODEL_PROVIDER = "gemini"  # "gemini" or "openai"
# TARGET_SEED_CHANNEL = "MagnatesMedia"  # Change to test different channels

# # Thresholds
# SIMILARITY_THRESHOLD = 0.6  # For embeddings (0.3-0.4 is good)
# MIN_SUBSCRIBERS = 10000
# MIN_VIDEOS = 10
# MAX_CANDIDATES_TO_SCORE = 12
# VIDEOS_PER_CANDIDATE = 10

# # Rate Limiting
# DELAY_BETWEEN_CANDIDATES = 3  # seconds
# DELAY_BETWEEN_SEEDS = 5  # seconds

# # === HELPER FUNCTIONS ===

# def save_seen_channels(seen_data_list, file_path):
#     """Saves seen channels to CSV with deduplication"""
#     if not seen_data_list:
#         print("  (No seen channel data to save)")
#         return
    
#     try:
#         df = pd.DataFrame(seen_data_list)
#         # Remove duplicates, keep latest
#         df = df.sort_values(by="Date_Added").drop_duplicates(
#             subset=['Channel_ID'], 
#             keep='last'
#         )
#         df.to_csv(file_path, index=False, encoding='utf-8-sig')
#         print(f"  💾 Saved {len(df)} seen channels to {file_path.name}")
#     except Exception as e:
#         print(f"  ❌ Error saving {file_path.name}: {e}")

# def load_seen_channels(file_path):
#     """Loads seen channels from CSV"""
#     seen_data = []
#     seen_ids = set()
    
#     if file_path.exists():
#         try:
#             df = pd.read_csv(file_path)
#             required_cols = ['Channel_ID', 'Channel_Name', 'Date_Added', 'Processing_Status']
            
#             if all(col in df.columns for col in required_cols):
#                 seen_data = df.to_dict('records')
#                 seen_ids = set(df['Channel_ID'].astype(str).tolist())
#                 print(f"  📂 Loaded {len(seen_ids)} previously seen channels")
#             else:
#                 print("  ⚠️  Seen channels CSV has wrong columns. Starting fresh.")
#         except Exception as e:
#             print(f"  ⚠️  Could not read seen channels: {e}")
#     else:
#         print("  📂 No existing seen channels file. Starting fresh.")
    
#     return seen_data, seen_ids

# # === MAIN FUNCTION ===

# def main():
#     start_time = time.time()
#     print("=" * 70)
#     print(f"PHASE 3+4 DISCOVERY & SCORING")
#     print(f"Target Channel: {TARGET_SEED_CHANNEL}")
#     print(f"LLM Provider: {MODEL_PROVIDER.upper()}")
#     print("=" * 70)
    
#     # --- 1. Load Keywords ---
#     print(f"\n📖 Loading keywords from {keywords_file_path.name}...")
    
#     try:
#         with open(keywords_file_path, 'r', encoding='utf-8') as f:
#             all_keywords = json.load(f)
        
#         if TARGET_SEED_CHANNEL not in all_keywords:
#             print(f"❌ ERROR: '{TARGET_SEED_CHANNEL}' not found in keywords file")
#             return
        
#         seed_keywords = all_keywords[TARGET_SEED_CHANNEL]
#         print(f"  ✅ Loaded {len(seed_keywords)} keywords for {TARGET_SEED_CHANNEL}")
#         print(f"     Keywords: {seed_keywords[:5]}...")
        
#     except Exception as e:
#         print(f"❌ ERROR loading keywords: {e}")
#         return
    
#     # --- 2. Load Seed Channel ID Mapping ---
#     print(f"\n🗺️  Loading channel IDs from {seed_video_data_path.name}...")
    
#     try:
#         df_videos = pd.read_csv(seed_video_data_path)
        
#         if 'Channel_Name' not in df_videos.columns or 'Channel_ID' not in df_videos.columns:
#             print("❌ ERROR: CSV must have 'Channel_Name' and 'Channel_ID' columns")
#             return
        
#         seed_id_map = df_videos.drop_duplicates(subset=['Channel_Name'])[
#             ['Channel_Name', 'Channel_ID']
#         ].set_index('Channel_Name')['Channel_ID'].to_dict()
        
#         seed_channel_id = seed_id_map.get(TARGET_SEED_CHANNEL)
        
#         if not seed_channel_id:
#             print(f"❌ ERROR: No Channel ID found for '{TARGET_SEED_CHANNEL}'")
#             return
        
#         print(f"  ✅ Found Channel ID: {seed_channel_id}")
        
#     except Exception as e:
#         print(f"❌ ERROR loading channel IDs: {e}")
#         return
    
#     # --- 3. Load Seen Channels ---
#     print(f"\n📂 Loading seen channels from {seen_channels_path.name}...")
    
#     seen_channels_data, seen_ids = load_seen_channels(seen_channels_path)
    
#     # Add seed channel if not already seen
#     current_time = datetime.now().isoformat()
    
#     if seed_channel_id not in seen_ids:
#         seen_channels_data.append({
#             "Channel_ID": seed_channel_id,
#             "Channel_Name": TARGET_SEED_CHANNEL,
#             "Date_Added": current_time,
#             "Processing_Status": "seed",
#             "Last_Score": None
#         })
#         seen_ids.add(seed_channel_id)
#         print(f"  ✅ Added seed '{TARGET_SEED_CHANNEL}' to seen list")
    
#     save_seen_channels(seen_channels_data, seen_channels_path)
    
#     # --- 4. Initialize Results ---
#     all_results = []
#     high_similarity_results = []
    
#     # === MAIN PROCESSING ===
#     print("\n" + "=" * 70)
#     print(f"PROCESSING SEED: {TARGET_SEED_CHANNEL}")
#     print("=" * 70)
    
#     # --- STEP 1: Multi-Focused Search ---
#     print(f"\n🔍 STEP 1: Searching YouTube (3 focused searches)...")
    
#     try:
#         candidate_ids = search_videos_multi_focused(
#             seed_keywords, 
#             max_results_per_search=10
#         )
#     except Exception as e:
#         print(f"❌ Search failed: {e}")
#         return
    
#     # Remove seed and already seen channels
#     candidate_ids = candidate_ids - seen_ids - {seed_channel_id}
    
#     if not candidate_ids:
#         print("  ⚠️  No new candidates found after filtering")
#         return
    
#     print(f"  ✅ {len(candidate_ids)} new candidates to evaluate")
    
#     # --- STEP 2: Get Metadata ---
#     print(f"\n📊 STEP 2: Fetching channel metadata...")
    
#     metadata = get_channel_metadata_batch(list(candidate_ids))
    
#     # Add to seen list
#     for meta in metadata:
#         if meta['id'] not in seen_ids:
#             seen_channels_data.append({
#                 "Channel_ID": meta['id'],
#                 "Channel_Name": meta['name'],
#                 "Date_Added": current_time,
#                 "Processing_Status": "discovered",
#                 "Last_Score": None
#             })
#             seen_ids.add(meta['id'])
    
#     save_seen_channels(seen_channels_data, seen_channels_path)
    
#     # --- STEP 3: Pre-Filter ---
#     print(f"\n🔍 STEP 3: Pre-filtering by subscribers and videos...")
    
#     qualified = []
    
#     for meta in metadata:
#         # Check subscriber count
#         if meta['subscribers'] == -1:
#             print(f"  - Filtering {meta['name']} (subscribers hidden)")
#             _update_status(seen_channels_data, meta['id'], 'filtered_hidden_subs')
#             continue
        
#         if meta['subscribers'] < MIN_SUBSCRIBERS:
#             print(f"  - Filtering {meta['name']} (subs: {meta['subscribers']:,})")
#             _update_status(seen_channels_data, meta['id'], 'filtered_subs')
#             continue
        
#         # Check video count
#         if meta['video_count'] < MIN_VIDEOS:
#             print(f"  - Filtering {meta['name']} (videos: {meta['video_count']})")
#             _update_status(seen_channels_data, meta['id'], 'filtered_videos')
#             continue
        
#         qualified.append(meta)
    
#     save_seen_channels(seen_channels_data, seen_channels_path)
    
#     if not qualified:
#         print("  ⚠️  No candidates passed pre-filtering")
#         return
    
#     # Sort by subscribers
#     qualified.sort(key=lambda x: x['subscribers'], reverse=True)
#     to_score = qualified
#     # to_score = qualified[:MAX_CANDIDATES_TO_SCORE]
    
#     print(f"  ✅ {len(qualified)} qualified, scoring top {len(to_score)}")
    
#     # --- STEP 4: Score Candidates ---
#     print(f"\n🎯 STEP 4: Scoring candidates...")
    
#     for i, candidate in enumerate(to_score, 1):
#         print(f"\n  [{i}/{len(to_score)}] {candidate['name']}")
#         print(f"     Subs: {candidate['subscribers']:,}")
        
#         try:
#             # Fetch videos
#             print(f"     Fetching {VIDEOS_PER_CANDIDATE} videos...")
#             videos = fetch_recent_videos(
#                 candidate['id'],
#                 max_results=VIDEOS_PER_CANDIDATE,
#                 filter_shorts=False  # Don't filter - some good channels use shorts
#             )
            
#             if len(videos) < 3:
#                 print(f"     ⚠️  Only {len(videos)} videos found, skipping")
#                 _update_status(seen_channels_data, candidate['id'], 'skipped_few_videos')
#                 continue
            
#             # Generate fingerprint
#             print(f"     Generating keywords with {MODEL_PROVIDER.upper()}...")
#             video_df = pd.DataFrame(videos)
            
#             cand_keywords = create_channel_fingerprint_llm(
#                 video_df,
#                 channel_name=candidate['name'],
#                 model_type=MODEL_PROVIDER
#             )
            
#             if not cand_keywords:
#                 print(f"     ⚠️  Keyword generation failed")
#                 _update_status(seen_channels_data, candidate['id'], 'failed_fingerprint')
#                 continue
            
#             print(f"     Keywords: {cand_keywords[:3]}...")
            
#             # Calculate similarity
#             similarity = calculate_embedding_similarity_hybrid(
#                 seed_keywords,
#                 cand_keywords
#             )
            
#             print(f"     📊 Similarity: {similarity:.3f}")
            
#             # Store result
#             result = {
#                 "Seed_Channel_Name": TARGET_SEED_CHANNEL,
#                 "Seed_Channel_ID": seed_channel_id,
#                 "Seed_Keywords": ", ".join(seed_keywords),  # Added for analysis
#                 "Discovered_Channel_Name": candidate['name'],
#                 "Discovered_Channel_ID": candidate['id'],
#                 "Discovered_Channel_URL": candidate['url'],
#                 "Discovered_Subs": candidate['subscribers'],
#                 "Discovered_Keywords": ", ".join(cand_keywords),  # Added for analysis
#                 "Similarity_Score": similarity,
#                 "Level": 1,
#                 "Timestamp": datetime.now().isoformat()
#             }
            
#             all_results.append(result)
            
#             # Check if high similarity
#             if similarity >= SIMILARITY_THRESHOLD:
#                 print(f"     ✅ HIGH SIMILARITY (>= {SIMILARITY_THRESHOLD})!")
#                 high_similarity_results.append(result)
#                 _update_status_with_score(
#                     seen_channels_data, 
#                     candidate['id'], 
#                     'scored_high_similarity',
#                     similarity
#                 )
#             else:
#                 _update_status_with_score(
#                     seen_channels_data,
#                     candidate['id'],
#                     'scored',
#                     similarity
#                 )
            
#             save_seen_channels(seen_channels_data, seen_channels_path)
#             time.sleep(DELAY_BETWEEN_CANDIDATES)
            
#         except Exception as e:
#             print(f"     ❌ Error: {str(e)[:100]}")
#             _update_status(seen_channels_data, candidate['id'], 'error_scoring')
    
#     # === SAVE FINAL RESULTS ===
#     print("\n" + "=" * 70)
#     print("SAVING RESULTS")
#     print("=" * 70)
    
#     # Save all results
#     if all_results:
#         df_all = pd.DataFrame(all_results)
#         df_all.to_csv(all_results_path, index=False, encoding='utf-8-sig')
#         print(f"✅ Saved {len(all_results)} results to {all_results_path.name}")
#     else:
#         print("ℹ️  No results to save")
    
#     # Save high similarity
#     if high_similarity_results:
#         df_high = pd.DataFrame(high_similarity_results)
#         df_high.to_csv(high_similarity_path, index=False, encoding='utf-8-sig')
#         print(f"✅ Saved {len(high_similarity_results)} high-similarity to {high_similarity_path.name}")
#     else:
#         print("ℹ️  No high-similarity results")
    
#     # === SUMMARY ===
#     elapsed = time.time() - start_time
    
#     print("\n" + "=" * 70)
#     print("SUMMARY")
#     print("=" * 70)
#     print(f"Seed processed: {TARGET_SEED_CHANNEL}")
#     print(f"Candidates scored: {len(all_results)}")
#     print(f"High similarity: {len(high_similarity_results)}")
#     print(f"Total runtime: {elapsed:.1f}s")
#     print("=" * 70)

# # === HELPER FUNCTIONS ===

# def _update_status(seen_data, channel_id, status):
#     """Update processing status for a channel"""
#     for entry in seen_data:
#         if entry['Channel_ID'] == channel_id:
#             entry['Processing_Status'] = status
#             break

# def _update_status_with_score(seen_data, channel_id, status, score):
#     """Update status and score"""
#     for entry in seen_data:
#         if entry['Channel_ID'] == channel_id:
#             entry['Processing_Status'] = status
#             entry['Last_Score'] = score
#             break

# if __name__ == "__main__":
#     main()


# ----------------working above


# # /run_discover_score_ab_test.py (A/B Testing: Embeddings vs LLM)

# import pandas as pd
# import json
# from pathlib import Path
# import time
# from datetime import datetime

# # --- Import Helper Functions ---
# from utils.youtube_utils import (
#     search_videos_multi_focused,
#     get_channel_metadata_batch,
#     fetch_recent_videos
# )
# from utils.fingerprint_llm_utils import (
#     create_channel_fingerprint_llm,
#     calculate_embedding_similarity_hybrid,
#     calculate_llm_similarity  # NEW: Add this function to fingerprint_llm_utils.py
# )

# # === CONFIGURATION ===
# base_dir = Path(__file__).resolve().parent

# # Input Files
# keywords_file_path = base_dir / "channel_keywords_simple_gemini_phrases.json"
# seed_video_data_path = base_dir / "sample_videos_new.csv"
# seen_channels_path = base_dir / "seen_channels.csv"

# # Output Files (4 files for A/B testing)
# all_results_embeddings_path = base_dir / "all_discovered_embeddings_26.csv"
# high_similarity_embeddings_path = base_dir / "high_similarity_embeddings_26.csv"
# all_results_llm_path = base_dir / "all_discovered_llm_26.csv"
# high_similarity_llm_path = base_dir / "high_similarity_llm_26.csv"

# # Settings
# MODEL_PROVIDER = "gpt"  # "gemini" or "openai"
# TARGET_SEED_CHANNEL = "MagnatesMedia"
# SEED_CHANNELS = [
#     "MagnatesMedia",
#     "TechnicalGuruji", 
#     "Business Inspection BD",
#     "Veritasium"
# ]

# # Thresholds (UPDATED per manager's request)
# SIMILARITY_THRESHOLD_EMBEDDINGS = 0.75  # Increased from 0.3
# SIMILARITY_THRESHOLD_LLM = 0.7         # LLM is more accurate, can be stricter
# MIN_SUBSCRIBERS = 10000
# MIN_VIDEOS = 10
# VIDEOS_PER_CANDIDATE = 10

# # Rate Limiting
# DELAY_BETWEEN_CANDIDATES = 3
# DELAY_BETWEEN_SEEDS = 5

# # === HELPER FUNCTIONS ===

# def save_seen_channels(seen_data_list, file_path):
#     """Saves seen channels to CSV with deduplication"""
#     if not seen_data_list:
#         print("  (No seen channel data to save)")
#         return
    
#     try:
#         df = pd.DataFrame(seen_data_list)
#         df = df.sort_values(by="Date_Added").drop_duplicates(
#             subset=['Channel_ID'], 
#             keep='last'
#         )
#         df.to_csv(file_path, index=False, encoding='utf-8-sig')
#         print(f"  💾 Saved {len(df)} seen channels to {file_path.name}")
#     except Exception as e:
#         print(f"  ❌ Error saving {file_path.name}: {e}")

# def load_seen_channels(file_path):
#     """Loads seen channels from CSV"""
#     seen_data = []
#     seen_ids = set()
    
#     if file_path.exists():
#         try:
#             df = pd.read_csv(file_path)
#             required_cols = ['Channel_ID', 'Channel_Name', 'Date_Added', 'Processing_Status']
            
#             if all(col in df.columns for col in required_cols):
#                 seen_data = df.to_dict('records')
#                 seen_ids = set(df['Channel_ID'].astype(str).tolist())
#                 print(f"  📂 Loaded {len(seen_ids)} previously seen channels")
#             else:
#                 print("  ⚠️  Seen channels CSV has wrong columns. Starting fresh.")
#         except Exception as e:
#             print(f"  ⚠️  Could not read seen channels: {e}")
#     else:
#         print("  📂 No existing seen channels file. Starting fresh.")
    
#     return seen_data, seen_ids

# def _update_status(seen_data, channel_id, status):
#     """Update processing status for a channel"""
#     for entry in seen_data:
#         if entry['Channel_ID'] == channel_id:
#             entry['Processing_Status'] = status
#             break

# def _update_status_with_scores(seen_data, channel_id, status, score_emb, score_llm):
#     """Update status with BOTH scores"""
#     for entry in seen_data:
#         if entry['Channel_ID'] == channel_id:
#             entry['Processing_Status'] = status
#             entry['Embedding_Score'] = score_emb
#             entry['LLM_Score'] = score_llm
#             break

# # === MAIN FUNCTION ===

# def main():
#     start_time = time.time()
#     print("=" * 70)
#     print(f"A/B TESTING: EMBEDDINGS vs LLM SCORING")
#     print(f"Target Channel: {TARGET_SEED_CHANNEL}")
#     print(f"LLM Provider: {MODEL_PROVIDER.upper()}")
#     print("=" * 70)
#     print(f"📊 Embedding Threshold: {SIMILARITY_THRESHOLD_EMBEDDINGS}")
#     print(f"🤖 LLM Threshold: {SIMILARITY_THRESHOLD_LLM}")
#     print(f"⚙️  Scoring: ALL qualified candidates (no limit)")
    
#     # --- 1. Load Keywords ---
#     print(f"\n📖 Loading keywords from {keywords_file_path.name}...")
    
#     try:
#         with open(keywords_file_path, 'r', encoding='utf-8') as f:
#             all_keywords = json.load(f)
        
#         if TARGET_SEED_CHANNEL not in all_keywords:
#             print(f"❌ ERROR: '{TARGET_SEED_CHANNEL}' not found in keywords file")
#             return
        
#         seed_keywords = all_keywords[TARGET_SEED_CHANNEL]
#         print(f"  ✅ Loaded {len(seed_keywords)} keywords for {TARGET_SEED_CHANNEL}")
#         print(f"     Keywords: {seed_keywords[:5]}...")
        
#     except Exception as e:
#         print(f"❌ ERROR loading keywords: {e}")
#         return
    
#     # --- 2. Load Seed Channel ID Mapping ---
#     print(f"\n🗺️  Loading channel IDs from {seed_video_data_path.name}...")
    
#     try:
#         df_videos = pd.read_csv(seed_video_data_path)
        
#         if 'Channel_Name' not in df_videos.columns or 'Channel_ID' not in df_videos.columns:
#             print("❌ ERROR: CSV must have 'Channel_Name' and 'Channel_ID' columns")
#             return
        
#         seed_id_map = df_videos.drop_duplicates(subset=['Channel_Name'])[
#             ['Channel_Name', 'Channel_ID']
#         ].set_index('Channel_Name')['Channel_ID'].to_dict()
        
#         seed_channel_id = seed_id_map.get(TARGET_SEED_CHANNEL)
        
#         if not seed_channel_id:
#             print(f"❌ ERROR: No Channel ID found for '{TARGET_SEED_CHANNEL}'")
#             return
        
#         print(f"  ✅ Found Channel ID: {seed_channel_id}")
        
#     except Exception as e:
#         print(f"❌ ERROR loading channel IDs: {e}")
#         return
    
#     # --- 3. Load Seen Channels ---
#     print(f"\n📂 Loading seen channels from {seen_channels_path.name}...")
    
#     seen_channels_data, seen_ids = load_seen_channels(seen_channels_path)
    
#     current_time = datetime.now().isoformat()
    
#     if seed_channel_id not in seen_ids:
#         seen_channels_data.append({
#             "Channel_ID": seed_channel_id,
#             "Channel_Name": TARGET_SEED_CHANNEL,
#             "Date_Added": current_time,
#             "Processing_Status": "seed",
#             "Embedding_Score": None,
#             "LLM_Score": None
#         })
#         seen_ids.add(seed_channel_id)
#         print(f"  ✅ Added seed '{TARGET_SEED_CHANNEL}' to seen list")
    
#     save_seen_channels(seen_channels_data, seen_channels_path)
    
#     # --- 4. Initialize Results (BOTH methods) ---
#     all_results_embeddings = []
#     high_similarity_embeddings = []
#     all_results_llm = []
#     high_similarity_llm = []
    
#     # === MAIN PROCESSING ===
#     print("\n" + "=" * 70)
#     print(f"PROCESSING SEED: {TARGET_SEED_CHANNEL}")
#     print("=" * 70)
    
#     # --- STEP 1: Multi-Focused Search ---
#     print(f"\n🔍 STEP 1: Searching YouTube...")
    
#     try:
#         candidate_ids = search_videos_multi_focused(
#             seed_keywords, 
#             max_results_per_search=20,
#             max_keywords=10
#         )
#     except Exception as e:
#         print(f"❌ Search failed: {e}")
#         return
    
#     candidate_ids = candidate_ids - seen_ids - {seed_channel_id}
    
#     if not candidate_ids:
#         print("  ⚠️  No new candidates found after filtering")
#         return
    
#     print(f"  ✅ {len(candidate_ids)} new candidates to evaluate")
    
#     # --- STEP 2: Get Metadata ---
#     print(f"\n📊 STEP 2: Fetching channel metadata...")
    
#     metadata = get_channel_metadata_batch(list(candidate_ids))
    
#     for meta in metadata:
#         if meta['id'] not in seen_ids:
#             seen_channels_data.append({
#                 "Channel_ID": meta['id'],
#                 "Channel_Name": meta['name'],
#                 "Date_Added": current_time,
#                 "Processing_Status": "discovered",
#                 "Embedding_Score": None,
#                 "LLM_Score": None
#             })
#             seen_ids.add(meta['id'])
    
#     save_seen_channels(seen_channels_data, seen_channels_path)
    
#     # --- STEP 3: Pre-Filter ---
#     print(f"\n🔍 STEP 3: Pre-filtering by subscribers and videos...")
    
#     qualified = []
    
#     for meta in metadata:
#         if meta['subscribers'] == -1:
#             print(f"  - Filtering {meta['name']} (subscribers hidden)")
#             _update_status(seen_channels_data, meta['id'], 'filtered_hidden_subs')
#             continue
        
#         if meta['subscribers'] < MIN_SUBSCRIBERS:
#             print(f"  - Filtering {meta['name']} (subs: {meta['subscribers']:,})")
#             _update_status(seen_channels_data, meta['id'], 'filtered_subs')
#             continue
        
#         if meta['video_count'] < MIN_VIDEOS:
#             print(f"  - Filtering {meta['name']} (videos: {meta['video_count']})")
#             _update_status(seen_channels_data, meta['id'], 'filtered_videos')
#             continue
        
#         qualified.append(meta)
    
#     save_seen_channels(seen_channels_data, seen_channels_path)
    
#     if not qualified:
#         print("  ⚠️  No candidates passed pre-filtering")
#         return
    
#     # Sort by subscribers (but score ALL of them!)
#     qualified.sort(key=lambda x: x['subscribers'], reverse=True)
#     to_score = qualified  # SCORE ALL (no limit!)
    
#     print(f"  ✅ {len(qualified)} qualified, scoring ALL {len(to_score)} candidates")
    
#     # --- STEP 4: Score Candidates (BOTH methods) ---
#     print(f"\n🎯 STEP 4: Scoring candidates with BOTH methods...")
#     print(f"   (This will take ~{len(to_score) * 10} seconds)\n")
    
#     for i, candidate in enumerate(to_score, 1):
#         print(f"  [{i}/{len(to_score)}] {candidate['name']}")
#         print(f"     Subs: {candidate['subscribers']:,}")
        
#         try:
#             # Fetch videos
#             print(f"     Fetching {VIDEOS_PER_CANDIDATE} videos...")
#             videos = fetch_recent_videos(
#                 candidate['id'],
#                 max_results=VIDEOS_PER_CANDIDATE,
#                 filter_shorts=False
#             )
            
#             if len(videos) < 3:
#                 print(f"     ⚠️  Only {len(videos)} videos found, skipping")
#                 _update_status(seen_channels_data, candidate['id'], 'skipped_few_videos')
#                 continue
            
#             # Generate fingerprint
#             print(f"     Generating keywords with {MODEL_PROVIDER.upper()}...")
#             video_df = pd.DataFrame(videos)
            
#             cand_keywords = create_channel_fingerprint_llm(
#                 video_df,
#                 channel_name=candidate['name'],
#                 model_type=MODEL_PROVIDER
#             )
            
#             if not cand_keywords:
#                 print(f"     ⚠️  Keyword generation failed")
#                 _update_status(seen_channels_data, candidate['id'], 'failed_fingerprint')
#                 continue
            
#             print(f"     Keywords: {cand_keywords[:3]}...")
            
#             # === METHOD 1: EMBEDDINGS ===
#             similarity_embeddings = calculate_embedding_similarity_hybrid(
#                 seed_keywords,
#                 cand_keywords
#             )
#             print(f"     📊 Embeddings: {similarity_embeddings:.3f}")
            
#             # === METHOD 2: LLM ===
#             similarity_llm = calculate_llm_similarity(
#                 seed_keywords,
#                 cand_keywords,
#                 TARGET_SEED_CHANNEL,
#                 candidate['name'],
#                 model_type=MODEL_PROVIDER
#             )
#             print(f"     🤖 LLM: {similarity_llm:.3f}")
            
#             # Build base result (common fields)
#             result_base = {
#                 "Seed_Channel_Name": TARGET_SEED_CHANNEL,
#                 "Seed_Channel_ID": seed_channel_id,
#                 "Seed_Keywords": ", ".join(seed_keywords),
#                 "Discovered_Channel_Name": candidate['name'],
#                 "Discovered_Channel_ID": candidate['id'],
#                 "Discovered_Channel_URL": candidate['url'],
#                 "Discovered_Subs": candidate['subscribers'],
#                 "Discovered_Keywords": ", ".join(cand_keywords),
#                 "Level": 1,
#                 "Timestamp": datetime.now().isoformat()
#             }
            
#             # Store EMBEDDINGS results
#             result_embeddings = {**result_base, "Similarity_Score": similarity_embeddings}
#             all_results_embeddings.append(result_embeddings)
            
#             if similarity_embeddings >= SIMILARITY_THRESHOLD_EMBEDDINGS:
#                 print(f"     ✅ HIGH (Embeddings >= {SIMILARITY_THRESHOLD_EMBEDDINGS})!")
#                 high_similarity_embeddings.append(result_embeddings)
            
#             # Store LLM results
#             result_llm = {**result_base, "Similarity_Score": similarity_llm}
#             all_results_llm.append(result_llm)
            
#             if similarity_llm >= SIMILARITY_THRESHOLD_LLM:
#                 print(f"     ✅ HIGH (LLM >= {SIMILARITY_THRESHOLD_LLM})!")
#                 high_similarity_llm.append(result_llm)
            
#             # Update seen channels with BOTH scores
#             _update_status_with_scores(
#                 seen_channels_data,
#                 candidate['id'],
#                 'scored',
#                 similarity_embeddings,
#                 similarity_llm
#             )
            
#             save_seen_channels(seen_channels_data, seen_channels_path)
#             time.sleep(DELAY_BETWEEN_CANDIDATES)
            
#         except Exception as e:
#             print(f"     ❌ Error: {str(e)[:100]}")
#             _update_status(seen_channels_data, candidate['id'], 'error_scoring')
    
#     # === SAVE RESULTS ===
#     print("\n" + "=" * 70)
#     print("SAVING RESULTS")
#     print("=" * 70)
    
#     # Save embeddings results
#     if all_results_embeddings:
#         df = pd.DataFrame(all_results_embeddings)
#         df.to_csv(all_results_embeddings_path, index=False, encoding='utf-8-sig')
#         print(f"✅ Embeddings (all): {len(df)} results → {all_results_embeddings_path.name}")
    
#     if high_similarity_embeddings:
#         df = pd.DataFrame(high_similarity_embeddings)
#         df.to_csv(high_similarity_embeddings_path, index=False, encoding='utf-8-sig')
#         print(f"✅ Embeddings (high): {len(df)} results → {high_similarity_embeddings_path.name}")
    
#     # Save LLM results
#     if all_results_llm:
#         df = pd.DataFrame(all_results_llm)
#         df.to_csv(all_results_llm_path, index=False, encoding='utf-8-sig')
#         print(f"✅ LLM (all): {len(df)} results → {all_results_llm_path.name}")
    
#     if high_similarity_llm:
#         df = pd.DataFrame(high_similarity_llm)
#         df.to_csv(high_similarity_llm_path, index=False, encoding='utf-8-sig')
#         print(f"✅ LLM (high): {len(df)} results → {high_similarity_llm_path.name}")
    
#     # === COMPARISON SUMMARY ===
#     elapsed = time.time() - start_time
    
#     print("\n" + "=" * 70)
#     print("A/B TEST COMPARISON")
#     print("=" * 70)
#     print(f"Seed processed: {TARGET_SEED_CHANNEL}")
#     print(f"Total candidates scored: {len(all_results_embeddings)}")
#     print()
#     print(f"📊 EMBEDDINGS (threshold >= {SIMILARITY_THRESHOLD_EMBEDDINGS}):")
#     print(f"   High similarity: {len(high_similarity_embeddings)} channels")
#     print()
#     print(f"🤖 LLM (threshold >= {SIMILARITY_THRESHOLD_LLM}):")
#     print(f"   High similarity: {len(high_similarity_llm)} channels")
#     print()
    
#     # Show difference
#     diff = len(high_similarity_embeddings) - len(high_similarity_llm)
#     if diff > 0:
#         print(f"📊 Embeddings found {diff} MORE high-similarity channels")
#     elif diff < 0:
#         print(f"🤖 LLM found {-diff} MORE high-similarity channels")
#     else:
#         print("⚖️  Both methods found same number of high-similarity channels")
    
#     print()
#     print(f"⏱️  Total runtime: {elapsed:.1f}s ({elapsed/len(all_results_embeddings):.1f}s per channel)")
#     print("=" * 70)

# if __name__ == "__main__":
#     main()




# WORKING -----------------------------------------------------


# import pandas as pd
# import json
# from pathlib import Path
# import time
# from datetime import datetime


# # --- Import Helper Functions ---
# from utils.youtube_utils import (
#     search_videos_multi_focused,
#     get_channel_metadata_batch,
#     fetch_recent_videos
# )
# from utils.fingerprint_llm_utils import (
#     create_channel_fingerprint_llm,
#     calculate_embedding_similarity_hybrid,
#     calculate_llm_similarity
# )


# # === CONFIGURATION ===
# base_dir = Path(__file__).resolve().parent


# # Input Files
# keywords_file_path = base_dir / "channel_keywords_simple_gemini_phrases_26.json"
# seed_video_data_path = base_dir / "sample_videos_26.csv"
# seen_channels_path = base_dir / "seen_channels.csv"


# # Output Files (6 files: 4 for individual + 2 for merged)
# all_results_embeddings_path = base_dir / "all_discovered_embeddings_28.csv"
# high_similarity_embeddings_path = base_dir / "high_similarity_embeddings_28.csv"
# all_results_llm_path = base_dir / "all_discovered_llm_28.csv"
# high_similarity_llm_path = base_dir / "high_similarity_llm_28.csv"

# # NEW: Merged output files
# merged_with_duplicates_path = base_dir / "FINAL_SEED_SUBSCRIBE_MERGED_DUPLICATES_28.csv"
# merged_final_path = base_dir / "FINAL_SEED_SUBSCRIBE_MERGED_28.csv"


# # Settings
# MODEL_PROVIDER = "gpt"  # "gemini" or "openai"

# # Multi-seed configuration
# SEED_CHANNELS = [
#     "Lenny's Podcast",
#     "The Diary of a CEO",
#     "Colin and Samir"
# ]


# # Column order (Manager's request)
# FINAL_COLUMN_ORDER = [
#     'Seed_Channel_Name',
#     'Seed_Channel_ID',
#     'Discovered_Channel_Name',
#     'Discovered_Channel_ID',
#     'Discovered_Channel_URL',
#     'Discovered_Subs',
#     'Level',
#     'Timestamp',
#     'Similarity_Score',
#     'Seed_Keywords',
#     'Discovered_Keywords'
# ]


# # Thresholds
# SIMILARITY_THRESHOLD_EMBEDDINGS = 0.75
# SIMILARITY_THRESHOLD_LLM = 0.7
# MIN_SUBSCRIBERS = 10000
# MIN_VIDEOS = 10
# VIDEOS_PER_CANDIDATE = 10


# # Rate Limiting
# DELAY_BETWEEN_CANDIDATES = 3
# DELAY_BETWEEN_SEEDS = 10


# # === HELPER FUNCTIONS ===


# def save_seen_channels(seen_data_list, file_path):
#     """Saves seen channels to CSV with deduplication"""
#     if not seen_data_list:
#         print("  (No seen channel data to save)")
#         return
    
#     try:
#         df = pd.DataFrame(seen_data_list)
#         df = df.sort_values(by="Date_Added").drop_duplicates(
#             subset=['Channel_ID'], 
#             keep='last'
#         )
#         df.to_csv(file_path, index=False, encoding='utf-8-sig')
#         print(f"  💾 Saved {len(df)} seen channels to {file_path.name}")
#     except Exception as e:
#         print(f"  ❌ Error saving {file_path.name}: {e}")


# def load_seen_channels(file_path):
#     """Loads seen channels from CSV"""
#     seen_data = []
#     seen_ids = set()
    
#     if file_path.exists():
#         try:
#             df = pd.read_csv(file_path)
#             required_cols = ['Channel_ID', 'Channel_Name', 'Date_Added', 'Processing_Status']
            
#             if all(col in df.columns for col in required_cols):
#                 seen_data = df.to_dict('records')
#                 seen_ids = set(df['Channel_ID'].astype(str).tolist())
#                 print(f"  📂 Loaded {len(seen_ids)} previously seen channels")
#             else:
#                 print("  ⚠️  Seen channels CSV has wrong columns. Starting fresh.")
#         except Exception as e:
#             print(f"  ⚠️  Could not read seen channels: {e}")
#     else:
#         print("  📂 No existing seen channels file. Starting fresh.")
    
#     return seen_data, seen_ids


# def _update_status(seen_data, channel_id, status):
#     """Update processing status for a channel"""
#     for entry in seen_data:
#         if entry['Channel_ID'] == channel_id:
#             entry['Processing_Status'] = status
#             break


# def _update_status_with_scores(seen_data, channel_id, status, score_emb, score_llm):
#     """Update status with BOTH scores"""
#     for entry in seen_data:
#         if entry['Channel_ID'] == channel_id:
#             entry['Processing_Status'] = status
#             entry['Embedding_Score'] = score_emb
#             entry['LLM_Score'] = score_llm
#             break


# def merge_and_deduplicate(embedding_results, llm_results):
#     """
#     Merge embedding and LLM results, remove duplicates
#     Strategy: Keep entry with higher similarity score
#     Returns: (df_with_duplicates, df_deduplicated)
#     """
    
#     # Combine both results
#     all_results = embedding_results + llm_results
    
#     if not all_results:
#         empty_df = pd.DataFrame(columns=FINAL_COLUMN_ORDER)
#         return empty_df, empty_df
    
#     # Create DataFrame with all results (including duplicates)
#     df_with_duplicates = pd.DataFrame(all_results)
    
#     print(f"\n📊 Merging Results:")
#     print(f"   Embedding results: {len(embedding_results)}")
#     print(f"   LLM results: {len(llm_results)}")
#     print(f"   Combined (with duplicates): {len(df_with_duplicates)}")
    
#     # Deduplicate based on Seed_Channel_ID + Discovered_Channel_ID
#     # Keep the one with highest similarity score
#     df_deduped = df_with_duplicates.sort_values('Similarity_Score', ascending=False)\
#                                    .drop_duplicates(
#                                        subset=['Seed_Channel_ID', 'Discovered_Channel_ID'],
#                                        keep='first'
#                                    )
    
#     duplicates_removed = len(df_with_duplicates) - len(df_deduped)
#     print(f"   After deduplication: {len(df_deduped)}")
#     print(f"   Duplicates removed: {duplicates_removed}")
    
#     # Sort by Seed_Channel_Name and Similarity_Score
#     df_deduped = df_deduped.sort_values(
#         ['Seed_Channel_Name', 'Similarity_Score'],
#         ascending=[True, False]
#     )
    
#     # Reorder columns for both DataFrames
#     df_with_duplicates = df_with_duplicates[FINAL_COLUMN_ORDER]
#     df_deduped = df_deduped[FINAL_COLUMN_ORDER]
    
#     return df_with_duplicates, df_deduped


# def process_seed_channel(seed_channel, seed_channel_id, seed_keywords, 
#                           seen_channels_data, seen_ids):
#     """
#     Process a single seed channel and return results for both methods
#     Returns: (embedding_results, llm_results)
#     """
    
#     print("\n" + "=" * 70)
#     print(f"PROCESSING SEED: {seed_channel}")
#     print("=" * 70)
    
#     current_time = datetime.now().isoformat()
    
#     # Initialize results for this seed
#     embedding_results = []
#     llm_results = []
    
#     # --- STEP 1: Multi-Focused Search ---
#     print(f"\n🔍 STEP 1: Searching YouTube...")
    
#     try:
#         candidate_ids = search_videos_multi_focused(
#             seed_keywords, 
#             max_results_per_search=20,
#             max_keywords=10
#         )
#     except Exception as e:
#         print(f"❌ Search failed: {e}")
#         return embedding_results, llm_results
    
#     candidate_ids = candidate_ids - seen_ids - {seed_channel_id}
    
#     if not candidate_ids:
#         print("  ⚠️  No new candidates found after filtering")
#         return embedding_results, llm_results
    
#     print(f"  ✅ {len(candidate_ids)} new candidates to evaluate")
    
#     # --- STEP 2: Get Metadata ---
#     print(f"\n📊 STEP 2: Fetching channel metadata...")
    
#     metadata = get_channel_metadata_batch(list(candidate_ids))
    
#     for meta in metadata:
#         if meta['id'] not in seen_ids:
#             seen_channels_data.append({
#                 "Channel_ID": meta['id'],
#                 "Channel_Name": meta['name'],
#                 "Date_Added": current_time,
#                 "Processing_Status": "discovered",
#                 "Embedding_Score": None,
#                 "LLM_Score": None
#             })
#             seen_ids.add(meta['id'])
    
#     save_seen_channels(seen_channels_data, seen_channels_path)
    
#     # --- STEP 3: Pre-Filter ---
#     print(f"\n🔍 STEP 3: Pre-filtering by subscribers and videos...")
    
#     qualified = []
    
#     for meta in metadata:
#         if meta['subscribers'] == -1:
#             print(f"  - Filtering {meta['name']} (subscribers hidden)")
#             _update_status(seen_channels_data, meta['id'], 'filtered_hidden_subs')
#             continue
        
#         if meta['subscribers'] < MIN_SUBSCRIBERS:
#             print(f"  - Filtering {meta['name']} (subs: {meta['subscribers']:,})")
#             _update_status(seen_channels_data, meta['id'], 'filtered_subs')
#             continue
        
#         if meta['video_count'] < MIN_VIDEOS:
#             print(f"  - Filtering {meta['name']} (videos: {meta['video_count']})")
#             _update_status(seen_channels_data, meta['id'], 'filtered_videos')
#             continue
        
#         qualified.append(meta)
    
#     save_seen_channels(seen_channels_data, seen_channels_path)
    
#     if not qualified:
#         print("  ⚠️  No candidates passed pre-filtering")
#         return embedding_results, llm_results
    
#     # Sort by subscribers (but score ALL of them!)
#     qualified.sort(key=lambda x: x['subscribers'], reverse=True)
#     to_score = qualified
    
#     print(f"  ✅ {len(qualified)} qualified, scoring ALL {len(to_score)} candidates")
    
#     # --- STEP 4: Score Candidates (BOTH methods) ---
#     print(f"\n🎯 STEP 4: Scoring candidates with BOTH methods...")
#     print(f"   (This will take ~{len(to_score) * 10} seconds)\n")
    
#     for i, candidate in enumerate(to_score, 1):
#         print(f"  [{i}/{len(to_score)}] {candidate['name']}")
#         print(f"     Subs: {candidate['subscribers']:,}")
        
#         try:
#             # Fetch videos
#             print(f"     Fetching {VIDEOS_PER_CANDIDATE} videos...")
#             videos = fetch_recent_videos(
#                 candidate['id'],
#                 max_results=VIDEOS_PER_CANDIDATE,
#                 filter_shorts=False
#             )
            
#             if len(videos) < 3:
#                 print(f"     ⚠️  Only {len(videos)} videos found, skipping")
#                 _update_status(seen_channels_data, candidate['id'], 'skipped_few_videos')
#                 continue
            
#             # Generate fingerprint
#             print(f"     Generating keywords with {MODEL_PROVIDER.upper()}...")
#             video_df = pd.DataFrame(videos)
            
#             cand_keywords = create_channel_fingerprint_llm(
#                 video_df,
#                 channel_name=candidate['name'],
#                 model_type=MODEL_PROVIDER
#             )
            
#             if not cand_keywords:
#                 print(f"     ⚠️  Keyword generation failed")
#                 _update_status(seen_channels_data, candidate['id'], 'failed_fingerprint')
#                 continue
            
#             print(f"     Keywords: {cand_keywords[:3]}...")
            
#             # === METHOD 1: EMBEDDINGS ===
#             similarity_embeddings = calculate_embedding_similarity_hybrid(
#                 seed_keywords,
#                 cand_keywords
#             )
#             print(f"     📊 Embeddings: {similarity_embeddings:.3f}")
            
#             # === METHOD 2: LLM ===
#             similarity_llm = calculate_llm_similarity(
#                 seed_keywords,
#                 cand_keywords,
#                 seed_channel,
#                 candidate['name'],
#                 model_type=MODEL_PROVIDER
#             )
#             print(f"     🤖 LLM: {similarity_llm:.3f}")
            
#             # Build base result (common fields)
#             result_base = {
#                 "Seed_Channel_Name": seed_channel,
#                 "Seed_Channel_ID": seed_channel_id,
#                 "Discovered_Channel_Name": candidate['name'],
#                 "Discovered_Channel_ID": candidate['id'],
#                 "Discovered_Channel_URL": candidate['url'],
#                 "Discovered_Subs": candidate['subscribers'],
#                 "Discovered_Keywords": ", ".join(cand_keywords),
#                 "Level": 1,
#                 "Timestamp": datetime.now().isoformat(),
#                 "Seed_Keywords": ", ".join(seed_keywords)
#             }
            
#             # Store EMBEDDINGS results (only high similarity)
#             if similarity_embeddings >= SIMILARITY_THRESHOLD_EMBEDDINGS:
#                 result_embeddings = {**result_base, "Similarity_Score": similarity_embeddings}
#                 embedding_results.append(result_embeddings)
#                 print(f"     ✅ HIGH (Embeddings >= {SIMILARITY_THRESHOLD_EMBEDDINGS})!")
            
#             # Store LLM results (only high similarity)
#             if similarity_llm >= SIMILARITY_THRESHOLD_LLM:
#                 result_llm = {**result_base, "Similarity_Score": similarity_llm}
#                 llm_results.append(result_llm)
#                 print(f"     ✅ HIGH (LLM >= {SIMILARITY_THRESHOLD_LLM})!")
            
#             # Update seen channels with BOTH scores
#             _update_status_with_scores(
#                 seen_channels_data,
#                 candidate['id'],
#                 'scored',
#                 similarity_embeddings,
#                 similarity_llm
#             )
            
#             save_seen_channels(seen_channels_data, seen_channels_path)
#             time.sleep(DELAY_BETWEEN_CANDIDATES)
            
#         except Exception as e:
#             print(f"     ❌ Error: {str(e)[:100]}")
#             _update_status(seen_channels_data, candidate['id'], 'error_scoring')
    
#     return embedding_results, llm_results


# # === MAIN FUNCTION ===


# def main():
#     start_time = time.time()
#     print("=" * 70)
#     print(f"MULTI-SEED CHANNEL DISCOVERY WITH A/B TESTING")
#     print(f"Seeds: {', '.join(SEED_CHANNELS)}")
#     print(f"LLM Provider: {MODEL_PROVIDER.upper()}")
#     print("=" * 70)
#     print(f"📊 Embedding Threshold: {SIMILARITY_THRESHOLD_EMBEDDINGS}")
#     print(f"🤖 LLM Threshold: {SIMILARITY_THRESHOLD_LLM}")
    
#     # --- 1. Load Keywords ---
#     print(f"\n📖 Loading keywords from {keywords_file_path.name}...")
    
#     try:
#         with open(keywords_file_path, 'r', encoding='utf-8') as f:
#             all_keywords = json.load(f)
#         print(f"  ✅ Loaded keywords for {len(all_keywords)} channels")
#     except Exception as e:
#         print(f"❌ ERROR loading keywords: {e}")
#         return
    
#     # --- 2. Load Seed Channel ID Mapping ---
#     print(f"\n🗺️  Loading channel IDs from {seed_video_data_path.name}...")
    
#     try:
#         df_videos = pd.read_csv(seed_video_data_path)
        
#         if 'Channel_Name' not in df_videos.columns or 'Channel_ID' not in df_videos.columns:
#             print("❌ ERROR: CSV must have 'Channel_Name' and 'Channel_ID' columns")
#             return
        
#         seed_id_map = df_videos.drop_duplicates(subset=['Channel_Name'])[
#             ['Channel_Name', 'Channel_ID']
#         ].set_index('Channel_Name')['Channel_ID'].to_dict()
        
#         print(f"  ✅ Loaded {len(seed_id_map)} channel mappings")
        
#     except Exception as e:
#         print(f"❌ ERROR loading channel IDs: {e}")
#         return
    
#     # --- 3. Load Seen Channels ---
#     print(f"\n📂 Loading seen channels from {seen_channels_path.name}...")
    
#     seen_channels_data, seen_ids = load_seen_channels(seen_channels_path)
    
#     current_time = datetime.now().isoformat()
    
#     # Add all seed channels to seen list
#     for seed_channel in SEED_CHANNELS:
#         seed_channel_id = seed_id_map.get(seed_channel)
#         if seed_channel_id and seed_channel_id not in seen_ids:
#             seen_channels_data.append({
#                 "Channel_ID": seed_channel_id,
#                 "Channel_Name": seed_channel,
#                 "Date_Added": current_time,
#                 "Processing_Status": "seed",
#                 "Embedding_Score": None,
#                 "LLM_Score": None
#             })
#             seen_ids.add(seed_channel_id)
#             print(f"  ✅ Added seed '{seed_channel}' to seen list")
    
#     save_seen_channels(seen_channels_data, seen_channels_path)
    
#     # --- 4. Initialize Aggregated Results ---
#     all_embedding_results = []
#     all_llm_results = []
    
#     # === PROCESS EACH SEED CHANNEL ===
#     for seed_idx, seed_channel in enumerate(SEED_CHANNELS, 1):
#         print(f"\n{'='*70}")
#         print(f"SEED {seed_idx}/{len(SEED_CHANNELS)}: {seed_channel}")
#         print(f"{'='*70}")
        
#         # Get seed info
#         if seed_channel not in all_keywords:
#             print(f"⚠️  WARNING: No keywords found for '{seed_channel}', skipping")
#             continue
        
#         seed_channel_id = seed_id_map.get(seed_channel)
#         if not seed_channel_id:
#             print(f"⚠️  WARNING: No Channel ID found for '{seed_channel}', skipping")
#             continue
        
#         seed_keywords = all_keywords[seed_channel]
#         print(f"Keywords: {seed_keywords[:5]}...")
        
#         # Process this seed
#         embedding_results, llm_results = process_seed_channel(
#             seed_channel,
#             seed_channel_id,
#             seed_keywords,
#             seen_channels_data,
#             seen_ids
#         )
        
#         # Aggregate results
#         all_embedding_results.extend(embedding_results)
#         all_llm_results.extend(llm_results)
        
#         print(f"\n✅ Seed '{seed_channel}' complete:")
#         print(f"   Embeddings: {len(embedding_results)} high-similarity channels")
#         print(f"   LLM: {len(llm_results)} high-similarity channels")
        
#         # Delay between seeds
#         if seed_idx < len(SEED_CHANNELS):
#             print(f"\n⏸️  Waiting {DELAY_BETWEEN_SEEDS}s before next seed...")
#             time.sleep(DELAY_BETWEEN_SEEDS)
    
#     # === SAVE INDIVIDUAL RESULTS ===
#     print("\n" + "=" * 70)
#     print("SAVING INDIVIDUAL RESULTS")
#     print("=" * 70)
    
#     if all_embedding_results:
#         df = pd.DataFrame(all_embedding_results)
#         df.to_csv(high_similarity_embeddings_path, index=False, encoding='utf-8-sig')
#         print(f"✅ Embeddings (high): {len(df)} results → {high_similarity_embeddings_path.name}")
    
#     if all_llm_results:
#         df = pd.DataFrame(all_llm_results)
#         df.to_csv(high_similarity_llm_path, index=False, encoding='utf-8-sig')
#         print(f"✅ LLM (high): {len(df)} results → {high_similarity_llm_path.name}")
    
#     # === MERGE AND SAVE ===
#     print("\n" + "=" * 70)
#     print("MERGING RESULTS")
#     print("=" * 70)
    
#     df_with_duplicates, df_final = merge_and_deduplicate(
#         all_embedding_results,
#         all_llm_results
#     )
    
#     # Save with duplicates
#     if not df_with_duplicates.empty:
#         df_with_duplicates.to_csv(merged_with_duplicates_path, index=False, encoding='utf-8-sig')
#         print(f"✅ Saved (with duplicates): {len(df_with_duplicates)} → {merged_with_duplicates_path.name}")
    
#     # Save deduplicated final
#     if not df_final.empty:
#         df_final.to_csv(merged_final_path, index=False, encoding='utf-8-sig')
#         print(f"✅ Saved (deduplicated): {len(df_final)} → {merged_final_path.name}")
    
#     # === FINAL SUMMARY ===
#     elapsed = time.time() - start_time
    
#     print("\n" + "=" * 70)
#     print("FINAL SUMMARY")
#     print("=" * 70)
#     print(f"Seeds processed: {len(SEED_CHANNELS)}")
#     print(f"Total embedding high-similarity: {len(all_embedding_results)}")
#     print(f"Total LLM high-similarity: {len(all_llm_results)}")
#     print(f"Combined (with duplicates): {len(df_with_duplicates)}")
#     print(f"Final unique channels: {len(df_final)}")
#     print()
    
#     if not df_final.empty:
#         print("Breakdown by seed:")
#         print(df_final.groupby('Seed_Channel_Name').size())
    
#     print()
#     print(f"⏱️  Total runtime: {elapsed/60:.1f} minutes")
#     print("=" * 70)
#     print("\n✅ ALL PROCESSING COMPLETE!")


# if __name__ == "__main__":
#     main()

# working-----------------


import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime


# --- Import Helper Functions ---
from utils.youtube_utils import (
    search_videos_multi_focused,
    get_channel_metadata_batch,
    fetch_recent_videos
)
# --- MODIFIED: Import all required LLM functions ---
from utils.fingerprint_llm_utils import (
    create_channel_fingerprint_llm,
    extract_niche_llm,  # <-- ADDED
    calculate_embedding_similarity_hybrid,
    calculate_llm_similarity
)


# === CONFIGURATION ===
base_dir = Path(__file__).resolve().parent


# --- MODIFIED: Point to your new JSON file with niches ---
keywords_file_path = base_dir / "channel_fingerprints_gpt_phrases_26.json"
seed_video_data_path = base_dir / "sample_videos_26.csv"
seen_channels_path = base_dir / "seen_channels.csv"


# Output Files (6 files: 4 for individual + 2 for merged)
all_results_embeddings_path = base_dir / "all_discovered_embeddings_29.csv"
high_similarity_embeddings_path = base_dir / "high_similarity_embeddings_29.csv"
all_results_llm_path = base_dir / "all_discovered_llm_29.csv"
high_similarity_llm_path = base_dir / "high_similarity_llm_29.csv"

# NEW: Merged output files
merged_with_duplicates_path = base_dir / "FINAL_SEED_SUBSCRIBE_MERGED_DUPLICATES_29.csv"
merged_final_path = base_dir / "FINAL_SEED_SUBSCRIBE_MERGED_29.csv"


# Settings
MODEL_PROVIDER = "gpt"  # "gemini" or "gpt"

# Multi-seed configuration
SEED_CHANNELS = [
    "Ali Abdaal",
    "Vox",
    "Johnny Harris",
    "Moon"
]


# Column order (Manager's request)
FINAL_COLUMN_ORDER = [
    'Seed_Channel_Name',
    'Seed_Channel_ID',
    'Seed_Niche',              # ← ADD THIS
    'Discovered_Channel_Name',
    'Discovered_Channel_ID',
    'Discovered_Channel_URL',
    'Discovered_Subs',
    'Discovered_Niche',        # ← ADD THIS
    'Level',
    'Timestamp',
    'Similarity_Score',
    'Seed_Keywords',
    'Discovered_Keywords'
]



# Thresholds
SIMILARITY_THRESHOLD_EMBEDDINGS = 0.75
SIMILARITY_THRESHOLD_LLM = 0.7
MIN_SUBSCRIBERS = 10000
MIN_VIDEOS = 10
VIDEOS_PER_CANDIDATE = 10


# Rate Limiting
DELAY_BETWEEN_CANDIDATES = 3
DELAY_BETWEEN_SEEDS = 10


# === HELPER FUNCTIONS ===


def save_seen_channels(seen_data_list, file_path):
    """Saves seen channels to CSV with deduplication"""
    if not seen_data_list:
        print("  (No seen channel data to save)")
        return
    
    try:
        # --- MODIFIED: Ensure all new columns are present ---
        df = pd.DataFrame(seen_data_list)
        
        # Define all possible columns to avoid errors if one is missing
        all_cols = ['Channel_ID', 'Channel_Name', 'Date_Added', 'Processing_Status', 'Embedding_Score', 'LLM_Score', 'niche']
        
        # Add any missing columns with None
        for col in all_cols:
            if col not in df.columns:
                df[col] = None
        
        # Reorder for consistency
        df = df[all_cols]

        df = df.sort_values(by="Date_Added").drop_duplicates(
            subset=['Channel_ID'], 
            keep='last'
        )
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"  💾 Saved {len(df)} seen channels to {file_path.name}")
    except Exception as e:
        print(f"  ❌ Error saving {file_path.name}: {e}")


def load_seen_channels(file_path):
    """Loads seen channels from CSV"""
    seen_data = []
    seen_ids = set()
    
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            # --- MODIFIED: Only require the base columns ---
            required_cols = ['Channel_ID', 'Channel_Name', 'Date_Added', 'Processing_Status']
            
            if all(col in df.columns for col in required_cols):
                # FillNA for potentially missing new columns
                if 'niche' not in df.columns:
                    df['niche'] = None
                df['niche'] = df['niche'].fillna("N/A")
                
                seen_data = df.to_dict('records')
                seen_ids = set(df['Channel_ID'].astype(str).tolist())
                print(f"  📂 Loaded {len(seen_ids)} previously seen channels")
            else:
                print("  ⚠️  Seen channels CSV has wrong columns. Starting fresh.")
        except Exception as e:
            print(f"  ⚠️  Could not read seen channels: {e}")
    else:
        print("  📂 No existing seen channels file. Starting fresh.")
    
    return seen_data, seen_ids


def _update_status(seen_data, channel_id, status):
    """Update processing status for a channel"""
    for entry in seen_data:
        if entry['Channel_ID'] == channel_id:
            entry['Processing_Status'] = status
            break


# --- MODIFIED: Add niche to the update function ---
def _update_status_with_scores(seen_data, channel_id, status, score_emb, score_llm, niche):
    """Update status with BOTH scores AND niche"""
    for entry in seen_data:
        if entry['Channel_ID'] == channel_id:
            entry['Processing_Status'] = status
            entry['Embedding_Score'] = score_emb
            entry['LLM_Score'] = score_llm
            entry['niche'] = niche  # <-- ADDED
            break


def merge_and_deduplicate(embedding_results, llm_results):
    """
    Merge embedding and LLM results, remove duplicates
    Strategy: Keep entry with higher similarity score
    Returns: (df_with_duplicates, df_deduplicated)
    """
    
    # Combine both results
    all_results = embedding_results + llm_results
    
    if not all_results:
        empty_df = pd.DataFrame(columns=FINAL_COLUMN_ORDER)
        return empty_df, empty_df
    
    # Create DataFrame with all results (including duplicates)
    df_with_duplicates = pd.DataFrame(all_results)
    
    print(f"\n📊 Merging Results:")
    print(f"   Embedding results: {len(embedding_results)}")
    print(f"   LLM results: {len(llm_results)}")
    print(f"   Combined (with duplicates): {len(df_with_duplicates)}")
    
    # Deduplicate based on Seed_Channel_ID + Discovered_Channel_ID
    # Keep the one with highest similarity score
    df_deduped = df_with_duplicates.sort_values('Similarity_Score', ascending=False)\
                                   .drop_duplicates(
                                       subset=['Seed_Channel_ID', 'Discovered_Channel_ID'],
                                       keep='first'
                                   )
    
    duplicates_removed = len(df_with_duplicates) - len(df_deduped)
    print(f"   After deduplication: {len(df_deduped)}")
    print(f"   Duplicates removed: {duplicates_removed}")
    
    # Sort by Seed_Channel_Name and Similarity_Score
    df_deduped = df_deduped.sort_values(
        ['Seed_Channel_Name', 'Similarity_Score'],
        ascending=[True, False]
    )
    
    # Reorder columns for both DataFrames
    df_with_duplicates = df_with_duplicates[FINAL_COLUMN_ORDER]
    df_deduped = df_deduped[FINAL_COLUMN_ORDER]
    
    return df_with_duplicates, df_deduped


# --- MODIFIED: Function signature now accepts seed_niche ---
def process_seed_channel(seed_channel, seed_channel_id, seed_keywords, 
                          seed_niche, # <-- ADDED
                          seen_channels_data, seen_ids):
    """
    Process a single seed channel and return results for both methods
    Returns: (embedding_results, llm_results)
    """
    
    print("\n" + "=" * 70)
    print(f"PROCESSING SEED: {seed_channel} (Niche: {seed_niche})")
    print("=" * 70)
    
    current_time = datetime.now().isoformat()
    
    # Initialize results for this seed
    embedding_results = []
    llm_results = []
    
    # --- STEP 1: Multi-Focused Search ---
    print(f"\n🔍 STEP 1: Searching YouTube...")
    
    try:
        candidate_ids = search_videos_multi_focused(
            seed_keywords, 
            max_results_per_search=20,
            max_keywords=10
        )
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return embedding_results, llm_results
    
    candidate_ids = candidate_ids - seen_ids - {seed_channel_id}
    
    if not candidate_ids:
        print("  ⚠️  No new candidates found after filtering")
        return embedding_results, llm_results
    
    print(f"  ✅ {len(candidate_ids)} new candidates to evaluate")
    
    # --- STEP 2: Get Metadata ---
    print(f"\n📊 STEP 2: Fetching channel metadata...")
    
    # --- MODIFIED: get_channel_metadata_batch now returns 'description' ---
    # We assume your youtube_utils is updated as per our last chat
    metadata = get_channel_metadata_batch(list(candidate_ids))
    
    for meta in metadata:
        if meta['id'] not in seen_ids:
            seen_channels_data.append({
                "Channel_ID": meta['id'],
                "Channel_Name": meta['name'],
                "Date_Added": current_time,
                "Processing_Status": "discovered",
                "Embedding_Score": None,
                "LLM_Score": None,
                "niche": None # <-- ADDED (will be filled in Step 4)
            })
            seen_ids.add(meta['id'])
    
    save_seen_channels(seen_channels_data, seen_channels_path)
    
    # --- STEP 3: Pre-Filter ---
    print(f"\n🔍 STEP 3: Pre-filtering by subscribers and videos...")
    
    qualified = []
    
    for meta in metadata:
        if meta['subscribers'] == -1:
            print(f"  - Filtering {meta['name']} (subscribers hidden)")
            _update_status(seen_channels_data, meta['id'], 'filtered_hidden_subs')
            continue
        
        if meta['subscribers'] < MIN_SUBSCRIBERS:
            print(f"  - Filtering {meta['name']} (subs: {meta['subscribers']:,})")
            _update_status(seen_channels_data, meta['id'], 'filtered_subs')
            continue
        
        if meta['video_count'] < MIN_VIDEOS:
            print(f"  - Filtering {meta['name']} (videos: {meta['video_count']})")
            _update_status(seen_channels_data, meta['id'], 'filtered_videos')
            continue
        
        qualified.append(meta)
    
    save_seen_channels(seen_channels_data, seen_channels_path)
    
    if not qualified:
        print("  ⚠️  No candidates passed pre-filtering")
        return embedding_results, llm_results
    
    # Sort by subscribers (but score ALL of them!)
    qualified.sort(key=lambda x: x['subscribers'], reverse=True)
    to_score = qualified
    
    print(f"  ✅ {len(qualified)} qualified, scoring ALL {len(to_score)} candidates")
    
    # --- STEP 4: Score Candidates (BOTH methods) ---
    print(f"\n🎯 STEP 4: Scoring candidates with BOTH methods...")
    print(f"   (This will take ~{len(to_score) * 10} seconds)\n")
    
    for i, candidate in enumerate(to_score, 1):
        print(f"  [{i}/{len(to_score)}] {candidate['name']}")
        print(f"     Subs: {candidate['subscribers']:,}")
        
        try:
            # --- MODIFIED: Fetch videos AND description ---
            # We assume your youtube_utils is updated to return (videos, description)
            print(f"     Fetching {VIDEOS_PER_CANDIDATE} videos...")
            videos, cand_desc = fetch_recent_videos(
                candidate['id'],
                max_results=VIDEOS_PER_CANDIDATE,
                filter_shorts=False
            )
            
            if len(videos) < 3:
                print(f"     ⚠️  Only {len(videos)} videos found, skipping")
                _update_status(seen_channels_data, candidate['id'], 'skipped_few_videos')
                continue
            
            video_df = pd.DataFrame(videos)
            video_titles_list = video_df['title'].tolist() # <-- ADDED
            
            # --- ADDED: Step 4a: Get Candidate Niche ---
            print(f"     Extracting niche with {MODEL_PROVIDER.upper()}...")
            cand_niche = extract_niche_llm(
                channel_description=cand_desc,
                video_titles=video_titles_list,
                channel_name=candidate['name'],
                model_type=MODEL_PROVIDER
            )

            # --- MODIFIED: Step 4b: Get Candidate Keywords ---
            print(f"     Generating keywords (Niche: {cand_niche})...")
            cand_keywords = create_channel_fingerprint_llm(
                video_df,
                channel_name=candidate['name'],
                niche=cand_niche, # <-- ADDED
                model_type=MODEL_PROVIDER
            )
            
            if not cand_keywords:
                print(f"     ⚠️  Keyword generation failed")
                _update_status(seen_channels_data, candidate['id'], 'failed_fingerprint')
                continue
            
            print(f"     Keywords: {cand_keywords[:3]}...")
            
            # === METHOD 1: EMBEDDINGS ===
            similarity_embeddings = calculate_embedding_similarity_hybrid(
                seed_keywords,
                cand_keywords
            )
            print(f"     📊 Embeddings: {similarity_embeddings:.3f}")
            
            # === METHOD 2: LLM ===
            # --- MODIFIED: Pass niches to LLM scorer ---
            similarity_llm = calculate_llm_similarity(
                seed_keywords,
                cand_keywords,
                seed_channel,
                candidate['name'],
                seed_niche=seed_niche, # <-- ADDED
                candidate_niche=cand_niche, # <-- ADDED
                model_type=MODEL_PROVIDER
            )
            print(f"     🤖 LLM: {similarity_llm:.3f}")
            
            # Build base result (common fields)
            # result_base = {
            #     "Seed_Channel_Name": seed_channel,
            #     "Seed_Channel_ID": seed_channel_id,
            #     "Discovered_Channel_Name": candidate['name'],
            #     "Discovered_Channel_ID": candidate['id'],
            #     "Discovered_Channel_URL": candidate['url'],
            #     "Discovered_Subs": candidate['subscribers'],
            #     "Discovered_Keywords": ", ".join(cand_keywords),
            #     "Level": 1,
            #     "Timestamp": datetime.now().isoformat(),
            #     "Seed_Keywords": ", ".join(seed_keywords)
            # }
            # Build base result (common fields)
            result_base = {
                "Seed_Channel_Name": seed_channel,
                "Seed_Channel_ID": seed_channel_id,
                "Seed_Niche": seed_niche,              # ← ADD THIS
                "Discovered_Channel_Name": candidate['name'],
                "Discovered_Channel_ID": candidate['id'],
                "Discovered_Channel_URL": candidate['url'],
                "Discovered_Subs": candidate['subscribers'],
                "Discovered_Niche": cand_niche,        # ← ADD THIS
                "Discovered_Keywords": ", ".join(cand_keywords),
                "Level": 1,
                "Timestamp": datetime.now().isoformat(),
                "Seed_Keywords": ", ".join(seed_keywords)
            }

            
            # Store EMBEDDINGS results (only high similarity)
            if similarity_embeddings >= SIMILARITY_THRESHOLD_EMBEDDINGS:
                result_embeddings = {**result_base, "Similarity_Score": similarity_embeddings}
                embedding_results.append(result_embeddings)
                print(f"     ✅ HIGH (Embeddings >= {SIMILARITY_THRESHOLD_EMBEDDINGS})!")
            
            # Store LLM results (only high similarity)
            if similarity_llm >= SIMILARITY_THRESHOLD_LLM:
                result_llm = {**result_base, "Similarity_Score": similarity_llm}
                llm_results.append(result_llm)
                print(f"     ✅ HIGH (LLM >= {SIMILARITY_THRESHOLD_LLM})!")
            
            # --- MODIFIED: Update seen channels with niche ---
            _update_status_with_scores(
                seen_channels_data,
                candidate['id'],
                'scored',
                similarity_embeddings,
                similarity_llm,
                cand_niche # <-- ADDED
            )
            
            save_seen_channels(seen_channels_data, seen_channels_path)
            time.sleep(DELAY_BETWEEN_CANDIDATES)
            
        except Exception as e:
            print(f"     ❌ Error: {str(e)[:100]}")
            _update_status(seen_channels_data, candidate['id'], 'error_scoring')
    
    return embedding_results, llm_results


# === MAIN FUNCTION ===


def main():
    start_time = time.time()
    print("=" * 70)
    print(f"MULTI-SEED CHANNEL DISCOVERY (NICHE-AWARE)") # <-- MODIFIED
    print(f"Seeds: {', '.join(SEED_CHANNELS)}")
    print(f"LLM Provider: {MODEL_PROVIDER.upper()}")
    print("=" * 70)
    print(f"📊 Embedding Threshold: {SIMILARITY_THRESHOLD_EMBEDDINGS}")
    print(f"🤖 LLM Threshold: {SIMILARITY_THRESHOLD_LLM}")
    
    # --- 1. Load Keywords AND Niches ---
    print(f"\n📖 Loading keywords & niches from {keywords_file_path.name}...")
    
    # --- MODIFIED: Create maps for both keywords and niches ---
    seed_keywords_map = {}
    seed_niche_map = {}
    
    try:
        with open(keywords_file_path, 'r', encoding='utf-8') as f:
            # Parse the new JSON structure
            data = json.load(f)
            for channel_name, details in data.get("channels", {}).items():
                if details.get("keywords"):
                    seed_keywords_map[channel_name] = details["keywords"]
                if details.get("niche"):
                    seed_niche_map[channel_name] = details["niche"]
                    
        print(f"  ✅ Loaded keywords for {len(seed_keywords_map)} channels")
        print(f"  ✅ Loaded niches for {len(seed_niche_map)} channels")
    except Exception as e:
        print(f"❌ ERROR loading keywords/niches: {e}")
        return
    
    # --- 2. Load Seed Channel ID Mapping ---
    print(f"\n🗺️  Loading channel IDs from {seed_video_data_path.name}...")
    
    try:
        df_videos = pd.read_csv(seed_video_data_path)
        
        if 'Channel_Name' not in df_videos.columns or 'Channel_ID' not in df_videos.columns:
            print("❌ ERROR: CSV must have 'Channel_Name' and 'Channel_ID' columns")
            return
        
        seed_id_map = df_videos.drop_duplicates(subset=['Channel_Name'])[
            ['Channel_Name', 'Channel_ID']
        ].set_index('Channel_Name')['Channel_ID'].to_dict()
        
        print(f"  ✅ Loaded {len(seed_id_map)} channel mappings")
        
    except Exception as e:
        print(f"❌ ERROR loading channel IDs: {e}")
        return
    
    # --- 3. Load Seen Channels ---
    print(f"\n📂 Loading seen channels from {seen_channels_path.name}...")
    
    seen_channels_data, seen_ids = load_seen_channels(seen_channels_path)
    
    current_time = datetime.now().isoformat()
    
    # Add all seed channels to seen list
    for seed_channel in SEED_CHANNELS:
        seed_channel_id = seed_id_map.get(seed_channel)
        if seed_channel_id and seed_channel_id not in seen_ids:
            seen_channels_data.append({
                "Channel_ID": seed_channel_id,
                "Channel_Name": seed_channel,
                "Date_Added": current_time,
                "Processing_Status": "seed",
                "Embedding_Score": None,
                "LLM_Score": None,
                "niche": seed_niche_map.get(seed_channel, "N/A") # <-- ADDED
            })
            seen_ids.add(seed_channel_id)
            print(f"  ✅ Added seed '{seed_channel}' to seen list")
    
    save_seen_channels(seen_channels_data, seen_channels_path)
    
    # --- 4. Initialize Aggregated Results ---
    all_embedding_results = []
    all_llm_results = []
    
    # === PROCESS EACH SEED CHANNEL ===
    for seed_idx, seed_channel in enumerate(SEED_CHANNELS, 1):
        print(f"\n{'='*70}")
        print(f"SEED {seed_idx}/{len(SEED_CHANNELS)}: {seed_channel}")
        print(f"{'='*70}")
        
        # --- MODIFIED: Check for both niche and keywords ---
        if seed_channel not in seed_keywords_map:
            print(f"⚠️  WARNING: No keywords found for '{seed_channel}', skipping")
            continue
        
        if seed_channel not in seed_niche_map:
            print(f"⚠️  WARNING: No niche found for '{seed_channel}', skipping")
            continue
        
        seed_channel_id = seed_id_map.get(seed_channel)
        if not seed_channel_id:
            print(f"⚠️  WARNING: No Channel ID found for '{seed_channel}', skipping")
            continue
        
        seed_keywords = seed_keywords_map[seed_channel]
        seed_niche = seed_niche_map[seed_channel] # <-- ADDED
        
        print(f"Niche: {seed_niche}") # <-- ADDED
        print(f"Keywords: {seed_keywords[:5]}...")
        
        # --- MODIFIED: Pass seed_niche to the processor ---
        embedding_results, llm_results = process_seed_channel(
            seed_channel,
            seed_channel_id,
            seed_keywords,
            seed_niche, # <-- ADDED
            seen_channels_data,
            seen_ids
        )
        
        # Aggregate results
        all_embedding_results.extend(embedding_results)
        all_llm_results.extend(llm_results)
        
        print(f"\n✅ Seed '{seed_channel}' complete:")
        print(f"   Embeddings: {len(embedding_results)} high-similarity channels")
        print(f"   LLM: {len(llm_results)} high-similarity channels")
        
        # Delay between seeds
        if seed_idx < len(SEED_CHANNELS):
            print(f"\n⏸️  Waiting {DELAY_BETWEEN_SEEDS}s before next seed...")
            time.sleep(DELAY_BETWEEN_SEEDS)
    
    # === SAVE INDIVIDUAL RESULTS ===
    print("\n" + "=" * 70)
    print("SAVING INDIVIDUAL RESULTS")
    print("=" * 70)
    
    if all_embedding_results:
        df = pd.DataFrame(all_embedding_results)
        df.to_csv(high_similarity_embeddings_path, index=False, encoding='utf-8-sig')
        print(f"✅ Embeddings (high): {len(df)} results → {high_similarity_embeddings_path.name}")
    
    if all_llm_results:
        df = pd.DataFrame(all_llm_results)
        df.to_csv(high_similarity_llm_path, index=False, encoding='utf-8-sig')
        print(f"✅ LLM (high): {len(df)} results → {high_similarity_llm_path.name}")
    
    # === MERGE AND SAVE ===
    print("\n" + "=" * 70)
    print("MERGING RESULTS")
    print("=" * 70)
    
    df_with_duplicates, df_final = merge_and_deduplicate(
        all_embedding_results,
        all_llm_results
    )
    
    # Save with duplicates
    if not df_with_duplicates.empty:
        df_with_duplicates.to_csv(merged_with_duplicates_path, index=False, encoding='utf-8-sig')
        print(f"✅ Saved (with duplicates): {len(df_with_duplicates)} → {merged_with_duplicates_path.name}")
    
    # Save deduplicated final
    if not df_final.empty:
        df_final.to_csv(merged_final_path, index=False, encoding='utf-8-sig')
        print(f"✅ Saved (deduplicated): {len(df_final)} → {merged_final_path.name}")
    
    # === FINAL SUMMARY ===
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Seeds processed: {len(SEED_CHANNELS)}")
    print(f"Total embedding high-similarity: {len(all_embedding_results)}")
    print(f"Total LLM high-similarity: {len(all_llm_results)}")
    print(f"Combined (with duplicates): {len(df_with_duplicates)}")
    print(f"Final unique channels: {len(df_final)}")
    print()
    
    if not df_final.empty:
        print("Breakdown by seed:")
        print(df_final.groupby('Seed_Channel_Name').size())
    
    print()
    print(f"⏱️  Total runtime: {elapsed/60:.1f} minutes")
    print("=" * 70)
    print("\n✅ ALL PROCESSING COMPLETE!")


if __name__ == "__main__":
    main()