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
keywords_file_path = base_dir / "channel_fingerprints_gpt.json"
seed_video_data_path = base_dir / "sample_videos.csv"
seen_channels_path = base_dir / "seen_channels.csv"


# Output Files (6 files: 4 for individual + 2 for merged)
all_results_embeddings_path = base_dir / "all_discovered_embeddings.csv"
high_similarity_embeddings_path = base_dir / "high_similarity_embeddings.csv"
all_results_llm_path = base_dir / "all_discovered_llm.csv"
high_similarity_llm_path = base_dir / "high_similarity_llm.csv"

# NEW: Merged output files
merged_with_duplicates_path = base_dir / "FINAL_SEED_SUBSCRIBE_MERGED_DUPLICATES.csv"
merged_final_path = base_dir / "FINAL_SEED_SUBSCRIBE_MERGED.csv"


# Settings
MODEL_PROVIDER = "gpt"  # "gemini" or "gpt"

# Multi-seed configuration
SEED_CHANNELS = [
    "Ali Abdaal"
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
            max_results_per_search=30,
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
                "Seed_Keywords": ", ".join(seed_keywords),
                "Discovered_Keywords": ", ".join(cand_keywords),
                "Level": 1,
                "Timestamp": datetime.now().isoformat()
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