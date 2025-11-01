import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime
import csv  # <-- ADDED for save-as-you-go
import os   # <-- ADDED for save-as-you-go


# --- Import Helper Functions ---
from utils.youtube_utils import (
    search_videos_multi_focused,
    get_channel_metadata_batch,
    fetch_recent_videos,
)

# --- MODIFIED: Import all required LLM functions ---
from utils.fingerprint_llm_utils import (
    get_channel_fingerprint_oneshot,
    calculate_embedding_similarity_hybrid,
    calculate_llm_similarity,
    # detect_channel_language_llm # <-- ADDED
)


# === CONFIGURATION ===
base_dir = Path(__file__).resolve().parent

# --- Input Files ---
keywords_file_path = base_dir / "channel_fingerprints_gpt_moon.json"
seed_video_data_path = base_dir / "sample_videos_moon.csv"
seen_channels_path = base_dir / "seen_channels_moon.csv" # High-level log

# --- NEW: Primary Output File (The "Everything" Cache) ---
intermediate_data_path = base_dir / "phase3_intermediate_data_moon.csv"

# --- NEW: Define ALL columns for the intermediate file ---
INTERMEDIATE_COLUMN_ORDER = [
    "Seed_Channel_Name", "Seed_Channel_ID", "Seed_Niche",
    "Discovered_Channel_Name", "Discovered_Channel_ID", "Discovered_Channel_URL",
    "Discovered_Subs", "Discovered_Video_Count", "Discovered_Country",
    "Discovered_Niche", # This will store the "niche" field
    "Discovered_Format", # <-- NEW
    "Discovered_Intent", # <-- NEW
    "Discovered_POV",    # <-- NEW
    "Discovered_Audience", # <-- NEW
    "Discovered_Keywords_JSON", # <-- NEW (for the keywords dict)
    "LLM_Score", "Embedding_Score",
    "Discovered_Channel_Description",
    "Seed_Keywords_JSON", # <-- NEW (changed from Seed_Keywords)
    "Level", "Timestamp",
]

AUTO_KEEP_COUNTRIES = [
    'US',  # United States
    'GB',  # United Kingdom
    'CA',  # Canada
    'AU',  # Australia
    'NZ',  # New Zealand
    'NG',  # Nigeria
    'Unknown' # Keep 'Unknown' for now, or remove to filter them
]

# --- Settings ---
MODEL_PROVIDER = "gpt"  # "gemini" or "gpt" (as requested, not changed)
SEED_CHANNELS = ["Moon"] # Your seed channels

# --- Filtering (for Data Collection) ---
MIN_SUBSCRIBERS = 10000
MIN_VIDEOS = 6
VIDEOS_PER_CANDIDATE = 20 # Your change, kept as requested

# --- Rate Limiting ---
DELAY_BETWEEN_CANDIDATES = 3
DELAY_BETWEEN_SEEDS = 10


# === HELPER FUNCTIONS ===

# --- NEW: Helper for Caching ---
def load_processed_channels_from_intermediate(file_path):
    """Loads the intermediate CSV and returns a set of processed channel IDs."""
    if not file_path.exists():
        return set()
    try:
        # Only read the one column we need, very fast
        df = pd.read_csv(file_path, usecols=["Discovered_Channel_ID"])
        return set(df["Discovered_Channel_ID"].astype(str).tolist())
    except Exception as e:
        print(f"  ⚠️  Could not read intermediate file {file_path.name}: {e}")
        return set()

# --- NEW: Helper for Caching ---
def append_to_intermediate_csv(data_dict, file_path):
    """Appends a single row (dict) to the intermediate CSV file."""
    # Check if file exists to write header
    file_exists = file_path.exists()
    
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8-sig') as f:
            # Use our global column order, ignore extra keys
            writer = csv.DictWriter(f, fieldnames=INTERMEDIATE_COLUMN_ORDER, extrasaction='ignore')
            
            if not file_exists:
                writer.writeheader()  # Write header only if file is new
            
            writer.writerow(data_dict)
            
    except Exception as e:
        print(f"  ❌ ERROR appending to {file_path.name}: {e}")


def save_seen_channels(seen_data_list, file_path):
    """Saves seen channels to CSV with deduplication"""
    if not seen_data_list:
        return
    try:
        df = pd.DataFrame(seen_data_list)
        all_cols = [
            "Channel_ID", "Channel_Name", "Date_Added", "Processing_Status",
            "Embedding_Score", "LLM_Score", "niche",
        ]
        for col in all_cols:
            if col not in df.columns:
                df[col] = None
        df = df[all_cols]
        df = df.sort_values(by="Date_Added").drop_duplicates(
            subset=["Channel_ID"], keep="last"
        )
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        # print(f"  💾 Saved {len(df)} seen channels to {file_path.name}") # Less verbose
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
                if "niche" not in df.columns: df["niche"] = None
                df["niche"] = df["niche"].fillna("N/A")
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
    """Update processing status for a channel"""
    for entry in seen_data:
        if entry["Channel_ID"] == channel_id:
            entry["Processing_Status"] = status
            break


def _update_status_with_scores(
    seen_data, channel_id, status, score_emb, score_llm, niche
):
    """Update status with BOTH scores AND niche"""
    for entry in seen_data:
        if entry["Channel_ID"] == channel_id:
            entry["Processing_Status"] = status
            entry["Embedding_Score"] = score_emb
            entry["LLM_Score"] = score_llm
            entry["niche"] = niche
            break


# --- MODIFIED: Main processing function ---
def process_seed_channel(
    seed_channel,
    seed_channel_id,
    seed_keywords,  # <-- This is now a DICTIONARY
    seed_niche,
    seen_channels_data,
    seen_ids,
    processed_ids,  # <-- NEW: Set of already processed IDs
    intermediate_file_path # <-- NEW: Path to save data
):
    """
    Process a single seed channel, saves results row-by-row
    Returns: (number_of_new_channels_processed)
    """

    print("\n" + "=" * 70)
    print(f"PROCESSING SEED: {seed_channel} (Niche: {seed_niche})")
    print(f"  (Will skip {len(processed_ids)} channels already in {intermediate_file_path.name})")
    print("=" * 70)

    current_time = datetime.now().isoformat()
    new_channels_processed_count = 0

    # --- NEW: Flatten seed keywords for embedding score & search ---
    seed_keywords_list = []
    if isinstance(seed_keywords, dict):
        for k, v in seed_keywords.items():
            if isinstance(v, list): seed_keywords_list.extend(v)
    elif isinstance(seed_keywords, list):
        print("⚠️ Seed keywords are a list, not a dict. Using as-is.")
        seed_keywords_list = seed_keywords # Handle old format just in case
    
    if not seed_keywords_list:
        print(f"❌ No seed keywords found for {seed_channel}. Skipping seed.")
        return 0

    # --- STEP 1: Multi-Focused Search ---
    print(f"\n🔍 STEP 1: Searching YouTube with {len(seed_keywords_list)} keywords...")
    try:
        candidate_ids = search_videos_multi_focused(
            seed_keywords_list, # Use the flattened list
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
                "Embedding_Score": None, "LLM_Score": None, "niche": None,
            })
            seen_ids.add(meta["id"])
    save_seen_channels(seen_channels_data, seen_channels_path) # Save new discoveries to log

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
    save_seen_channels(seen_channels_data, seen_channels_path) # Save filter status to log

    # --- STEP 4: Filter against *already processed* channels (from cache) ---
    print(f"\n🔍 STEP 4: Filtering against {len(processed_ids)} already processed channels...")
    candidates_to_process = []
    for c in qualified:
        if c['id'] not in processed_ids:
            candidates_to_process.append(c)
        # else:
            # print(f"  - Skipping {c['name']} (already processed)")
            
    if not candidates_to_process:
        print("  ✅ No new qualified candidates to process for this seed.")
        return 0 # Return 0 new channels
    print(f"  🎯 {len(candidates_to_process)} new candidates to score (out of {len(qualified)} qualified).")

    # --- STEP 5: Score Candidates (Save-as-you-go) ---
    print(f"\n🎯 STEP 5: Scoring {len(candidates_to_process)} candidates...")

    for i, candidate in enumerate(candidates_to_process, 1):
        print(f"\n  [{i}/{len(candidates_to_process)}] {candidate['name']}")
        print(f"     Subs: {candidate['subscribers']:,} | Videos: {candidate['video_count']:,}")

        try:
            # ... (Fetch videos and description - this part is the same) ...
            # ... (video_df = pd.DataFrame(videos)) ...
            print(f"     Fetching {VIDEOS_PER_CANDIDATE} videos...")
            videos, cand_desc = fetch_recent_videos(
                candidate["id"], max_results=VIDEOS_PER_CANDIDATE, filter_shorts=False
            )
            
            if len(videos) < 3:
                print(f"     ⚠️  Only {len(videos)} videos found, logging and skipping")
                _update_status(seen_channels_data, candidate["id"], "skipped_few_videos")
                continue

            video_df = pd.DataFrame(videos)
            video_titles_list = video_df["title"].tolist()
            # --- Step 5a: One-Shot Fingerprint (Replaces 3 steps) ---
            print(f"     Extracting one-shot fingerprint (Profile + Keywords)...")
            fingerprint_data = get_channel_fingerprint_oneshot(
                channel_name=candidate["name"],
                channel_description=cand_desc,
                video_df=video_df,
                model_provider=MODEL_PROVIDER
            )

            if not fingerprint_data or "profile" not in fingerprint_data or "keywords" not in fingerprint_data:
                print(f"     ⚠️  One-shot fingerprint generation failed.")
                _update_status(seen_channels_data, candidate["id"], "failed_fingerprint")
                continue

            cand_profile = fingerprint_data.get("profile", {})
            cand_keywords_dict = fingerprint_data.get("keywords", {})

            # Flatten candidate keywords for embedding score
            cand_keywords_list = []
            for k, v in cand_keywords_dict.items():
                if isinstance(v, list): cand_keywords_list.extend(v)


            # For now, let's just save the data. We'll fix the scoring next.
            similarity_llm = 0.0 # Placeholder
            similarity_embeddings = calculate_embedding_similarity_hybrid(
                seed_keywords_list, cand_keywords_list
            )
            print(f"     📊 Embeddings: {similarity_embeddings:.3f}")

            # --- Step 5e: Build the "Everything" Dictionary ---
            candidate_full_data = {
                "Seed_Channel_Name": seed_channel,
                "Seed_Channel_ID": seed_channel_id,
                "Seed_Niche": seed_niche,
                "Discovered_Channel_Name": candidate["name"],
                "Discovered_Channel_ID": candidate["id"],
                "Discovered_Channel_URL": candidate["url"],
                "Discovered_Subs": candidate["subscribers"],
                "Discovered_Video_Count": candidate["video_count"], # <-- ADDED
                "Discovered_Country": candidate.get("country", "Unknown"),
                "Discovered_Niche": cand_profile.get("niche", "Unknown"),
                "Discovered_Format": cand_profile.get("format", "Unknown"),
                "Discovered_Intent": cand_profile.get("intent", "Unknown"),
                "Discovered_POV": cand_profile.get("ideology", "Unknown"),
                "Discovered_Audience": cand_profile.get("target_audience", "Unknown"),
                "Discovered_Keywords_JSON": json.dumps(cand_keywords_dict), # Save keywords JSON
                "LLM_Score": similarity_llm,                          # <-- ADDED
                "Embedding_Score": similarity_embeddings,             # <-- ADDED        # <-- ADDED
                "Discovered_Keywords": ", ".join(cand_keywords_list), # Save flattened list
                "Seed_Keywords": ", ".join(seed_keywords_list),       # Save flattened list
                "Level": 1,
                "Timestamp": datetime.now().isoformat(),
            }

            # --- Step 5f: SAVE TO DISK (This is the cache!) ---
            append_to_intermediate_csv(candidate_full_data, intermediate_data_path)
            new_channels_processed_count += 1
            print(f"     ✅ Saved data for {candidate['name']} to {intermediate_data_path.name}")

            # Update high-level log
            _update_status_with_scores(
                seen_channels_data,
                candidate["id"],
                "scored",
                similarity_embeddings,
                similarity_llm,
                cand_profile.get("niche", "Unkown")
            )
            save_seen_channels(seen_channels_data, seen_channels_path)
            
            time.sleep(DELAY_BETWEEN_CANDIDATES)

        except Exception as e:
            print(f"     ❌ Error on {candidate['name']}: {str(e)[:100]}")
            _update_status(seen_channels_data, candidate["id"], "error_scoring")

    # Return count of channels we just processed
    return new_channels_processed_count


# === MAIN FUNCTION ===
def main():
    start_time = time.time()
    print("=" * 70)
    print(f"DATA COLLECTION PIPELINE (PHASE 1)")
    print(f"Seeds: {', '.join(SEED_CHANNELS)}")
    print(f"LLM Provider: {MODEL_PROVIDER.upper()}")
    print("=" * 70)

    # --- 1. Load Keywords AND Niches (CORRECTED) ---
    print(f"\n📖 Loading keywords & niches from {keywords_file_path.name}...")
    seed_keywords_map = {}
    seed_niche_map = {}
    try:
        with open(keywords_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for channel_name, details in data.get("channels", {}).items():
                if details.get("niche"):
                    seed_niche_map[channel_name] = details["niche"]
                
                if details.get("keywords"):
                    keywords_data = details["keywords"] # This is the DICTIONARY
                    
                    # --- THIS IS THE FIX ---
                    # We now save the WHOLE DICTIONARY
                    seed_keywords_map[channel_name] = keywords_data
                    # --- END OF FIX ---

                    # This part is just for printing
                    all_keywords_list_for_print = []
                    if isinstance(keywords_data, dict):
                        for category, keyword_list in keywords_data.items():
                            if isinstance(keyword_list, list):
                                all_keywords_list_for_print.extend(keyword_list)
                    elif isinstance(keywords_data, list):
                        all_keywords_list_for_print = keywords_data
                    
                    print(
                        f"  📌 {channel_name}: {len(all_keywords_list_for_print)} keywords from {len(keywords_data) if isinstance(keywords_data, dict) else 1} categories"
                    )

        print(f"  ✅ Loaded keywords for {len(seed_keywords_map)} channels")
        print(f"  ✅ Loaded niches for {len(seed_niche_map)} channels")
    except Exception as e:
        print(f"❌ ERROR loading keywords/niches: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Load Seed Channel ID Mapping ---
    print(f"\n🗺️  Loading channel IDs from {seed_video_data_path.name}...")
    try:
        df_videos = pd.read_csv(seed_video_data_path)
        if ("Channel_Name" not in df_videos.columns or "Channel_ID" not in df_videos.columns):
            print("❌ ERROR: CSV must have 'Channel_Name' and 'Channel_ID' columns")
            return
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
    print(f"\n📂 Loading seen channels log from {seen_channels_path.name}...")
    seen_channels_data, seen_ids = load_seen_channels(seen_channels_path)
    current_time = datetime.now().isoformat()
    for seed_channel in SEED_CHANNELS:
        seed_channel_id = seed_id_map.get(seed_channel)
        if seed_channel_id and seed_channel_id not in seen_ids:
            seen_channels_data.append({
                "Channel_ID": seed_channel_id, "Channel_Name": seed_channel,
                "Date_Added": current_time, "Processing_Status": "seed",
                "Embedding_Score": None, "LLM_Score": None,
                "niche": seed_niche_map.get(seed_channel, "N/A"),
            })
            seen_ids.add(seed_channel_id)
            print(f"  ✅ Added seed '{seed_channel}' to seen list")
    save_seen_channels(seen_channels_data, seen_channels_path)

    # --- 4. Load ALREADY PROCESSED channels (from cache) ---
    print(f"\n🔄 Loading already processed channels from {intermediate_data_path.name}...")
    processed_channel_ids = load_processed_channels_from_intermediate(intermediate_data_path)
    print(f"  ✅ Found {len(processed_channel_ids)} channels in cache to skip.")

    # --- 5. Initialize Aggregated Results ---
    total_new_channels_found = 0

    # === PROCESS EACH SEED CHANNEL ===
    for seed_idx, seed_channel in enumerate(SEED_CHANNELS, 1):
        print(f"\n{'=' * 70}")
        print(f"SEED {seed_idx}/{len(SEED_CHANNELS)}: {seed_channel}")
        print(f"{'=' * 70}")

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

        seed_keywords = seed_keywords_map[seed_channel] # This is now the DICT
        seed_niche = seed_niche_map[seed_channel]
        print(f"Niche: {seed_niche}")
        # print(f"Keywords: {seed_keywords_map[seed_channel][:5]}...") # Can't print dict like this

        # --- MODIFIED: Pass the new args to the processor ---
        new_channels_this_seed = process_seed_channel(
            seed_channel,
            seed_channel_id,
            seed_keywords, # Pass the DICT
            seed_niche,
            seen_channels_data,
            seen_ids,
            processed_channel_ids,     # <-- PASS THE SET
            intermediate_data_path # <-- PASS THE PATH
        )

        # Aggregate results
        total_new_channels_found += new_channels_this_seed

        print(f"\n✅ Seed '{seed_channel}' complete:")
        print(f"   Processed and saved {new_channels_this_seed} new channels this run.")

        # Delay between seeds
        if seed_idx < len(SEED_CHANNELS):
            print(f"\n⏸️  Waiting {DELAY_BETWEEN_SEEDS}s before next seed...")
            time.sleep(DELAY_BETWEEN_SEEDS)

    # === FINAL SUMMARY (REWRITTEN) ===
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Seeds processed: {len(SEED_CHANNELS)}")
    print(f"Total new channels processed this run: {total_new_channels_found}")
    print(f"Total processed channels in cache: {len(processed_channel_ids) + total_new_channels_found}")
    print(f"⏱️  Total runtime: {elapsed / 60:.1f} minutes")
    print("\n✅ Next step: Run 'phase3_filter.py' to analyze results.")
    print(f"   Data is saved in: {intermediate_data_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()