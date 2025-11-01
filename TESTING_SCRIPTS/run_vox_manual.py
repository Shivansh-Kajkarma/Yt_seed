import pandas as pd
import numpy as np
import os
import sys
import json
import re # <-- Import re for safe filenames
from typing import List, Dict
from collections import Counter

# --- 1. Add local utils to the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes utils are in 'utils/' subfolder, change if they are in the same folder
sys.path.append(os.path.join(script_dir, '..')) 
sys.path.append(script_dir) # Add current dir too

try:
    # Try importing from a 'utils' subfolder first
    import utils.youtube_utils as yt_utils
    import utils.fingerprint_llm_utils as llm_utils
    print("✅ Successfully imported local utils (from 'utils/' folder).")
except ImportError:
    try:
        # Fallback: try importing from the same folder
        import utils.youtube_utils as yt_utils
        import utils.fingerprint_llm_utils as llm_utils
        print("✅ Successfully imported local utils (from same folder).")
    except ImportError as e:
        print(f"❌ CRITICAL ERROR: Could not import local utils.")
        print(f"Make sure 'youtube_utils.py' and 'fingerprint_llm_utils.py' are accessible.")
        print(f"Error: {e}")
        sys.exit(1)

# --- 2. File Paths & Config ---
SEED_CHANNEL_NAME = "Vox"
SEED_VIDEOS_CSV = '/home/rareboy/Internship/Kajkarma/sample_videos.csv'

# --- NEW CACHE LOCATIONS ---
VIDEO_CACHE_DIR = "video_cache" # For video CSVs
CHANNEL_ID_CACHE_FILE = "channel_id_cache.json" # For Channel IDs
OUTPUT_CSV_FILE = f"retest_{SEED_CHANNEL_NAME.lower()}_specific_competitors.csv"

# Your hand-picked list of competitors to test
CANDIDATES_TO_TEST = [
    "Insider", "VICE News", "NowThis World", "AJ+", "The Atlantic",
    "Wired", "Cheddar", "Johnny Harris", "Wendover Productions",
    "PolyMatter", "RealLifeLore", "ColdFusion", "Veritasium",
    "TED-Ed", "CGP Grey", "Kurzgesagt", "Vice"
]

# --- 3. Helper Functions ---

def flatten_keywords(kw_dict: dict) -> List[str]:
    """ Flattens the keyword dictionary into a list for the embedding model. """
    all_kws = []
    if not isinstance(kw_dict, dict):
        return []
    for kws in kw_dict.values():
        if isinstance(kws, list):
            all_kws.extend(kws)
    return list(set(all_kws))

def load_id_cache() -> Dict[str, str]:
    """Loads the Channel ID cache from JSON."""
    if os.path.exists(CHANNEL_ID_CACHE_FILE):
        try:
            with open(CHANNEL_ID_CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_id_cache(cache: Dict[str, str]):
    """Saves the Channel ID cache to JSON."""
    with open(CHANNEL_ID_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_safe_filename(channel_name: str) -> str:
    """Creates a filesystem-safe name for caching."""
    safe_name = re.sub(r'[^\w_.-]', '', channel_name.replace(" ", "_"))
    return f"{safe_name}_videos.csv"

# --- 4. Main Test Function ---
def run_specific_competitor_test():
    print(f"--- Starting Specific Competitor Test for '{SEED_CHANNEL_NAME}' ---")
    
    if not os.path.exists(VIDEO_CACHE_DIR):
        os.makedirs(VIDEO_CACHE_DIR)
        print(f"Created video cache directory: {VIDEO_CACHE_DIR}")

    # --- A. Load Seed Channel ("Vox") Fingerprint ---
    print(f"\n[STEP 1/3] Loading and fingerprinting Seed Channel '{SEED_CHANNEL_NAME}'...")
    try:
        df_seed_videos = pd.read_csv(SEED_VIDEOS_CSV)
        df_seed_videos = df_seed_videos[df_seed_videos['Channel_Name'] == SEED_CHANNEL_NAME]
        if df_seed_videos.empty:
            print(f"❌ No video data found for '{SEED_CHANNEL_NAME}' in {SEED_VIDEOS_CSV}")
            return
    except Exception as e:
        print(f"❌ Failed to load {SEED_VIDEOS_CSV}: {e}")
        return

    seed_fingerprint = llm_utils.get_channel_fingerprint_oneshot(
        channel_name=SEED_CHANNEL_NAME,
        channel_description=df_seed_videos['channel_description'].iloc[0],
        video_df=df_seed_videos,
        model_provider="gpt"
    )
    
    if not seed_fingerprint:
        print(f"❌ Failed to generate seed fingerprint for '{SEED_CHANNEL_NAME}'. Exiting.")
        return
        
    seed_profile = seed_fingerprint.get("profile", {})
    seed_keywords_dict = seed_fingerprint.get("keywords", {})
    seed_keywords_flat = flatten_keywords(seed_keywords_dict)
    
    seed_categories_str = ", ".join(seed_keywords_dict.keys())
    seed_keywords_str = ", ".join(seed_keywords_flat)
    
    print(f"✅ '{SEED_CHANNEL_NAME}' Fingerprint Generated:")
    print(json.dumps(seed_profile, indent=2))

    # --- B. Loop, Fetch, Cache, and Score Candidates ---
    print(f"\n[STEP 2/3] Processing {len(CANDIDATES_TO_TEST)} specific candidates...")
    
    results = []
    id_cache = load_id_cache()
    
    for i, cand_name in enumerate(CANDIDATES_TO_TEST):
        
        print(f"\n--- Processing {cand_name} ({i+1}/{len(CANDIDATES_TO_TEST)}) ---")
        
        # 1. Get Channel ID (from Cache or API)
        cand_id = id_cache.get(cand_name)
        if not cand_id:
            print(f"  Resolving Channel ID for '{cand_name}' (Live API call)...")
            try:
                cand_id = yt_utils.extract_channel_id(cand_name)
                if not cand_id:
                    print(f"  ❌ Could not resolve Channel ID. Skipping.")
                    results.append({'Channel_Name': cand_name, 'Status': 'ID Resolve Failed'})
                    continue
                
                print(f"  > Found & Caching ID: {cand_id}")
                id_cache[cand_name] = cand_id
                save_id_cache(id_cache)
            except Exception as e:
                print(f"  ❌ API Error resolving ID: {e}")
                results.append({'Channel_Name': cand_name, 'Status': 'ID Resolve Error'})
                continue
        else:
            print(f"  Found cached Channel ID: {cand_id}")

        # 2. Get Video Data (from Cache or API)
        cache_file_path = os.path.join(VIDEO_CACHE_DIR, get_safe_filename(cand_name))
        
        try:
            if os.path.exists(cache_file_path):
                print(f"  Loading cached videos from {cache_file_path}...")
                df_cand_videos = pd.read_csv(cache_file_path)
                cand_desc = df_cand_videos['channel_description'].iloc[0]
            else:
                print(f"  Fetching 30 recent videos for {cand_name} (Live API call)...")
                cand_videos_list, cand_desc = yt_utils.fetch_recent_videos(
                    channel_id=cand_id,
                    max_results=30
                )
                
                if not cand_videos_list:
                    print(f"  ⚠️ No videos returned. Skipping.")
                    results.append({'Channel_Name': cand_name, 'Status': 'No Videos Found'})
                    continue
                
                df_cand_videos = pd.DataFrame(cand_videos_list)
                
                df_cand_videos['Channel_Name'] = cand_name
                df_cand_videos['Channel_ID'] = cand_id
                df_cand_videos['channel_description'] = cand_desc
                df_cand_videos.to_csv(cache_file_path, index=False)
                print(f"  ✅ Saved {len(df_cand_videos)} videos to cache.")
        
        except Exception as e:
            print(f"  ❌ Failed to fetch or load videos: {e}")
            results.append({'Channel_Name': cand_name, 'Status': 'Video Fetch Error'})
            continue
            
        # 3. Generate Candidate Fingerprint (Live LLM Call)
        print(f"  Fingerprinting {cand_name}...")
        cand_fingerprint = llm_utils.get_channel_fingerprint_oneshot(
            channel_name=cand_name,
            channel_description=cand_desc,
            video_df=df_cand_videos,
            model_provider="gpt"
        )
        
        if not cand_fingerprint:
            print(f"  ❌ Failed to generate fingerprint. Skipping.")
            results.append({'Channel_Name': cand_name, 'Status': 'Fingerprint Failed'})
            continue
            
        cand_profile = cand_fingerprint.get("profile", {})
        cand_keywords_dict = cand_fingerprint.get("keywords", {})
        cand_keywords_flat = flatten_keywords(cand_keywords_dict)
        cand_categories_str = ", ".join(cand_keywords_dict.keys())
        cand_keywords_str = ", ".join(cand_keywords_flat)

        # 4. Calculate NEW Scores (BOTH V3 and V4)
        print(f"  Scoring {cand_name} against '{SEED_CHANNEL_NAME}'...")
        
        # V3 - Holistic (Profile-Only)
        holistic_data = llm_utils.calculate_profile_score_llm_holistic(
            seed_profile=seed_profile,
            candidate_profile=cand_profile,
            candidate_channel_name=cand_name,
            seed_channel_name=SEED_CHANNEL_NAME,
            model_provider="gpt"
        )
        
        # V4 - Audience Match (Keywords-Aware)
        audience_data = llm_utils.calculate_profile_score_llm_with_keywords(
            seed_profile=seed_profile,
            candidate_profile=cand_profile,
            seed_keywords=seed_keywords_dict,
            candidate_keywords=cand_keywords_dict,
            seed_channel_name=SEED_CHANNEL_NAME,
            candidate_channel_name=cand_name,
            model_provider="gpt"
        )
        
        # Embedding Score
        new_emb_score = 0.0
        if llm_utils.embedding_model:
             new_emb_score = llm_utils.calculate_embedding_similarity_hybrid(
                 seed_keywords_flat,
                 cand_keywords_flat
             )

        # 5. Store Results (--- THIS IS THE UPDATED BLOCK ---)
        # Using the exact keys from your logs
        results.append({
            "Channel_Name": cand_name,
            "Status": "Success",
            
            # V3 Scores (from your logs)
            "Holistic_Similarity_Score": holistic_data.get("similarity_score", 0.0),
            "Holistic_Competitor_Score": holistic_data.get("competitor_score", 0.0),
            "Holistic_Audience_Overlap": holistic_data.get("audience_overlap_score", 0.0),
            "Holistic_Reasoning": holistic_data.get("reason", "N/A"),
            
            # V4 Scores (from your logs)
            "Audience_Match_Score": audience_data.get("audience_match_score", 0.0),
            "Audience_Match_Reasoning": audience_data.get("reason", "N/A"),
            
            # Embedding Score
            "Embedding_Score": new_emb_score,
            
            # Profile Data
            "Cand_Ideology": cand_profile.get("ideology", "N/A"),
            "Cand_Format": cand_profile.get("format", "N/A"),
            "Cand_Niche": cand_profile.get("niche", "N/A"),
            "Seed_Ideology": seed_profile.get("ideology", "N/A"),
            "Seed_Format": seed_profile.get("format", "N/A"),
            "Seed_Niche": seed_profile.get("niche", "N/A"),
            
            # Keyword Data
            "Seed_Categories": seed_categories_str,
            "Cand_Categories": cand_categories_str,
            "Seed_Keywords": seed_keywords_str,
            "Cand_Keywords": cand_keywords_str
        })
        print(f"  ✅ Scores: Holistic_Comp={holistic_data.get('competitor_score', 0.0)}, Audience_Match={audience_data.get('audience_match_score', 0.0)}")


    # --- C. Show Final Report ---
    print(f"\n[STEP 3/3] --- {SEED_CHANNEL_NAME} SPECIFIC COMPETITOR TEST COMPLETE ---")
    if not results:
        print("No channels were successfully re-scored.")
        return

    df_report = pd.DataFrame(results)
    
    # Sort by the new Audience Match score
    df_report = df_report.sort_values(by="Audience_Match_Score", ascending=False)
    
    try:
        # --- THIS PRINTS THE SUMMARY TO YOUR TERMINAL ---
        print("\n📊 FINAL SUMMARY (Sorted by Audience Match Score):")
        cols_to_print = ["Channel_Name", "Holistic_Competitor_Score", "Audience_Match_Score", "Status"]
        print(df_report[cols_to_print].to_markdown(index=False, floatfmt=".3f"))
    except ImportError:
        print("Install 'tabulate' (pip install tabulate) to see a formatted table here.")
        print(df_report) # Fallback to default print
    
    df_report.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\n✅ Full report saved to {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    try:
        import tabulate
    except ImportError:
        print("⚠️  Warning: 'tabulate' module not found. `pip install tabulate` for cleaner log output.")
        
    if not llm_utils.embedding_model:
        print("❌ Embedding model did not load.")
        print("Please check your internet connection and SentenceTransformer installation.")
    else:
        run_specific_competitor_test()