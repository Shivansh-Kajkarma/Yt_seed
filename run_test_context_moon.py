import pandas as pd
import numpy as np
import os
import sys
import json
from typing import List, Dict
from collections import Counter # Import Counter

# --- 1. Add local utils to the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    import utils.youtube_utils as yt_utils
    import utils.fingerprint_llm_utils as llm_utils
    print("✅ Successfully imported local utils.")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import local utils.")
    print(f"Make sure 'youtube_utils.py' and 'fingerprint_llm_utils.py' are in the same folder.")
    print(f"Error: {e}")
    sys.exit(1)

# --- 2. File Paths & Config ---
SEED_VIDEOS_CSV = 'sample_videos_moon.csv'
CANDIDATE_DATA_CSV = '/home/rareboy/Internship/Kajkarma/phase3_intermediate_data_moon.csv'

CHANNELS_TO_RETEST = [
    "Vox",
    "Tom Nicholas",
    "The Young Turks",
    "David Pakman Show",
    "The Rational National",
    "Pop Culture Detective",
    "Noah Samsen",
    "Mina Le",
    "Tara Mooknee",
    "tiffanyferg"
]

# --- 3. Helper Function to Bridge Incompatibility ---
def flatten_keywords(kw_dict: dict) -> List[str]:
    """
    Takes the {'Category': [...]} dict from the new fingerprint
    and flattens it into a single list for the old embedding function.
    """
    all_kws = []
    if not isinstance(kw_dict, dict):
        return []
    for kws in kw_dict.values():
        if isinstance(kws, list):
            all_kws.extend(kws)
    return list(set(all_kws)) # Return unique keywords

# --- 4. Main Test Function ---
def run_context_aware_retest():
    print("--- Starting Context-Aware Re-Test for 'Moon' ---")
    
    # --- A. Load Seed Channel ("Moon") Fingerprint ---
    print(f"\n[STEP 1/4] Loading and fingerprinting Seed Channel 'Moon'...")
    try:
        df_seed_videos = pd.read_csv(SEED_VIDEOS_CSV)
        df_seed_videos = df_seed_videos[df_seed_videos['Channel_Name'] == 'Moon']
        if df_seed_videos.empty:
            print(f"❌ No video data found for 'Moon' in {SEED_VIDEOS_CSV}")
            return
    except Exception as e:
        print(f"❌ Failed to load {SEED_VIDEOS_CSV}: {e}")
        return

    seed_fingerprint = llm_utils.get_channel_fingerprint_oneshot(
        channel_name="Moon",
        channel_description=df_seed_videos['channel_description'].iloc[0],
        video_df=df_seed_videos,
        model_provider="gpt"
    )
    
    if not seed_fingerprint:
        print("❌ Failed to generate seed fingerprint for 'Moon'. Exiting.")
        return
        
    seed_profile = seed_fingerprint.get("profile", {})
    seed_keywords_dict = seed_fingerprint.get("keywords", {})
    seed_keywords_flat = flatten_keywords(seed_keywords_dict)
    
    # --- NEW: Create string versions for caching ---
    seed_categories_str = ", ".join(seed_keywords_dict.keys())
    seed_keywords_str = ", ".join(seed_keywords_flat)
    
    print(f"✅ 'Moon' Fingerprint Generated:")
    print(json.dumps(seed_profile, indent=2))
    print(f"  > Seed Categories: {seed_categories_str}")

    # --- B. Load and Filter Candidate Channels ---
    print(f"\n[STEP 2/4] Loading Candidate data from {CANDIDATE_DATA_CSV}...")
    try:
        df_candidates_all = pd.read_csv(CANDIDATE_DATA_CSV)
    except Exception as e:
        print(f"❌ Failed to load {CANDIDATE_DATA_CSV}: {e}")
        return

    df_candidates_test = df_candidates_all[
        df_candidates_all['Discovered_Channel_Name'].isin(CHANNELS_TO_RETEST)
    ].drop_duplicates(subset=['Discovered_Channel_ID']).reset_index()
    
    print(f"✅ Found {len(df_candidates_test)} of your 10 target channels to re-test.")

    # --- C. Loop, Fetch, and Re-Score Candidates ---
    print(f"\n[STEP 3/4] Fetching, Fingerprinting, and Re-Scoring {len(df_candidates_test)} candidates...")
    
    results = []
    
    for index, row in df_candidates_test.iterrows():
        cand_name = row['Discovered_Channel_Name']
        cand_id = row['Discovered_Channel_ID']
        old_llm_score = row['LLM_Score']
        old_emb_score = row['Embedding_Score']
        
        print(f"\n--- Processing {cand_name} ({index+1}/{len(df_candidates_test)}) ---")
        
        # 1. Fetch Candidate Videos (LIVE)
        print(f"  Fetching recent videos for {cand_name}...")
        try:
            cand_videos_list, cand_desc = yt_utils.fetch_recent_videos(
                channel_id=cand_id,
                max_results=30
            )
            if not cand_videos_list:
                print(f"  ⚠️ No videos returned for {cand_name}. Skipping.")
                continue
            df_cand_videos = pd.DataFrame(cand_videos_list)
        except Exception as e:
            print(f"  ❌ Failed to fetch videos for {cand_name}: {e}")
            continue
            
        # 2. Generate Candidate Fingerprint
        print(f"  Fingerprinting {cand_name}...")
        cand_fingerprint = llm_utils.get_channel_fingerprint_oneshot(
            channel_name=cand_name,
            channel_description=cand_desc,
            video_df=df_cand_videos,
            model_provider="gpt"
        )
        
        if not cand_fingerprint:
            print(f"  ❌ Failed to generate fingerprint for {cand_name}. Skipping.")
            continue
            
        cand_profile = cand_fingerprint.get("profile", {})
        cand_keywords_dict = cand_fingerprint.get("keywords", {})
        cand_keywords_flat = flatten_keywords(cand_keywords_dict)
        
        # --- NEW: Create string versions for caching ---
        cand_categories_str = ", ".join(cand_keywords_dict.keys())
        cand_keywords_str = ", ".join(cand_keywords_flat)

        # 3. Calculate NEW Scores
        print(f"  Scoring {cand_name} against 'Moon'...")
        
        # NEW CONTEXT-AWARE LLM SCORE
        new_llm_score = llm_utils.calculate_profile_score_llm(
            seed_profile=seed_profile,
            candidate_profile=cand_profile,
            seed_keywords= seed_keywords_dict,
            candidate_keywords= cand_keywords_dict,
            seed_channel_name="Moon",
            candidate_channel_name=cand_name,
            model_provider="gpt"
        )
        
        # NEW STYLE-BIASED EMBEDDING SCORE
        new_emb_score = 0.0
        if llm_utils.embedding_model:
             new_emb_score = llm_utils.calculate_embedding_similarity_hybrid(
                 seed_keywords_flat,
                 cand_keywords_flat
             )
        else:
             print("  ⚠️ Skipping embedding score, model not loaded.")
        
        print(f"  ✅ New Scores: LLM={new_llm_score}, EMB={new_emb_score} (Old: LLM={old_llm_score}, EMB={old_emb_score})")

        # 4. Store Results (--- THIS IS THE MODIFIED BLOCK ---)
        results.append({
            "Channel_Name": cand_name,
            "New_LLM_Score": new_llm_score,
            "New_Emb_Score": new_emb_score,
            "Old_LLM_Score": old_llm_score,
            "Old_Emb_Score": old_emb_score,
            
            # --- NEW: Caching all profile data ---
            "Cand_Ideology": cand_profile.get("ideology", "N/A"),
            "Cand_Format": cand_profile.get("format", "N/A"),
            "Cand_Niche": cand_profile.get("niche", "N/A"),
            "Cand_Intent": cand_profile.get("intent", "N/A"),
            "Cand_Speaker": cand_profile.get("speaker", "N/A"),
            "Cand_Audience": cand_profile.get("target_audience", "N/A"),
            
            # --- NEW: Caching seed profile data ---
            "Seed_Ideology": seed_profile.get("ideology", "N/A"),
            "Seed_Format": seed_profile.get("format", "N/A"),
            "Seed_Niche": seed_profile.get("niche", "N/A"),
            
            # --- NEW: Caching keyword/category strings ---
            "Seed_Categories": seed_categories_str,
            "Seed_Keywords": seed_keywords_str,
            "Cand_Categories": cand_categories_str,
            "Cand_Keywords": cand_keywords_str
        })

    # --- D. Show Final Report ---
    print(f"\n[STEP 4/4] --- CONTEXT-AWARE RE-TEST COMPLETE ---")
    if not results:
        print("No channels were successfully re-scored.")
        return

    df_report = pd.DataFrame(results)
    
    # Sort by the new LLM score
    df_report = df_report.sort_values(by="New_LLM_Score", ascending=False)
    
    print(df_report.to_markdown(index=False, floatfmt=".3f"))
    
    output_filename = "retest_moon_competitors_report.csv"
    df_report.to_csv(output_filename, index=False)
    print(f"\n✅ Report saved to {output_filename}")


if __name__ == "__main__":
    if not llm_utils.embedding_model:
        print("❌ Embedding model did not load.")
        print("Please check your internet connection and SentenceTransformer installation.")
    else:
        run_context_aware_retest()