import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

# --- Make sure utils are importable ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    # --- 1. Import local utils ---
    # We DO NOT need youtube_utils here anymore!
    import utils.fingerprint_llm_utils as llm_utils
    print("✅ Successfully imported local LLM utils.")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import 'utils' folder.")
    print(f"Make sure {__file__} is in the 'SCRIPTS' folder.")
    sys.exit(1)

# --- 2. Helper Functions ---

def flatten_keywords(kw_dict: dict) -> List[str]:
    """ Flattens the keyword dictionary into a list for the embedding model. """
    all_kws = []
    if not isinstance(kw_dict, dict):
        return []
    for kws in kw_dict.values():
        if isinstance(kws, list):
            all_kws.extend(kws)
    return list(set(all_kws))

def find_latest_file_path(directory: Path, prefix: str, suffix: str) -> Optional[Path]:
    """Finds the most recent file in a directory matching a pattern."""
    try:
        latest_file = max(
            directory.glob(f"{prefix}*{suffix}"),
            key=os.path.getctime
        )
        return latest_file
    except ValueError:
        return None # No files found

# --- 3. Main Function ---
def main(run_tag: str):
    """
    Runs the full Phase 3 LLM-SCORING pipeline for a given run_tag.
    """
    print(f"--- 🚀 Starting Phase 3: LLM Scoring for '{run_tag}' ---")
    
    # --- Define file paths based on our new structure ---
    FINGERPRINT_DIR = BASE_DIR / "FINGERPRINTS_ONESHOT"
    DISCOVERY_CACHE_DIR = BASE_DIR / "PHASE_2_DISCOVERY_CACHE"
    REPORT_DIR = BASE_DIR / "PHASE_3_REPORTS"
    
    os.makedirs(REPORT_DIR, exist_ok=True) # Create final report dir

    # --- A. Load Seed Channel Fingerprint (from Phase 1) ---
    print(f"\n[STEP 1/4] Loading Seed Fingerprint...")
    SEED_FINGERPRINT_FILE = find_latest_file_path(
        FINGERPRINT_DIR, f"fingerprints_oneshot_{run_tag}", ".json"
    )
    
    if not SEED_FINGERPRINT_FILE:
        print(f"❌ ERROR: No Phase 1 fingerprint file found for tag '{run_tag}'.")
        print(f"Looked in: {FINGERPRINT_DIR}")
        print("Please run Phase 1 first.")
        return

    print(f"  Using Seed File: {SEED_FINGERPRINT_FILE.name}")
    try:
        with open(SEED_FINGERPRINT_FILE, 'r') as f:
            seed_data = json.load(f)
            
        # --- This is complex, but it finds the *first* channel in the JSON ---
        if "channels" not in seed_data or not seed_data["channels"]:
             print("❌ ERROR: Seed fingerprint file has no 'channels' data.")
             return
             
        first_channel_key = next(iter(seed_data["channels"]))
        seed_fingerprint = seed_data["channels"][first_channel_key].get("fingerprint", {})
        seed_name = seed_data["channels"][first_channel_key].get("channel_name", run_tag)
        
        if not seed_fingerprint:
            print("❌ ERROR: No 'fingerprint' data found for the first channel in the JSON.")
            return

    except Exception as e:
        print(f"❌ ERROR: Failed to load or parse seed fingerprint: {e}")
        return

    seed_profile = seed_fingerprint.get("profile", {})
    seed_keywords_dict = seed_fingerprint.get("keywords", {})
    seed_keywords_flat = flatten_keywords(seed_keywords_dict)
    seed_categories_str = ", ".join(seed_keywords_dict.keys())
    seed_keywords_str = ", ".join(seed_keywords_flat)
    
    print(f"✅ Seed '{seed_name}' fingerprint loaded successfully.")
    print(f"  > Profile: {seed_profile.get('niche')} | {seed_profile.get('format')} | {seed_profile.get('ideology')}")

    # --- B. Load Discovered Candidate Data (from Phase 2) ---
    print(f"\n[STEP 2/4] Loading Discovered Candidates...")
    DISCOVERY_CACHE_FILE = DISCOVERY_CACHE_DIR / f"phase2_discovered_raw_data_{run_tag}.csv"

    if not DISCOVERY_CACHE_FILE.exists():
        print(f"❌ ERROR: Phase 2 cache file not found: {DISCOVERY_CACHE_FILE.name}")
        print("Please run Phase 2 first.")
        return

    print(f"  Using Cache File: {DISCOVERY_CACHE_FILE.name}")
    try:
        df_candidates = pd.read_csv(DISCOVERY_CACHE_FILE)
        # Fill NaNs in critical text columns to prevent errors
        df_candidates['Discovered_Channel_Description'] = df_candidates['Discovered_Channel_Description'].fillna("")
        df_candidates['Discovered_Videos_JSON'] = df_candidates['Discovered_Videos_JSON'].fillna("[]")

        if df_candidates.empty:
            print(f"⚠️ Warning: Cache file {DISCOVERY_CACHE_FILE.name} is empty. Nothing to process.")
            return
    except Exception as e:
        print(f"❌ ERROR: Failed to load cache data: {e}")
        return

    print(f"✅ Found {len(df_candidates)} candidates to analyze.")

    # --- C. Loop, Fingerprint (from cache), and Score (LLM calls) ---
    print(f"\n[STEP 3/4] Starting LLM analysis loop (this will take time)...")
    
    results = []
    
    for index, row in df_candidates.iterrows():
        cand_name = row.get('Discovered_Channel_Name')
        cand_id = row.get('Discovered_Channel_ID')
        
        if not cand_name or not cand_id:
            print(f"  ⚠️ Skipping row {index} due to missing Name or ID.")
            continue
            
        print(f"\n--- Processing {cand_name} ({index+1}/{len(df_candidates)}) ---")
        
        try:
            # 1. Load Candidate Data (from CSV Cache)
            #    THIS IS THE FIX! NO API CALLS!
            print(f"  Loading cached data for {cand_name}...")
            cand_desc = row.get('Discovered_Channel_Description')
            cand_videos_json = row.get('Discovered_Videos_JSON')
            
            try:
                cand_videos_list = json.loads(cand_videos_json)
            except json.JSONDecodeError:
                print(f"  ⚠️ Invalid video JSON for {cand_name}. Skipping.")
                continue
                
            if not cand_videos_list:
                print(f"  ⚠️ No videos found in cache for {cand_name}. Skipping.")
                continue
                
            df_cand_videos = pd.DataFrame(cand_videos_list)
            
            # 2. Generate Candidate Fingerprint (LLM Call)
            print(f"  Fingerprinting {cand_name} (LLM Call 1)...")
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
            cand_categories_str = ", ".join(cand_keywords_dict.keys())
            cand_keywords_str = ", ".join(cand_keywords_flat)
            print(f"  > Profile: {cand_profile.get('niche')} | {cand_profile.get('format')} | {cand_profile.get('ideology')}")

            # 3. Calculate Final Scores
            print(f"  Scoring {cand_name} vs '{seed_name}' (LLM Call 2)...")
            
            # --- Holistic Profile Score (LLM Call) ---
            score_data = llm_utils.calculate_profile_score_llm_holistic(
                seed_profile=seed_profile,
                candidate_profile=cand_profile,
                seed_channel_name=seed_name,
                candidate_channel_name=cand_name,
                model_provider="gpt"
            )
            
            sim_score = score_data.get("similarity_score", 0.0)
            comp_score = score_data.get("competitor_score", 0.0)
            aud_score = score_data.get("audience_overlap_score", 0.0)
            llm_reason = score_data.get("reason", "N/A")
            
            # --- Embedding Score (Local) ---
            emb_score = llm_utils.calculate_embedding_similarity_hybrid(
                 seed_keywords_flat,
                 cand_keywords_flat
            )
            
            print(f"  ✅ Scores: Comp={comp_score:.2f}, Aud={aud_score:.2f}, Sim={sim_score:.2f}, Emb={emb_score:.2f}")

            # 4. Store Results
            results.append({
                "Seed_Channel_Name": seed_name,
                "Discovered_Channel_Name": cand_name,
                "Discovered_Channel_ID": cand_id,
                
                "Competitor_Score": comp_score,
                "Audience_Overlap_Score": aud_score,
                "Similarity_Score": sim_score,
                "Embedding_Keyword_Score": emb_score,
                "LLM_Reason": llm_reason,
                
                "Cand_Niche": cand_profile.get("niche", "N/A"),
                "Cand_Format": cand_profile.get("format", "N/A"),
                "Cand_Intent": cand_profile.get("intent", "N/A"),
                "Cand_Speaker": cand_profile.get("speaker", "N/A"),
                "Cand_Ideology": cand_profile.get("ideology", "N/A"),
                "Cand_Audience": cand_profile.get("target_audience", "N/A"),
                
                "Seed_Niche": seed_profile.get("niche", "N/A"),
                "Seed_Format": seed_profile.get("format", "N/A"),
                "Seed_Ideology": seed_profile.get("ideology", "N/A"),
                
                "Discovered_Subs": row.get("Discovered_Subs", 0),
                "Discovered_Country": row.get("Discovered_Country", "Unknown"),
                
                "Seed_Categories": seed_categories_str,
                "Cand_Categories": cand_categories_str,
                "Seed_Keywords_Count": len(seed_keywords_flat),
                "Cand_Keywords_Count": len(cand_keywords_flat)
            })
            
            time.sleep(1) # 1 second delay between candidates
            
        except Exception as e:
            print(f"  ❌❌ UNEXPECTED ERROR during processing for {cand_name}: {e}")
            print(f"  Skipping this channel and continuing...")
            continue

    # --- D. Save Final Report ---
    print(f"\n[STEP 4/4] --- ANALYSIS COMPLETE ---")
    if not results:
        print("No channels were successfully scored.")
        return

    df_report = pd.DataFrame(results)
    
    # Sort by the most important score
    df_report = df_report.sort_values(by="Competitor_Score", ascending=False)
    
    # Save to a new timestamped CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    FINAL_REPORT_FILE = REPORT_DIR / f"FINAL_Report_{run_tag}_{timestamp}.csv"
    
    df_report.to_csv(FINAL_REPORT_FILE, index=False, encoding='utf-8')
    print(f"\n✅✅✅ Final Report saved to {FINAL_REPORT_FILE.name} ✅✅✅")
    
    # Print a snippet of the report
    try:
        print("\n--- Top 10 Competitors Found ---")
        cols_to_show = [
            'Discovered_Channel_Name', 
            'Competitor_Score', 
            'Audience_Overlap_Score', 
            'Similarity_Score', 
            'Embedding_Keyword_Score',
            'LLM_Reason'
        ]
        print(df_report[cols_to_show].head(10).to_markdown(index=False, floatfmt=".3f"))
    except ImportError:
        print("Install 'tabulate' (pip install tabulate) to see a formatted table here.")
        print(df_report[['Discovered_Channel_Name', 'Competitor_Score', 'Audience_Overlap_Score']].head(10))
    except Exception:
        # Fallback if markdown fails for any reason
        print(df_report[['Discovered_Channel_Name', 'Competitor_Score', 'Audience_Overlap_Score']].head(10))

if __name__ == "__main__":
    try:
        import tabulate
    except ImportError:
        print("⚠️  Warning: 'tabulate' module not found. `pip install tabulate` for cleaner log output.")
        
    if not llm_utils.embedding_model:
        print("❌ CRITICAL: Embedding model did not load.")
        print("Please check your internet connection and SentenceTransformer installation.")
        sys.exit(1)
        
    # if len(sys.argv) < 2:
    #     print("❌ ERROR: You must provide a run_tag as an argument.")
    #     print("This must match the tag from your Phase 1 & 2 runs.")
    #     print("Example: python SCRIPTS/phase3_llm_scoring.py moon")
    #     print("Example: python SCRIPTS/phase3_llm_scoring.py vox")
    #     sys.exit(1)
        
    # # Use the CLI argument as the run_tag
    # run_tag_arg = sys.argv[1].lower().strip()
    # print(run_tag_arg)
    run_tag_arg = "moon"
    main(run_tag_arg)