import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
import time
from datetime import datetime
import re

# --- 1. SETUP: Load .env and define base directory ---
# This script should be in your SCRIPTS/ folder
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    # --- Import the GPT client AND the new tier function from your utils ---
    from utils.fingerprint_llm_utils import gpt_client, get_channel_tier_gpt
    print("✅ Successfully imported GPT client and get_channel_tier_gpt function.")
except ImportError:
    print("❌ CRITICAL: Could not import 'gpt_client' or 'get_channel_tier_gpt'.")
    print("   Make sure they are defined in 'utils/fingerprint_llm_utils.py'.")
    sys.exit(1)

if not gpt_client:
    print("❌ CRITICAL: 'gpt_client' is None. Check your OPENAI_API_KEY in .env")
    sys.exit(1)


# ==================================================
# 3. CONFIGURATION (Paths)
# ==================================================
# --- These are the exact paths you provided ---
# Make sure these filenames are 100% correct
# SEED_FINGERPRINT_FILE = BASE_DIR / "FINGERPRINTS_ONESHOT" / "fingerprints_oneshot_vox_20251103_102550.json"
# CANDIDATE_LIST_FILE = BASE_DIR / "PHASE_2_5_TRIAGE_REPORTS" / "phase2_5_triage_report_vox_filtered_20251103_115538_0.45.csv"

SEED_FINGERPRINT_FILE = BASE_DIR / "FINGERPRINTS_ONESHOT" / "fingerprints_oneshot_moon_20251101_211341.json"
CANDIDATE_LIST_FILE = BASE_DIR / "PHASE_2_5_TRIAGE_REPORTS" / "phase2_5_triage_report_moon_filtered_20251103_170315.csv"
# --- Define the new output directory for these reports ---
FINAL_REPORT_DIR = BASE_DIR / "PHASE_3_REPORTS"
FINAL_REPORT_DIR.mkdir(exist_ok=True)
# ==================================================


# --- 5. MAIN SCRIPT ---
def main():
    """
    Main function to run the "Client's Gut Check" analysis.
    Loads the filtered triage report, runs GPT tier analysis,
    and saves a full JSON, a full CSV, and a high-level client CSV.
    """
    print(f"--- 🚀 Starting Client's Tier Analysis (GPT Version) ---")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # --- Define final output paths ---
    run_tag = SEED_FINGERPRINT_FILE.stem.split('_')[2] # Gets 'moon' from the filename
    
    JSON_OUTPUT_FILE = FINAL_REPORT_DIR / f"client_tier_analysis_{run_tag}_{timestamp}.json"
    INTERNAL_CSV_FILE = FINAL_REPORT_DIR / f"phase3_INTERNAL_report_{run_tag}_{timestamp}.csv"
    CLIENT_CSV_FILE = FINAL_REPORT_DIR / f"phase3_CLIENT_report_{run_tag}_{timestamp}.csv"


    # --- A. Load Seed Profile & Keywords ---
    print(f"\n[STEP 1/3] Loading Seed Profile from {SEED_FINGERPRINT_FILE.name}...")
    try:
        with open(SEED_FINGERPRINT_FILE, 'r') as f:
            seed_data = json.load(f)
            
        first_channel_key = next(iter(seed_data["channels"]))
        fingerprint = seed_data["channels"][first_channel_key].get("fingerprint", {})
        
        seed_profile_obj = fingerprint.get("profile")
        seed_keywords_obj = fingerprint.get("keywords")
        seed_channel_name = seed_data["channels"][first_channel_key].get("channel_name", "UnknownSeed")
        
        if not seed_profile_obj or not seed_keywords_obj:
            print("❌ ERROR: 'profile' or 'keywords' key missing from fingerprint JSON.")
            return

        seed_profile_str = json.dumps(seed_profile_obj, indent=2)
        
        #
        # --- THIS IS THE FIX (LOADING KEYWORDS) ---
        #
        # We now flatten all keywords into a string for the new prompt
        all_keywords = []
        for category, keywords in seed_keywords_obj.items():
            all_keywords.append(f"\nCategory: {category}")
            all_keywords.extend([f"- {kw}" for kw in keywords])
        seed_keywords_str = "\n".join(all_keywords)
        # --- END FIX ---
        
        print(f"  ✅ Loaded profile for '{seed_channel_name}'")
        print(f"  ✅ Loaded {len(seed_keywords_obj)} seed categories.")

    except Exception as e:
        print(f"❌ ERROR: Failed to load or parse seed fingerprint: {e}")
        return

    # --- B. Load the Filtered Candidate List ---
    print(f"\n[STEP 2/3] Loading Filtered Candidate List from {CANDIDATE_LIST_FILE.name}...")
    try:
        df_candidates = pd.read_csv(CANDIDATE_LIST_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: Candidate CSV file not found.")
        print(f"   Check path: {CANDIDATE_LIST_FILE}")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to load candidate CSV: {e}")
        return
    
    print(f"   Found {len(df_candidates)} candidates to process.")

    # --- C. Run the LLM Tier Analysis Loop ---
    print(f"\n[STEP 3/3] Starting LLM Tier Analysis Loop...")
    
    json_results = {}
    
    # We will add the new tier data to this list
    new_data_list = []
    
    for index, row in df_candidates.iterrows():
        channel_name = row['Discovered_Channel_Name']
        print(f"\n--- Processing: {channel_name} ({index+1}/{len(df_candidates)}) ---")
        
        try:
            videos_json = row['Discovered_Videos_JSON']
            videos_list = json.loads(videos_json)
            if not videos_list:
                print("  ⚠️  No videos in cache. Skipping.")
                tier_data = {"tier": -1, "reason": "ERROR: No videos in cache."}
            else:
                video_titles = [vid.get('title', '') for vid in videos_list if vid.get('title')]
                
                #
                # --- THIS IS THE FIX (PASSING KEYWORDS) ---
                #
                # We now pass the new 'seed_keywords_str' variable
                #
                tier_data = get_channel_tier_gpt(
                    seed_name=seed_channel_name,
                    candidate_name=channel_name,
                    candidate_titles=video_titles,
                    seed_profile_str=seed_profile_str,
                    seed_categories_str = "",
                    seed_keywords_str=seed_keywords_str # <-- THE FIX
                )
            
            print(f"  ✅ Tier: {tier_data.get('tier')}")
            print(f"  > Reason: {tier_data.get('reason')}")
            
            # Save for JSON
            json_results[channel_name] = tier_data
            
            # Save for the new CSVs
            new_data_list.append({
                "Discovered_Channel_ID": row['Discovered_Channel_ID'], # Use ID as a key
                "Tier": tier_data.get('tier'),
                "Reason": tier_data.get('reason')
            })
            
            time.sleep(1) # Be nice to the API

        except Exception as e:
            print(f"  ❌❌ UNEXPECTED ERROR on {channel_name}: {e}")
            json_results[channel_name] = {"tier": -1, "reason": f"ERROR: {str(e)[:100]}"}
            new_data_list.append({
                "Discovered_Channel_ID": row['Discovered_Channel_ID'],
                "Tier": -1,
                "Reason": f"ERROR: {str(e)[:100]}"
            })
            
    # --- D. Save All Final Outputs ---
    print("\n--- ANALYSIS COMPLETE ---")
    
    # 1. Save the JSON file (your original request)
    try:
        with open(JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ (1/3) Full JSON Report saved to: {JSON_OUTPUT_FILE.name}")
    except Exception as e:
        print(f"❌ ERROR saving JSON: {e}")

    # 2. Create and save the TWO CSV files
    try:
        # Create a DataFrame from our new tier data
        df_new_data = pd.DataFrame(new_data_list)
        
        # Merge this new data back into the original candidate DataFrame
        df_internal_report = pd.merge(
            df_candidates, 
            df_new_data, 
            on="Discovered_Channel_ID", 
            how="left"
        )
        
        # Sort by the new Tier column
        df_internal_report = df_internal_report.sort_values(by="Tier", ascending=True)
        
        # Save the FULL Internal CSV
        df_internal_report.to_csv(INTERNAL_CSV_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ (2/3) Full INTERNAL Report CSV saved to: {INTERNAL_CSV_FILE.name}")

        # --- Create the Client-Facing CSV ---
        
        # Add 'Tier' and 'Reason' to the list of columns you wanted
        client_columns = [
            "Tier",
            "Seed_Channel_Name",
            "Discovered_Channel_Name",
            "Discovered_Subs",
            "Score_Emb_Combined_AvgVec",
            "Score_Emb_Hybrid_Titles",
            "Reason", # The client will want to see the reason!
            "Discovered_Channel_URL"
        ]
        
        # Select only these columns for the client
        df_client_report = df_internal_report[client_columns]
        
        # Save the Client-Facing CSV
        df_client_report.to_csv(CLIENT_CSV_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ (3/3) High-Level CLIENT Report CSV saved to: {CLIENT_CSV_FILE.name}")

    except Exception as e:
        print(f"❌ ERROR saving CSV reports: {e}")
        
    print("\n--- All Done! ---")

# --- 6. RUN THE SCRIPT ---
if __name__ == "__main__":
    main()