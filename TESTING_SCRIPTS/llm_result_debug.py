import os
import sys
import pandas as pd
import pprint
from typing import Tuple, Dict, List
import re # For cleaning names

# --- 1. Add local utils to the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    import utils.fingerprint_llm_utils as llm_utils
    print("✅ Successfully imported local utils.")
    
    if not llm_utils.gpt_client:
        print("🔴 GPT Client not loaded! Please check your utils file.")
        sys.exit(1)
    
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import local utils.")
    print(f"Error: {e}")
    sys.exit(1)

# --- 2. Config & Helper Function ---
pp = pprint.PrettyPrinter(indent=2)
VIDEO_CACHE_DIR = "video_cache"
SEED_VIDEOS_CSV = 'sample_videos_moon.csv'
OUTPUT_CSV = 'llm_comparison_results.csv'  # Output file

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

def load_real_video_data(channel_name: str, is_seed: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Loads video data from the appropriate cache file.
    Returns (DataFrame, channel_description)
    """
    if is_seed:
        # Load from the main seed file
        try:
            df = pd.read_csv(SEED_VIDEOS_CSV)
            df_channel = df[df['Channel_Name'] == channel_name].copy()
            if df_channel.empty:
                print(f"❌ No data for seed '{channel_name}' in {SEED_VIDEOS_CSV}")
                return pd.DataFrame(), ""
            
            # Get description (it's the same for all rows)
            desc = df_channel['channel_description'].iloc[0]
            return df_channel, str(desc)
        except Exception as e:
            print(f"❌ Failed to load seed file {SEED_VIDEOS_CSV}: {e}")
            return pd.DataFrame(), ""
    else:
        # Load from the 'video_cache' directory
        # Create the safe name, e.g., "The Young Turks" -> "The_Young_Turks_videos.csv"
        safe_name = re.sub(r'[^\w_.-]', '', channel_name.replace(" ", "_"))
        cache_file_path = os.path.join(VIDEO_CACHE_DIR, f"{safe_name}_videos.csv")
        
        try:
            if not os.path.exists(cache_file_path):
                print(f"❌ Cache file not found: {cache_file_path}")
                print(f"Make sure you ran the 'test_final_context_moon.py' script first.")
                return pd.DataFrame(), ""
                
            df = pd.read_csv(cache_file_path)
            if df.empty:
                print(f"❌ Cache file is empty: {cache_file_path}")
                return pd.DataFrame(), ""
            
            # Get description
            desc = df['channel_description'].iloc[0]
            return df, str(desc)
        except Exception as e:
            print(f"❌ Failed to load cache file {cache_file_path}: {e}")
            return pd.DataFrame(), ""

# --- 3. Main Test Function (Now returns results dict) ---
def run_end_to_end_test(seed_name: str, candidate_name: str) -> Dict:
    """
    Runs the full pipeline:
    1. Load data for both
    2. Fingerprint both
    3. Run both scoring functions
    
    Returns dict with all results for CSV storage
    """
    print("\n" + "="*80)
    print(f"🧪 E2E TEST: Seed='{seed_name}' vs. Candidate='{candidate_name}'")
    print("="*80)
    
    # Initialize result dict with default values
    result = {
        'Seed_Channel': seed_name,
        'Candidate_Channel': candidate_name,
        'Status': 'Failed',
        'Seed_Niche': None,
        'Candidate_Niche': None,
        'Seed_Format': None,
        'Candidate_Format': None,
        'Seed_Intent': None,
        'Candidate_Intent': None,
        'Holistic_Score': None,
        'Holistic_Reasoning': None,
        'Audience_Match_Score': None,
        'Audience_Match_Reasoning': None,
    }
    
    # --- 1. Load Data ---
    print(f"\n[STEP 1/4] Loading real data...")
    seed_df, seed_desc = load_real_video_data(seed_name, is_seed=True)
    cand_df, cand_desc = load_real_video_data(candidate_name)
    
    if seed_df.empty or cand_df.empty:
        print("❌ Failed to load data. Aborting test.")
        result['Status'] = 'Data Load Failed'
        return result
    print(f"✅ Loaded {len(seed_df)} seed videos and {len(cand_df)} candidate videos.")

    # --- 2. Get Fingerprints (LIVE LLM CALL) ---
    print(f"\n[STEP 2/4] Generating fingerprints (Live LLM Call)...")
    seed_fp = llm_utils.get_channel_fingerprint_oneshot(seed_name, seed_desc, seed_df)
    cand_fp = llm_utils.get_channel_fingerprint_oneshot(candidate_name, cand_desc, cand_df)
    
    if not seed_fp or not cand_fp:
        print("❌ Failed to generate fingerprints. Aborting test.")
        result['Status'] = 'Fingerprint Failed'
        return result
        
    print(f"✅ Fingerprint for '{seed_name}':")
    pp.pprint(seed_fp['profile'])
    print(f"✅ Fingerprint for '{candidate_name}':")
    pp.pprint(cand_fp['profile'])
    
    # Extract for scoring and CSV
    seed_profile = seed_fp.get('profile', {})
    seed_keywords = seed_fp.get('keywords', {})
    cand_profile = cand_fp.get('profile', {})
    cand_keywords = cand_fp.get('keywords', {})
    
    # Store profile data
    result['Seed_Niche'] = seed_profile.get('niche', 'Unknown')
    result['Candidate_Niche'] = cand_profile.get('niche', 'Unknown')
    result['Seed_Format'] = seed_profile.get('format', 'Unknown')
    result['Candidate_Format'] = cand_profile.get('format', 'Unknown')
    result['Seed_Intent'] = seed_profile.get('intent', 'Unknown')
    result['Candidate_Intent'] = cand_profile.get('intent', 'Unknown')

    # --- 3. Run Holistic Score (Profile-Only) ---
    print(f"\n[STEP 3/4] Running 'Holistic' Score (Profile-Only)...")
    holistic_result = llm_utils.calculate_profile_score_llm_holistic(
        seed_profile=seed_profile,
        candidate_profile=cand_profile,
        seed_channel_name=seed_name,
        candidate_channel_name=candidate_name
    )
    print("--- 'Holistic' (V3) Result ---")
    pp.pprint(holistic_result)
    
    # Store holistic results
    result['Holistic_Score'] = holistic_result.get('score', None)
    result['Holistic_Reasoning'] = holistic_result.get('reasoning', None)

    # --- 4. Run Audience Match Score (Keywords-Aware) ---
    print(f"\n[STEP 4/4] Running 'Audience Match' Score (Keywords-Aware)...")
    audience_result = llm_utils.calculate_profile_score_llm_with_keywords(
        seed_profile=seed_profile,
        candidate_profile=cand_profile,
        seed_keywords=seed_keywords,
        candidate_keywords=cand_keywords,
        seed_channel_name=seed_name,
        candidate_channel_name=candidate_name
    )
    print("--- 'Audience Match' (V4) Result ---")
    pp.pprint(audience_result)
    
    # Store audience match results
    result['Audience_Match_Score'] = audience_result.get('score', None)
    result['Audience_Match_Reasoning'] = audience_result.get('reasoning', None)
    
    result['Status'] = 'Success'
    print(f"\n--- TEST COMPLETE for {seed_name} vs {candidate_name} ---")
    return result

# --- 4. Run the Tests and Save Results ---
if __name__ == "__main__":
    all_results: List[Dict] = []
    
    for channel in CHANNELS_TO_RETEST:
        result = run_end_to_end_test(seed_name="Moon", candidate_name=channel)
        all_results.append(result)
    
    # --- Save to CSV ---
    print("\n" + "="*80)
    print("💾 Saving results to CSV...")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ Results saved to: {OUTPUT_CSV}")
    print(f"Total comparisons: {len(all_results)}")
    print(f"Successful: {len(results_df[results_df['Status'] == 'Success'])}")
    print(f"Failed: {len(results_df[results_df['Status'] != 'Success'])}")
    
    # Print summary
    print("\n📊 SUMMARY:")
    print(results_df[['Candidate_Channel', 'Holistic_Score', 'Audience_Match_Score', 'Status']].to_string(index=False))