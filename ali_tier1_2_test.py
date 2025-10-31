import pandas as pd
from pathlib import Path
import json
import sys
import time

# --- Import the function we want to test ---
try:
    # Make sure your utils file has the (v5) function that
    # accepts 5 arguments (seed_name, seed_kws, cand_name, cand_desc, cand_kws)
    from utils.fingerprint_llm_utils import is_direct_competitor_llm_final_check
    LLM_CHECK_ENABLED = True
    print("✅ LLM Final Check function imported.")
except ImportError:
    print("⚠️ WARNING: Could not import 'is_direct_competitor_llm_final_check'.")
    print("   Please add the new function to fingerprint_llm_utils.py first.")
    LLM_CHECK_ENABLED = False
    sys.exit() # Exit if we can't import the function

# === CONFIGURATION ===
base_dir = Path(__file__).resolve().parent

# --- THIS IS THE CORRECT INPUT FILE (as you said) ---
intermediate_data_path = base_dir / "phase3_intermediate_data_moon.csv" 

# --- THIS IS YOUR REQUESTED OUTPUT FILE ---
output_validation_path = base_dir / "final_moon_tier1_2_check.csv"

# --- Tier rules to find the channels to test ---
TIER_1_LLM_MIN = 0.80
TIER_2_LLM_MIN = 0.70 # We are checking everything >= 0.70

DELAY_BETWEEN_LLM_CHECKS = 2 # Seconds

# ============================================
# === TEST SCRIPT ===
# ============================================
def main():
    print("=" * 70)
    print(f"🚀 Running Validation LLM Check for Tier 1 & 2 Channels")
    print("=" * 70)

    # --- 1. Load Data ---
    print(f"Loading data from: {intermediate_data_path.name}")
    try:
        df = pd.read_csv(intermediate_data_path)
        # Clean NAs for ALL columns, especially ones going to LLM
        df['Discovered_Niche'] = df['Discovered_Niche'].fillna("Unknown - Unknown")
        df['Discovered_Channel_Description'] = df['Discovered_Channel_Description'].fillna("")
        df['Discovered_Keywords'] = df['Discovered_Keywords'].fillna("")
        df['Seed_Keywords'] = df['Seed_Keywords'].fillna("")
        df['LLM_Score'] = df['LLM_Score'].fillna(0.0)
        
    except FileNotFoundError:
         print(f"❌ ERROR: File not found: {intermediate_data_path.name}")
         return
    except KeyError as e:
         print(f"❌ ERROR: Missing a required column: {e}. CSV is not correct.")
         return
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return
    
    if df.empty:
        print("No channels found. Exiting.")
        return
    print(f"✅ Loaded {len(df)} total channels.")

    # --- 2. Get Seed Info (General Purpose) ---
    try:
        # Get data from the *first row* of the CSV
        seed_channel_name = df['Seed_Channel_Name'].iloc[0]
        seed_keywords_str = df['Seed_Keywords'].iloc[0] # Get the raw keyword string
        
        print(f"   Seed Channel: {seed_channel_name}")
        if not seed_keywords_str:
            print("   ❌ FATAL ERROR: 'Seed_Keywords' column is empty. Cannot run LLM checks.")
            return
            
    except Exception as e:
        print(f"   ❌ FATAL ERROR: Could not read Seed data from CSV. {e}")
        return
        
    # --- 3. Filter for Tier 1 & 2 Channels ---
    # Find channels that *would have been* auto-accepted
    df_to_check = df[df['LLM_Score'] >= TIER_2_LLM_MIN].copy()
    
    if df_to_check.empty:
        print("   No channels found with LLM_Score >= 0.70 to validate. Exiting.")
        return
        
    print(f"   Found {len(df_to_check)} Tier 1 & 2 channels (LLM Score >= 0.70) to validate.")
    
    # --- 4. Run the Final LLM Check on each ---
    print("-" * 70)
    print(f"🤖 Calling is_direct_competitor_llm_final_check on {len(df_to_check)} channels...")
    
    validation_results = []
    
    for index, row in df_to_check.iterrows():
        # Determine original tier
        original_tier = 1 if row['LLM_Score'] >= TIER_1_LLM_MIN else 2
        
        print(f"   Checking [Tier {original_tier}]: {row['Discovered_Channel_Name']} (Original Score: {row['LLM_Score']})...")
        
        validation_result = is_direct_competitor_llm_final_check(
            seed_name=seed_channel_name,
            seed_keywords_str=seed_keywords_str,
            candidate_name=row['Discovered_Channel_Name'],
            candidate_description=row['Discovered_Channel_Description'],
            candidate_keywords_str=row['Discovered_Keywords'],
            model_provider="gpt"
        )
        
        # Add the channel info to the result
        validation_result['Discovered_Channel_Name'] = row['Discovered_Channel_Name']
        validation_result['Original_Tier'] = original_tier
        validation_result['Original_LLM_Score'] = row['LLM_Score']
        validation_result['Original_Niche'] = row['Discovered_Niche']
        
        validation_results.append(validation_result)
        
        time.sleep(DELAY_BETWEEN_LLM_CHECKS) # Rate limit

    print("✅ Validation checks complete.")

    # --- 5. Save Results to New CSV ---
    try:
        df_validation = pd.DataFrame(validation_results)
        
        # Reorder columns for clarity
        cols_order = [
            'Discovered_Channel_Name', 'Original_Tier', 'Original_LLM_Score', 'Original_Niche',
            'is_competitor', 'confidence', 'reason'
        ]
        existing_cols = [col for col in cols_order if col in df_validation.columns]
        df_validation = df_validation[existing_cols]
        
        df_validation.to_csv(output_validation_path, index=False, encoding='utf-8-sig')
        
        print("-" * 70)
        print(f"🎉 Saved validation results to: {output_validation_path.name}")
        print("-" * 70)
        print("\nFinal Validation Results:")
        print(df_validation.to_string(index=False))
        
    except Exception as e:
        print(f"❌ ERROR saving file: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()