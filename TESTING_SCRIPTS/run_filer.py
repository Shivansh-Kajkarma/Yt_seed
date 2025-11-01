import pandas as pd
from pathlib import Path
import time
import json
import re

# === Import Utils ===
try:
    # Make sure your utils file has the (v4) function that
    # accepts 6 arguments (with description and keywords)
    from utils.fingerprint_llm_utils import is_direct_competitor_llm_final_check
    LLM_CHECK_ENABLED = True
    print("✅ LLM Final Check function imported.")
except ImportError:
    print("⚠️ WARNING: Could not import 'is_direct_competitor_llm_final_check'.")
    print("   Please add the new function to fingerprint_llm_utils.py first.")
    LLM_CHECK_ENABLED = False

# === CONFIGURATION ===
base_dir = Path(__file__).resolve().parent
intermediate_data_path = base_dir / "phase3_intermediate_data_moon.csv" # Main input

# --- OUTPUT FILES ---
# 1. The full report with all channels
full_report_path = base_dir / "FINAL_COMPETITORS_Full_Report_moon.csv" 
# 2. The final, clean list of *only* the competitors
final_list_path = base_dir / "FINAL_COMPETITORS_List_moon.csv"

# ============================================
# === 📊 YOUR FILTER RULES (ALL IN ONE PLACE) ===
# ============================================

# --- 1. Final Filter Rules (Applied AFTER tiered logic) ---
# AUTO_KEEP_COUNTRIES = [] # <-- REMOVED as requested.
MIN_SUBS = 10000 # Set this to your desired minimum

# --- 2. Tiered Logic Rules (*** YOUR ORIGINAL LOGIC ***) ---
TIER_1_LLM_MIN = 0.80 # (LLM >= 0.80) -> Auto-Accept (Tier 1)
TIER_2_LLM_MIN = 0.70 # (LLM 0.70 - 0.79) -> Auto-Accept (Tier 2)
TIER_3_LLM_MIN = 0.55 # (LLM 0.55 - 0.69) -> **SEND TO EXTRA LLM CHECK** (Tier 3)
TIER_3_EMBEDDING_MIN = 0.75 # Embedding must be high to justify the extra call
# Tier 4 is < 0.55 (Auto-Reject)

# --- 3. API Settings ---
DELAY_BETWEEN_LLM_CHECKS = 2 # Seconds

# ============================================
# === TIERED LOGIC FUNCTION (*** YOUR ORIGINAL LOGIC ***) ===
# ============================================
def final_competitor_classification(row, seed_name, seed_niche, seed_keywords_str):
    """
    Tiered classification with extra LLM call only for borderline cases.
    """
    llm_score = row['LLM_Score']
    embedding_score = row['Embedding_Score']
    
    # TIER 1: Obvious Competitors (Auto-Accept)
    if llm_score >= TIER_1_LLM_MIN:
        return {
            'Status': 'COMPETITOR', 'Tier': 1, 'Method': 'LLM_SCORE',
            'Reason': f'LLM Score >= {TIER_1_LLM_MIN}', 'Extra_LLM_Call': False
        }
    
    # TIER 2: Strong Competitors (Auto-Accept)
    elif llm_score >= TIER_2_LLM_MIN:
        return {
            'Status': 'COMPETITOR', 'Tier': 2, 'Method': 'LLM_SCORE',
            'Reason': f'LLM Score >= {TIER_2_LLM_MIN}', 'Extra_LLM_Call': False
        }
    
    # TIER 3: Borderline (CALL EXTRA LLM FOR VALIDATION)
    elif llm_score >= TIER_3_LLM_MIN: 
        if embedding_score >= TIER_3_EMBEDDING_MIN:
            print(f"  ⚠️ Borderline case: {row['Discovered_Channel_Name']} (LLM: {llm_score}, Embed: {embedding_score})")
            print(f"     Making extra validation call...")
            
            time.sleep(DELAY_BETWEEN_LLM_CHECKS)
            
            validation = is_direct_competitor_llm_final_check(
                seed_name=seed_name,
                seed_keywords_str=seed_keywords_str, # <-- PASSING SEED KWS
                candidate_name=row['Discovered_Channel_Name'],
                candidate_description=row['Discovered_Channel_Description'],
                candidate_keywords_str=row['Discovered_Keywords'],
                model_provider="gpt"
            )
            
            if validation['is_competitor']: # <-- THIS IS THE "YES" ANSWER
                return {
                    'Status': 'COMPETITOR', # <-- IT IS MARKED "COMPETITOR"
                    'Tier': 3, 'Method': 'EXTRA_LLM_VALIDATION',
                    'Reason': validation['reason'], 'Extra_LLM_Call': True
                }
            else: # This is the "No" answer
                return {
                    'Status': 'FILTERED', 'Tier': 3, 'Method': 'EXTRA_LLM_VALIDATION',
                    'Reason': validation['reason'], 'Extra_LLM_Call': True
                }
        else:
            # Embedding doesn't confirm
            return {
                'Status': 'FILTERED', 'Tier': 3, 'Method': 'EMBEDDING_REJECT',
                'Reason': f'LLM {llm_score} borderline, but Embedding {embedding_score} < {TIER_3_EMBEDDING_MIN}',
                'Extra_LLM_Call': False
            }
    
    # TIER 4: Clear Noise (Auto-Reject)
    else:  # llm_score < TIER_3_LLM_MIN
        return {
            'Status': 'FILTERED', 'Tier': 4, 'Method': 'LLM_SCORE_LOW',
            'Reason': f'LLM Score {llm_score} < {TIER_3_LLM_MIN}',
            'Extra_LLM_Call': False
        }

# ============================================
# === MAIN SCRIPT (This is 100% correct) ===
# ============================================
def main():
    print("=" * 70)
    print("🚀 Starting FINAL MERGED Filter Script (Original Tiers, No Country Filter)")
    print("=" * 70)
    
    if not LLM_CHECK_ENABLED:
        print("❌ ERROR: 'is_direct_competitor_llm_final_check' not found in utils.")
        return

    # --- 1. Load Data ---
    print(f"Loading data from: {intermediate_data_path.name}")
    try:
        df = pd.read_csv(intermediate_data_path)
        # Clean NAs
        df['Discovered_Niche'] = df['Discovered_Niche'].fillna("Unknown - Unknown")
        df['Discovered_Country'] = df['Discovered_Country'].fillna("Unknown")
        df['LLM_Score'] = df['LLM_Score'].fillna(0.0)
        df['Embedding_Score'] = df['Embedding_Score'].fillna(0.0)
        df['Discovered_Subs'] = df['Discovered_Subs'].fillna(0)
        df['Discovered_Channel_Description'] = df['Discovered_Channel_Description'].fillna("")
        df['Discovered_Keywords'] = df['Discovered_Keywords'].fillna("")
        
    except FileNotFoundError:
         print(f"❌ ERROR: File not found: {intermediate_data_path.name}")
         return
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return
    
    if df.empty:
        print("No channels found. Exiting.")
        return
    print(f"✅ Loaded {len(df)} total channels.")

    try:
        # Get data from the *first row* of the CSV
        seed_channel_name = df['Seed_Channel_Name'].iloc[0]
        seed_niche = df['Seed_Niche'].iloc[0] # We still pass this, even if unused
        seed_keywords_str = df['Seed_Keywords'].iloc[0] # <-- CRITICAL
        
        print(f"   Seed Channel: {seed_channel_name} (Niche: {seed_niche})")
        if not seed_keywords_str:
            print("   ❌ FATAL ERROR: 'Seed_Keywords' column is empty. Cannot run LLM checks.")
            return
            
    except Exception as e:
        print(f"   ❌ FATAL ERROR: Could not read Seed data from CSV. {e}")
        return

    # --- 2. Get Seed Info (General Purpose) ---
    try:
        seed_channel_name = df['Seed_Channel_Name'].iloc[0]
        seed_niche = df['Seed_Niche'].iloc[0]
        print(f"   Seed Channel: {seed_channel_name} (Niche: {seed_niche})")
    except Exception as e:
        print(f"   ❌ FATAL ERROR: Could not read Seed_Channel_Name or Seed_Niche from CSV. {e}")
        return
    
    # --- 3. Apply Tiered Classification ---
    print("-" * 70)
    print(f"🤖 Applying Tiered Classification Logic to all {len(df)} channels...")
    
    results = df.apply(
        final_competitor_classification, 
        axis=1, 
        args=(seed_channel_name, seed_niche, seed_keywords_str) # <-- Pass seed keyword string
    )
    
    df_results = pd.json_normalize(results)
    df_full_report = df.join(df_results) 
    print("✅ Classification complete.")

    # --- 4. Save the FULL Debug Report ---
    try:
        report_cols = [
            'Seed_Channel_Name', 'Discovered_Channel_Name', 'Discovered_Niche',
            'Discovered_Subs', 'Discovered_Country', 'LLM_Score', 'Embedding_Score',
            'Status', 'Tier', 'Method', 'Reason', 'Extra_LLM_Call',
            'Discovered_Channel_ID', 'Discovered_Channel_URL'
        ]
        existing_report_cols = [col for col in report_cols if col in df_full_report.columns]
        
        df_full_report[existing_report_cols].to_csv(full_report_path, index=False, encoding="utf-8-sig")
        print(f"\n💾 Saved FULL analysis (all {len(df_full_report)} channels) to: {full_report_path.name}")
    except Exception as e:
        print(f"\n❌ ERROR saving full report: {e}")

    # --- 5. Apply FINAL Filters to get the Competitor List ---
    print("-" * 70)
    print("📊 Applying final filters (Status, Subs) to get competitor list...")
    print(f"   (Country filter is REMOVED as requested)")
    
    final_competitors_df = df_full_report[
        (df_full_report['Status'] == 'COMPETITOR') &
        # (df_full_report['Discovered_Country'].isin(AUTO_KEEP_COUNTRIES)) & # <-- REMOVED
        (df_full_report['Discovered_Subs'] >= MIN_SUBS)
    ]
    
    print(f"   - {len(df_full_report[df_full_report['Status'] == 'COMPETITOR'])} channels passed 'COMPETITOR' status (from Tiers 1, 2, & 3).")
    print(f"   - {len(final_competitors_df)} channels remain after adding Sub filter.")

    # --- 6. Report & Save Final List ---
    print("-" * 70)
    print(f"🎉 FINAL RESULTS: {len(final_competitors_df)} competitors found!")
    print("-" * 70)
    
    if not final_competitors_df.empty:
        print("\n🏆 Final Competitor List (Sorted by Tier, then LLM Score):")
        
        # --- REASON COLUMN IS INCLUDED ---
        display_cols = [
            'Tier', 'Discovered_Channel_Name', 'Discovered_Niche',
            'LLM_Score', 'Embedding_Score', 'Discovered_Subs', 'Discovered_Country', 
            'Method', 'Reason' # <-- REASON IS HERE
        ]
        
        existing_cols = [col for col in display_cols if col in final_competitors_df.columns]
        print(final_competitors_df[existing_cols].sort_values(by=['Tier', 'LLM_Score'], ascending=[True, False]).to_string(index=False))
    
    try:
        final_competitors_df[existing_cols].to_csv(final_list_path, index=False, encoding="utf-8-sig")
        print(f"\n💾 Saved FINAL competitor list to: {final_list_path.name}")
    except Exception as e:
        print(f"\n❌ ERROR saving final list: {e}")

    print("=" * 70)

if __name__ == "__main__":
    main()
