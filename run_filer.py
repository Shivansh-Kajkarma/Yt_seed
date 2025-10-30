import pandas as pd
from pathlib import Path
import time
import json
import re # Not used, but good to keep

# === Import Utils ===
try:
    from utils.fingerprint_llm_utils import is_direct_competitor_llm_final_check
    LLM_CHECK_ENABLED = True
    print("✅ LLM Final Check function imported.")
except ImportError:
    print("⚠️ WARNING: Could not import 'is_direct_competitor_llm_final_check'.")
    print("   Please add the new function to fingerprint_llm_utils.py first.")
    LLM_CHECK_ENABLED = False

# === CONFIGURATION ===
base_dir = Path(__file__).resolve().parent
intermediate_data_path = base_dir / "phase3_intermediate_data.csv" # Main input

# --- OUTPUT FILES ---
# 1. The full report with all channels and why they were filtered/kept
full_report_path = base_dir / "FINAL_COMPETITORS_Full_Report.csv" 
# 2. The final, clean list of *only* the competitors
final_list_path = base_dir / "FINAL_COMPETITORS_List.csv"

# ============================================
# === 📊 YOUR FILTER RULES (ALL IN ONE PLACE) ===
# ============================================

# --- 1. Final Filter Rules (Applied AFTER tiered logic) ---
AUTO_KEEP_COUNTRIES = [
    'US',  # United States
    'GB',  # United Kingdom
    'CA',  # Canada
    'AU',  # Australia
    'NZ',  # New Zealand
    'NG',  # Nigeria
]
MIN_SUBS = 50000 # Set this to your desired minimum

# --- 2. Tiered Logic Rules ---
TIER_1_LLM_MIN = 0.80
TIER_2_LLM_MIN = 0.70
TIER_3_LLM_MIN = 0.55
TIER_3_EMBEDDING_MIN = 0.75

# --- 3. API Settings ---
DELAY_BETWEEN_LLM_CHECKS = 2 # Seconds

# ============================================
# === TIERED LOGIC FUNCTION (Your brilliant idea) ===
# ============================================
def final_competitor_classification(row, seed_name, seed_niche):
    """
    Tiered classification with extra LLM call only for borderline cases.
    Receives seed_name and seed_niche as arguments.
    """
    llm_score = row['LLM_Score']
    embedding_score = row['Embedding_Score']
    
    # TIER 1: Obvious Competitors (no extra call needed)
    if llm_score >= TIER_1_LLM_MIN:
        return {
            'Status': 'COMPETITOR',
            'Tier': 1,
            'Method': 'LLM_SCORE',
            'Reason': f'LLM Score >= {TIER_1_LLM_MIN}',
            'Extra_LLM_Call': False
        }
    
    # TIER 2: Strong Competitors (no extra call needed)
    elif llm_score >= TIER_2_LLM_MIN:
        return {
            'Status': 'COMPETITOR',
            'Tier': 2,
            'Method': 'LLM_SCORE',
            'Reason': f'LLM Score >= {TIER_2_LLM_MIN}',
            'Extra_LLM_Call': False
        }
    
    # TIER 3: Borderline (CALL EXTRA LLM FOR VALIDATION)
    elif llm_score >= TIER_3_LLM_MIN:
        # Only call extra LLM if embedding also suggests similarity
        if embedding_score >= TIER_3_EMBEDDING_MIN:
            print(f"  ⚠️ Borderline case: {row['Discovered_Channel_Name']} (LLM: {llm_score}, Embed: {embedding_score})")
            print(f"     Making extra validation call...")
            
            time.sleep(DELAY_BETWEEN_LLM_CHECKS) # Rate limit
            
            # EXTRA LLM CALL - Uses arguments, not globals
            validation = is_direct_competitor_llm_final_check(
                seed_name=seed_name,
                seed_niche_format=seed_niche,
                candidate_name=row['Discovered_Channel_Name'],
                candidate_niche_format=row['Discovered_Niche'],
                model_provider="gpt"
            )
            
            if validation['is_competitor']:
                return {
                    'Status': 'COMPETITOR',
                    'Tier': 3,
                    'Method': 'EXTRA_LLM_VALIDATION',
                    'Reason': validation['reason'],
                    'Extra_LLM_Call': True
                }
            else:
                return {
                    'Status': 'FILTERED',
                    'Tier': 3,
                    'Method': 'EXTRA_LLM_VALIDATION',
                    'Reason': validation['reason'],
                    'Extra_LLM_Call': True
                }
        else:
            # Embedding doesn't confirm, likely false positive
            return {
                'Status': 'FILTERED',
                'Tier': 3,
                'Method': 'EMBEDDING_REJECT',
                'Reason': f'LLM {llm_score} borderline, but Embedding {embedding_score} < {TIER_3_EMBEDDING_MIN}',
                'Extra_LLM_Call': False
            }
    
    # TIER 4: Clear Noise (no extra call needed)
    else:  # llm_score < TIER_3_LLM_MIN
        return {
            'Status': 'FILTERED',
            'Tier': 4,
            'Method': 'LLM_SCORE_LOW',
            'Reason': f'LLM Score {llm_score} < {TIER_3_LLM_MIN}',
            'Extra_LLM_Call': False
        }

# ============================================
# === MAIN SCRIPT ===
# ============================================
def main():
    print("=" * 70)
    print("🚀 Starting FINAL MERGED Filter Script")
    print("=" * 70)
    
    if not LLM_CHECK_ENABLED:
        print("❌ ERROR: 'is_direct_competitor_llm_final_check' not found in utils.")
        print("   Please add the new function to fingerprint_llm_utils.py first.")
        return

    # --- 1. Load Data ---
    print(f"Loading data from: {intermediate_data_path.name}")
    try:
        df = pd.read_csv(intermediate_data_path)
        # Clean NAs
        df['Discovered_Niche'] = df['Discovered_Niche'].fillna("Unknown - Unknown")
        df['Language'] = df['Language'].fillna("un")
        df['Discovered_Country'] = df['Discovered_Country'].fillna("Unknown")
        df['LLM_Score'] = df['LLM_Score'].fillna(0.0)
        df['Embedding_Score'] = df['Embedding_Score'].fillna(0.0)
        df['Discovered_Subs'] = df['Discovered_Subs'].fillna(0)
    except FileNotFoundError:
         print(f"❌ ERROR: File not found: {intermediate_data_path.name}")
         print("   Did you run 'run_discovery_phase3.py' and 'add_country_data.py'?")
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
        seed_channel_name = df['Seed_Channel_Name'].iloc[0]
        seed_niche = df['Seed_Niche'].iloc[0]
        print(f"   Seed Channel: {seed_channel_name} (Niche: {seed_niche})")
    except Exception as e:
        print(f"   ❌ FATAL ERROR: Could not read Seed_Channel_Name or Seed_Niche from CSV. {e}")
        return
    
    # --- 3. Apply Tiered Classification (The "Debug" part) ---
    print("-" * 70)
    print(f"🤖 Applying Tiered Classification Logic to all {len(df)} channels...")
    
    results = df.apply(
        final_competitor_classification, 
        axis=1, 
        args=(seed_channel_name, seed_niche) # Pass seed info here
    )
    
    df_results = pd.json_normalize(results)
    df_full_report = df.join(df_results) # This is the full debug report
    print("✅ Classification complete.")

    # --- 4. Save the FULL Debug Report (as requested) ---
    try:
        # Define all columns we want in the full report
        report_cols = [
            'Seed_Channel_Name', 'Discovered_Channel_Name', 'Discovered_Niche',
            'Discovered_Subs', 'Discovered_Country', 'Language', 
            'LLM_Score', 'Embedding_Score',
            'Status', 'Tier', 'Method', 'Reason', 'Extra_LLM_Call',
            'Discovered_Channel_ID', 'Discovered_Channel_URL'
        ]
        # Filter list to only columns that actually exist
        existing_report_cols = [col for col in report_cols if col in df_full_report.columns]
        
        df_full_report[existing_report_cols].to_csv(full_report_path, index=False, encoding="utf-8-sig")
        print(f"\n💾 Saved FULL analysis (all {len(df_full_report)} channels) to: {full_report_path.name}")
    except Exception as e:
        print(f"\n❌ ERROR saving full report: {e}")

    # --- 5. Apply FINAL Filters to get the Competitor List ---
    print("-" * 70)
    print("📊 Applying final filters (Status, Country, Subs) to get competitor list...")
    
    final_competitors_df = df_full_report[
        (df_full_report['Status'] == 'COMPETITOR') &
        (df_full_report['Discovered_Country'].isin(AUTO_KEEP_COUNTRIES)) &
        (df_full_report['Discovered_Subs'] >= MIN_SUBS)
    ]
    
    print(f"   - {len(df_full_report[df_full_report['Status'] == 'COMPETITOR'])} channels passed 'COMPETITOR' status.")
    print(f"   - {len(final_competitors_df)} channels remain after adding Country & Sub filters.")

    # --- 6. Report & Save Final List ---
    print("-" * 70)
    print(f"🎉 FINAL RESULTS: {len(final_competitors_df)} competitors found!")
    print("-" * 70)
    
    if not final_competitors_df.empty:
        print("\n🏆 Final Competitor List (Sorted by Tier, then LLM Score):")
        display_cols = [
            'Tier', 'Discovered_Channel_Name', 'Discovered_Niche',
            'LLM_Score', 'Embedding_Score', 'Discovered_Subs', 'Discovered_Country', 'Method', 'Reason'
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