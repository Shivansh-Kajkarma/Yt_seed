# import pandas as pd
# from pathlib import Path
# import re # For more robust niche parsing

# # === CONFIGURATION ===
# base_dir = Path(__file__).resolve().parent
# intermediate_data_path = base_dir / "phase3_intermediate_data.csv"
# # Changed filename to reflect added filter
# final_competitors_path = base_dir / "FINAL_COMPETITORS_LangCountrySubsFiltered.csv" 

# # ============================================
# # === 📊 YOUR FILTER RULES - CUSTOMIZE THESE ===
# # ============================================

# # --- Score Thresholds ---
# # MIN_LLM_SCORE = 0.0       # Set to 0 to effectively disable
# # MIN_EMBEDDING_SCORE = 0.0 # Set to 0 to effectively disable

# # --- Basic Channel Filters ---
# MIN_SUBS = 50000          # Minimum subscribers for a competitor (NOW ENABLED)

# # --- Language & Country Filters ---
# REQUIRED_LANGUAGE = "en"  # Only keep channels primarily in English
# COUNTRY_EXCLUSIONS = ["IN"] # List of country codes to EXCLUDE (e.g., ["IN", "PK"])
# # Set COUNTRY_EXCLUSIONS = [] if you don't want to filter by country

# # --- Niche & Format Filters (Case-Insensitive) ---
# # FORMAT_EXCLUSIONS = []    # Empty list disables this filter
# # FORMAT_REQUIREMENTS = []  # Empty list disables this filter

# # ============================================
# # === FILTERING LOGIC (Usually no need to change below) ===
# # ============================================

# def contains_keyword(text_str, keyword_list):
#     """Checks if a string contains any keyword from a list (case-insensitive)."""
#     if not isinstance(text_str, str) or not keyword_list:
#         return False
#     text_low = text_str.lower()
#     for keyword in keyword_list:
#         if keyword.lower() in text_low:
#             return True
#     return False

# def main():
#     print("=" * 70)
#     print("🚀 Starting Phase 3: Filtering Competitors (Language, Country, Subs)") # Updated title
#     print("=" * 70)
#     print(f"Loading data from: {intermediate_data_path.name}")
#     try:
#         df = pd.read_csv(intermediate_data_path)
#         # Fill NA values to avoid errors during filtering
#         df['Discovered_Niche'] = df['Discovered_Niche'].fillna("Unknown - Unknown")
#         df['Language'] = df['Language'].fillna("un")
#         df['Discovered_Country'] = df['Discovered_Country'].fillna("Unknown")
#         df['LLM_Score'] = df['LLM_Score'].fillna(0.0)
#         df['Embedding_Score'] = df['Embedding_Score'].fillna(0.0)
#         df['Discovered_Subs'] = df['Discovered_Subs'].fillna(0) # FillNA important for subs filter

#     except FileNotFoundError:
#         print(f"❌ ERROR: File not found: {intermediate_data_path.name}")
#         print("   Did you run 'run_discovery_phase3.py' successfully first?")
#         return
#     except KeyError as e:
#          print(f"❌ ERROR: Missing expected column in CSV: {e}")
#          print("   Did you run 'add_country_data.py' successfully?")
#          return
#     except Exception as e:
#         print(f"❌ ERROR loading file: {e}")
#         return

#     print(f"✅ Loaded {len(df)} total scored candidates.")
#     print("-" * 70)
#     print("📊 Applying Filters...")

#     initial_count = len(df)
#     df_filtered = df.copy() # Start with all data

#     # --- Filter 1: Language ---
#     df_filtered = df_filtered[df_filtered['Language'] == REQUIRED_LANGUAGE]
#     print(f"  - Language Filter ('{REQUIRED_LANGUAGE}'): {len(df_filtered)} / {initial_count} remain")

#     # --- Filter 2: Country ---
#     if COUNTRY_EXCLUSIONS:
#         if 'Discovered_Country' in df_filtered.columns:
#             country_mask = df_filtered['Discovered_Country'].isin(COUNTRY_EXCLUSIONS)
#             df_filtered = df_filtered[~country_mask] # ~ means NOT in the list
#             print(f"  - Country Filter (Exclude {', '.join(COUNTRY_EXCLUSIONS)}): {len(df_filtered)} remain")
#         else:
#              print("  - Country Filter: Skipped (Column 'Discovered_Country' not found in CSV)")
#     else:
#         print("  - Country Filter: Skipped (no exclusions defined)")

#     # --- Filter 3: Subscribers (NOW ENABLED) ---
#     if MIN_SUBS > 0 :
#         df_filtered = df_filtered[df_filtered['Discovered_Subs'] >= MIN_SUBS]
#         print(f"  - Subscriber Filter (>={MIN_SUBS:,}): {len(df_filtered)} remain")
#     else:
#         print(f"  - Subscriber Filter: SKIPPED (MIN_SUBS is 0)")


#     # --- Filters 4-6 are SKIPPED ---
#     print(f"  - Score Filters: SKIPPED")
#     print(f"  - Niche Format Exclusion Filter: SKIPPED")
#     print(f"  - Niche Format Requirement Filter: SKIPPED")


#     # --- Final Results ---
#     print("-" * 70)
#     print(f"✅ Found {len(df_filtered)} channels matching Language, Country & Subscriber criteria!")

#     # Sort by original LLM score just for viewing consistency
#     df_final = df_filtered.sort_values(by="LLM_Score", ascending=False)

#     # Select and reorder columns
#     try:
#         final_columns = [col for col in 'INTERMEDIATE_COLUMN_ORDER' if col in df_final.columns]
#         df_final = df_final[final_columns]
#     except Exception as e:
#         print(f"  ⚠️ Warning: Could not reorder columns - {e}")

#     # Display top results
#     if not df_final.empty:
#         print("\nTop 5 Channels (Sorted by original LLM Score):")
#         # Added Subs to the printout
#         print(df_final[['Seed_Channel_Name', 'Discovered_Channel_Name', 'Discovered_Niche', 'LLM_Score', 'Discovered_Subs', 'Discovered_Country']].head().to_string(index=False))

#     # Save the filtered list
#     try:
#         df_final.to_csv(final_competitors_path, index=False, encoding="utf-8-sig")
#         print(f"\n💾 Saved list (Lang/Country/Subs filtered) to: {final_competitors_path.name}")
#     except Exception as e:
#         print(f"\n❌ ERROR saving final file: {e}")

#     print("=" * 70)
#     print("🎉 Filtering Complete!")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()


import pandas as pd
from pathlib import Path
import re
import time # <-- Add time for rate limiting

# === CONFIGURATION ===
base_dir = Path(__file__).resolve().parent
intermediate_data_path = base_dir / "phase3_intermediate_data.csv"
final_competitors_path = base_dir / "FINAL_COMPETITORS_LLM_Checked.csv" # New filename

# === Import the new LLM function ===
# Make sure your fingerprint_llm_utils is importable
try:
    from utils.fingerprint_llm_utils import is_direct_competitor_llm
    LLM_CHECK_ENABLED = True
    print("✅ LLM Competitor Check function imported.")
except ImportError:
    print("⚠️ WARNING: Could not import 'is_direct_competitor_llm'. LLM Check will be disabled.")
    LLM_CHECK_ENABLED = False


# ============================================
# === 📊 YOUR FILTER RULES - CUSTOMIZE THESE ===
# ============================================

# --- Score Thresholds ---
MIN_LLM_SCORE = 0.6       # Lower this slightly maybe, as the final check is strict
MIN_EMBEDDING_SCORE = 0.55 # Lower this slightly maybe

# --- Basic Channel Filters ---
MIN_SUBS = 50000

# --- Language & Country Filters ---
REQUIRED_LANGUAGE = "en"
COUNTRY_EXCLUSIONS = ["IN", "Unknown"]

# --- Niche & Format Filters (Applied BEFORE final LLM check) ---
# Keep these looser now, let the final LLM decide the tricky cases
FORMAT_EXCLUSIONS = ["podcast", "interview", "shorts channel", "vlog"] # Maybe remove "motivation"?
FORMAT_REQUIREMENTS = [] # Keep this disabled, let LLM handle nuance

# --- Final LLM Check Settings ---
DELAY_BETWEEN_LLM_CHECKS = 2 # Seconds to wait between API calls

# ============================================
# === FILTERING LOGIC ===
# ============================================

def contains_keyword(text_str, keyword_list):
    # ... (same function as before) ...
    if not isinstance(text_str, str) or not keyword_list: return False
    text_low = text_str.lower()
    for keyword in keyword_list:
        if keyword.lower() in text_low: return True
    return False


def main():
    print("=" * 70)
    print("🚀 Starting Phase 3: Filtering Competitors (with Final LLM Check)")
    print("=" * 70)
    # ... (Loading data and initial prints - same as before) ...
    print(f"Loading data from: {intermediate_data_path.name}")
    try:
        df = pd.read_csv(intermediate_data_path)
        df['Discovered_Niche'] = df['Discovered_Niche'].fillna("Unknown - Unknown")
        df['Language'] = df['Language'].fillna("un")
        df['Discovered_Country'] = df['Discovered_Country'].fillna("Unknown")
        df['LLM_Score'] = df['LLM_Score'].fillna(0.0)
        df['Embedding_Score'] = df['Embedding_Score'].fillna(0.0)
        df['Discovered_Subs'] = df['Discovered_Subs'].fillna(0)
    except Exception as e: # Catch file not found, key error, etc.
        print(f"❌ ERROR loading file: {e}")
        return

    print(f"✅ Loaded {len(df)} total scored candidates.")
    print("-" * 70)
    print("📊 Applying Initial Filters...")

    initial_count = len(df)
    df_filtered = df.copy()

    # --- Initial Filters (Fast, No API) ---
    df_filtered = df_filtered[df_filtered['Language'] == REQUIRED_LANGUAGE]
    print(f"  - Language Filter ('{REQUIRED_LANGUAGE}'): {len(df_filtered)} / {initial_count} remain")

    if COUNTRY_EXCLUSIONS:
        if 'Discovered_Country' in df_filtered.columns:
            country_mask = df_filtered['Discovered_Country'].isin(COUNTRY_EXCLUSIONS)
            df_filtered = df_filtered[~country_mask]
            print(f"  - Country Filter (Exclude {', '.join(COUNTRY_EXCLUSIONS)}): {len(df_filtered)} remain")
        else: print("  - Country Filter: Skipped (Column missing)")
    else: print("  - Country Filter: Skipped (no exclusions)")

    if MIN_SUBS > 0 :
        df_filtered = df_filtered[df_filtered['Discovered_Subs'] >= MIN_SUBS]
        print(f"  - Subscriber Filter (>={MIN_SUBS:,}): {len(df_filtered)} remain")
    else: print(f"  - Subscriber Filter: SKIPPED")

    df_filtered = df_filtered[
        (df_filtered['LLM_Score'] >= MIN_LLM_SCORE) &
        (df_filtered['Embedding_Score'] >= MIN_EMBEDDING_SCORE)
    ]
    print(f"  - Score Filters (LLM>={MIN_LLM_SCORE}, Embed>={MIN_EMBEDDING_SCORE}): {len(df_filtered)} remain")

    if FORMAT_EXCLUSIONS:
        exclude_mask = df_filtered['Discovered_Niche'].apply(lambda niche: contains_keyword(niche, FORMAT_EXCLUSIONS))
        df_filtered = df_filtered[~exclude_mask]
        print(f"  - Niche Format Exclusion Filter (Remove if contains {', '.join(FORMAT_EXCLUSIONS)}): {len(df_filtered)} remain")
    else: print("  - Niche Format Exclusion Filter: Skipped")

    if FORMAT_REQUIREMENTS:
        require_mask = df_filtered['Discovered_Niche'].apply(lambda niche: contains_keyword(niche, FORMAT_REQUIREMENTS))
        df_filtered = df_filtered[require_mask]
        print(f"  - Niche Format Requirement Filter (Keep if contains {', '.join(FORMAT_REQUIREMENTS)}): {len(df_filtered)} remain")
    else: print("  - Niche Format Requirement Filter: Skipped")

    # --- Final LLM Competitor Check (Slower, uses API) ---
    print("-" * 70)
    print(f"🤖 Applying Final LLM Competitor Check on {len(df_filtered)} candidates...")
    
    llm_check_results = []
    if not LLM_CHECK_ENABLED:
        print("   LLM Check is DISABLED (could not import function). Skipping.")
        # If disabled, assume all remaining are competitors for now
        llm_check_results = ["Yes"] * len(df_filtered)
    elif df_filtered.empty:
        print("   No candidates remaining to check.")
    else:
        # Loop through the filtered DataFrame rows
        for index, row in df_filtered.iterrows():
            print(f"   Checking [{index+1}/{initial_count}]: {row['Discovered_Channel_Name']}...")
            
            # Call the new LLM function
            result = is_direct_competitor_llm(
                seed_name=row['Seed_Channel_Name'],
                seed_niche_format=row['Seed_Niche'],
                candidate_name=row['Discovered_Channel_Name'],
                candidate_niche_format=row['Discovered_Niche'],
                model_provider="gpt" # Make sure this matches your util function
            )
            llm_check_results.append(result)
            
            # Add a delay between calls
            time.sleep(DELAY_BETWEEN_LLM_CHECKS)

    # Add results as a new column
    df_filtered['LLM_Competitor_Check'] = llm_check_results

    # Apply the final filter based on LLM check
    final_competitors = df_filtered[df_filtered['LLM_Competitor_Check'] == 'Yes']
    
    print(f"  - Final LLM Check: {len(final_competitors)} remain")

    # --- Final Results ---
    print("-" * 70)
    print(f"✅ Found {len(final_competitors)} final competitors after LLM check!")
    
    df_final = final_competitors.sort_values(by="LLM_Score", ascending=False)
    
    # Add the LLM Check column to the output
    display_columns = ['Seed_Channel_Name', 'Discovered_Channel_Name', 'Discovered_Niche', 'LLM_Score', 'Embedding_Score', 'Discovered_Country', 'LLM_Competitor_Check']
    # Select and reorder columns for the final output
    try:
        # Get all original columns + the new check column
        final_columns = [col for col in "INTERMEDIATE_COLUMN_ORDER" if col in df_final.columns] + ['LLM_Competitor_Check']
        df_final = df_final[final_columns]
    except Exception as e:
        print(f"  ⚠️ Warning: Could not reorder columns - {e}")

    # Display top results
    if not df_final.empty:
        print("\nTop 5 Final Competitors:")
        print(df_final[[col for col in display_columns if col in df_final.columns]].head().to_string(index=False))

    # Save the final list
    try:
        df_final.to_csv(final_competitors_path, index=False, encoding="utf-8-sig")
        print(f"\n💾 Saved final list to: {final_competitors_path.name}")
    except Exception as e:
        print(f"\n❌ ERROR saving final file: {e}")

    print("=" * 70)
    print("🎉 Filtering Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()