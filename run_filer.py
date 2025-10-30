import pandas as pd
from pathlib import Path
import re # Keep for future use, though not needed for this version

# === CONFIGURATION ===
base_dir = Path(__file__).resolve().parent
intermediate_data_path = base_dir / "phase3_intermediate_data.csv"
# I've given it a new name to be clear what's in it
final_competitors_path = base_dir / "FINAL_COMPETITORS_CountrySubFiltered.csv" 
final_competitors_path_meta = base_dir / "FINAL_COMPETITORS_CountrySubFiltered_meta.csv" 

# ============================================
# === 📊 YOUR FILTER RULES ===
# ============================================

# --- 1. Country Filter (ENABLED) ---
# We will ONLY keep channels from these countries, as you listed
AUTO_KEEP_COUNTRIES = [
    'US',  # United States
    'GB',  # United Kingdom
    'CA',  # Canada
    'AU',  # Australia
    'NZ',  # New Zealand
    'NG',  # Nigeria
]

# --- 2. Subscriber Filter (ENABLED) ---
MIN_SUBS = 0          # Minimum subscribers

# --- Other Filters (DISABLED FOR NOW, as requested) ---
# MIN_LLM_SCORE = 0.0
# MIN_EMBEDDING_SCORE = 0.0
# FORMAT_EXCLUSIONS = []
# LLM_CHECK_ENABLED = False 

# ============================================
# === FILTERING LOGIC ===
# ============================================

def main():
    print("=" * 70)
    print("🚀 Starting Filter Script (Country & Subscribers ONLY)")
    print("=" * 70)
    print(f"Loading data from: {intermediate_data_path.name}")
    try:
        df = pd.read_csv(intermediate_data_path)
        # Fill NA values to avoid errors during filtering
        df['Discovered_Country'] = df['Discovered_Country'].fillna("Unknown")
        df['Discovered_Subs'] = df['Discovered_Subs'].fillna(0)
        
        # Fill other NAs just in case, though we aren't using them
        df['Discovered_Niche'] = df['Discovered_Niche'].fillna("Unknown - Unknown")
        df['LLM_Score'] = df['LLM_Score'].fillna(0.0)
        df['Embedding_Score'] = df['Embedding_Score'].fillna(0.0)

    except FileNotFoundError:
         print(f"❌ ERROR: File not found: {intermediate_data_path.name}")
         print("   Did you run 'run_discovery_phase3.py' and 'add_country_data.py'?")
         return
    except KeyError as e:
         print(f"❌ ERROR: Missing expected column in CSV: {e}")
         print("   Did you run 'add_country_data.py' successfully?")
         return
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return

    print(f"✅ Loaded {len(df)} total scored candidates.")
    print("-" * 70)
    print("📊 Applying Filters...")

    initial_count = len(df)
    df_filtered = df.copy()

    # --- Filter 1: Country Filter (Your Plan) ---
    if AUTO_KEEP_COUNTRIES:
        country_mask = df_filtered['Discovered_Country'].isin(AUTO_KEEP_COUNTRIES)
        df_filtered = df_filtered[country_mask] # Keep ONLY rows IN the list
        print(f"  - Country Filter (Keep {', '.join(AUTO_KEEP_COUNTRIES)}): {len(df_filtered)} / {initial_count} remain")
    else:
        print("  - Country Filter: SKIPPED (no countries defined)")

    # --- Filter 2: Subscribers ---
    if MIN_SUBS > 0 :
        df_filtered = df_filtered[df_filtered['Discovered_Subs'] >= MIN_SUBS]
        print(f"  - Subscriber Filter (>={MIN_SUBS:,}): {len(df_filtered)} remain")
    else: 
        print(f"  - Subscriber Filter: SKIPPED (MIN_SUBS is 0)")

    # --- Other filters are skipped ---
    print("  - Score Filters: SKIPPED (as requested)")
    print("  - Niche Format Filter: SKIPPED (as requested)")
    print("  - Final LLM Competitor Check: SKIPPED (as requested)")

    # --- Final Results ---
    print("-" * 70)
    print(f"✅ Found {len(df_filtered)} channels matching Country & Subscriber criteria!")
    
    # Sort by Subscribers (highest first) since that's a main filter
    df_final = df_filtered.sort_values(by="Discovered_Subs", ascending=False)
    
    # Define columns for display
    display_columns = [
        'Seed_Channel_Name', 'Discovered_Channel_Name', 'Discovered_Niche',
        'Discovered_Subs', 'Discovered_Country', 'LLM_Score', 'Embedding_Score'
    ]
    existing_display_cols = [col for col in display_columns if col in df_final.columns]

    # Display all results that passed
    if not df_final.empty:
        print("\nChannels Found (Sorted by Subscribers):")
        # Print all rows, not just head
        print(df_final[existing_display_cols].to_string(index=False))
    
    # Save the final list
    try:
        # Save all original columns for the filtered channels
        df_final.to_csv(final_competitors_path, index=False, encoding="utf-8-sig")
        df_final[display_columns].to_csv(final_competitors_path_meta, index=False, encoding="utf-8-sig")
        print(f"\n💾 Saved final list to: {final_competitors_path.name}")
    except Exception as e:
        print(f"\n❌ ERROR saving final file: {e}")

    print("=" * 70)
    print("🎉 Filtering Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()