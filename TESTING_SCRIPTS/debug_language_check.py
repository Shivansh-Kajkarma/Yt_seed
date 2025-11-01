import pandas as pd
from pathlib import Path
import time
from utils.youtube_utils import get_channel_metadata_batch # We only need this function

# === CONFIGURATION ===
base_dir = Path(__file__).resolve().parent
intermediate_data_path = base_dir / "phase3_intermediate_data.csv"
backup_data_path = base_dir / "phase3_intermediate_data_BACKUP.csv" # For safety

print("=" * 70)
print("🔧 Adding Missing Country Data to CSV")
print("=" * 70)

# --- 1. Load Existing Data ---
print(f"Loading data from: {intermediate_data_path.name}")
try:
    df = pd.read_csv(intermediate_data_path)
except FileNotFoundError:
    print(f"❌ ERROR: File not found: {intermediate_data_path.name}")
    exit()
except Exception as e:
    print(f"❌ ERROR loading file: {e}")
    exit()

if 'Discovered_Country' in df.columns:
    print("✅ 'Discovered_Country' column already exists.")
    # Check if it's all "Unknown"
    if (df['Discovered_Country'] == 'Unknown').all():
        print("   Column is all 'Unknown'. Proceeding to fetch and overwrite...")
    else:
        print("   Column already contains data. Backing up and overwriting...")
        # continue
else:
    print("   'Discovered_Country' column not found, will add it.")

if 'Discovered_Channel_ID' not in df.columns:
    print("❌ ERROR: 'Discovered_Channel_ID' column is missing. Cannot proceed.")
    exit()

print(f"Loaded {len(df)} channels.")

# --- 2. Get Unique Channel IDs ---
unique_channel_ids = df['Discovered_Channel_ID'].unique().tolist()
print(f"Found {len(unique_channel_ids)} unique channels to fetch country for.")

if not unique_channel_ids:
    print("No channels found in the file.")
    exit()

# --- 3. Fetch Metadata (including Country) ---
print("Fetching metadata from YouTube API (this uses quota)...")
# This is the ONLY API call. It's batched and cheap.
metadata_list = get_channel_metadata_batch(unique_channel_ids)

if not metadata_list:
    print("❌ ERROR: Failed to fetch metadata from YouTube API.")
    exit()

# --- 4. Create a Mapping from ID to Country ---
country_map = {item['id']: item.get('country', 'Unknown') for item in metadata_list}
print(f"Successfully fetched metadata for {len(country_map)} channels.")

# --- 5. Add the 'Discovered_Country' Column ---
print("Adding 'Discovered_Country' column to the data...")
# Use the map to add the country based on the channel ID
df['Discovered_Country'] = df['Discovered_Channel_ID'].map(country_map).fillna("Unknown")

# --- 6. Save Backup and Overwrite Original ---
try:
    print(f"Saving backup to: {backup_data_path.name}")
    # Save backup just in case
    df_backup = pd.read_csv(intermediate_data_path)
    df_backup.to_csv(backup_data_path, index=False, encoding='utf-8-sig')

    print(f"Overwriting original file with added country data: {intermediate_data_path.name}")
    df.to_csv(intermediate_data_path, index=False, encoding='utf-8-sig')
    
    print("\n✅ Successfully added 'Discovered_Country' column!")
    print(f"   Quota cost estimate: ~{ (len(unique_channel_ids) // 50) + 1 } units.")

except Exception as e:
    print(f"\n❌ ERROR saving files: {e}")
    print("   Your original file might be unchanged, check the backup.")

print("=" * 70)
print("🔧 Process Complete")
print("=" * 70)