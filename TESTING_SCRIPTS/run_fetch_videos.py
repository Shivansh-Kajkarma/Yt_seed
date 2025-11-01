import pandas as pd
import sys
from pathlib import Path

from utils.fingerprint_utils import create_fingerprint 

base_dir = Path(__file__).resolve().parent 

def main():
    """
    Main function to run the FINGERPRINTING process from a local CSV.
    This version DOES NOT call the YouTube API.
    """
    print("Loading local video data (API calls are OFF)...")
    
    video_file_path = base_dir / "sample_videos_30.csv" # <-- LOAD NEW FILE
    try:
        video_df = pd.read_csv(video_file_path)
    except FileNotFoundError:
        print(f"ERROR: '{video_file_path.name}' not found.")
        print("Please run the 'STEP 1' version of this script once.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Successfully loaded {len(video_df)} videos from '{video_file_path.name}'.")
    
    video_df['description'] = video_df['description'].fillna('')

    # --- Process fingerprints from the loaded CSV ---
    unique_channels = video_df['Channel_Name'].unique()
    channel_fingerprints = {}

    for channel_name in unique_channels:
        print(f"\n--- Processing Channel: {channel_name} ---")
        
        channel_video_df = video_df[video_df['Channel_Name'] == channel_name].copy()
        
        if channel_video_df.empty:
            print(f"No video data for {channel_name}. Skipping.")
            continue
            
        print(f"Found {len(channel_video_df)} videos. Creating fingerprint...")
        fingerprint = create_fingerprint(channel_video_df, top_n=15)
        
        print(f"Fingerprint for {channel_name}:")
        print(fingerprint)
        channel_fingerprints[channel_name] = fingerprint

    print("\n--- All Fingerprints Generated ---")
    print(channel_fingerprints)
    print("\nPhase 2 (Fingerprinting) test complete.")


if __name__ == "__main__":
    main()