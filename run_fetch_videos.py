import pandas as pd
import sys
from pathlib import Path

# We are COMMENTING OUT the API call function
# from utils.youtube_utils import fetch_for_seed_channels 

# We are ADDING the fingerprint function
from utils.fingerprint_utils import create_fingerprint 

# Get the project root directory
base_dir = Path(__file__).resolve().parent 

def main():
    """
    Main function to run the FINGERPRINTING process from a local CSV.
    This version DOES NOT call the YouTube API.
    """
    print("Loading local video data (API calls are OFF)...")
    
    # --- API CALLS COMMENTED OUT ---
    # print("Loading seed channels...")
    # seed_file_path = base_dir / "seed_channels.csv"
    # try:
    #     seed_df = pd.read_csv(seed_file_path)
    # except FileNotFoundError:
    #     print(f"ERROR: 'seed_channels.csv' not found at {seed_file_path}")
    #     return
    # print(f"Found {len(seed_df)} channels. Fetching videos...")
    # video_df = fetch_for_seed_channels(seed_df.head(2))
    # ... (rest of the API call logic)
    # --- END OF COMMENTED OUT BLOCK ---

    # --- NEW: Load directly from sample_videos.csv ---
    video_file_path = base_dir / "sample_videos.csv"
    try:
        # Use your hardcoded path if base_dir fails, but base_dir should be correct
        # video_file_path = "/home/rareboy/Internship/Kajkarma/sample_videos.csv"
        video_df = pd.read_csv(video_file_path)
    except FileNotFoundError:
        print(f"ERROR: 'sample_videos.csv' not found at {video_file_path}")
        print("Please run the API call script once to generate this file.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Successfully loaded {len(video_df)} videos from 'sample_videos.csv'.")
    
    # Fill any potential missing descriptions with an empty string
    video_df['description'] = video_df['description'].fillna('')

    # --- NEW: Process fingerprints from the loaded CSV ---
    
    # Get a list of the unique channels in our CSV file
    unique_channels = video_df['Channel_Name'].unique()
    
    # Store fingerprints for Phase 3
    channel_fingerprints = {}

    for channel_name in unique_channels:
        print(f"\n--- Processing Channel: {channel_name} ---")
        
        # 1. Filter the DataFrame for *only* this channel's videos
        channel_video_df = video_df[video_df['Channel_Name'] == channel_name].copy()
        
        if channel_video_df.empty:
            print(f"No video data for {channel_name}. Skipping.")
            continue
            
        # 2. Create the fingerprint
        print(f"Found {len(channel_video_df)} videos. Creating fingerprint...")
        fingerprint = create_fingerprint(channel_video_df, top_n=15) # Get top 15 keywords
        
        # 3. Print and store the results for Phase 2
        print(f"Fingerprint for {channel_name}:")
        print(fingerprint)
        channel_fingerprints[channel_name] = fingerprint

    print("\n--- All Fingerprints Generated ---")
    print(channel_fingerprints)
    print("\nPhase 2 (Fingerprinting) test complete.")


if __name__ == "__main__":
    main()