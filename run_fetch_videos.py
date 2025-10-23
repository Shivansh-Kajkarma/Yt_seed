# # /run_fetch_videos.py  (STEP 1 VERSION)

# import pandas as pd
# import sys
# from pathlib import Path
# from utils.youtube_utils import fetch_for_seed_channels 
# # We don't need fingerprinting in this script
# # from utils.fingerprint_utils import create_fingerprint 

# base_dir = Path(__file__).resolve().parent 

# def main():
#     """
#     Main function to run the video fetching process.
#     This version ONLY calls the API to create the 30-video sample file.
#     """
#     print("--- RUNNING IN API MODE (This will use quota) ---")
    
#     # --- This part is now active ---
#     print("Loading seed channels...")
#     seed_file_path = base_dir / "seed_channels.csv"
#     try:
#         seed_df = pd.read_csv(seed_file_path)
#     except FileNotFoundError:
#         print(f"ERROR: 'seed_channels.csv' not found at {seed_file_path}")
#         return

#     print(f"Found {len(seed_df)} channels. Fetching 30 videos for first 2 channels...")
    
#     # 1. Call the API for 30 videos
#     video_df = fetch_for_seed_channels(seed_df.head(2), limit_per_channel=30)

#     if video_df.empty:
#         print("No videos fetched.")
#         return

#     # 2. Save to a NEW file
#     output_path = base_dir / "sample_videos_30.csv" # <-- SAVE TO NEW FILE
#     video_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
#     print(f"\n✅ Success! Saved {len(video_df)} videos to 'sample_videos_30.csv'")
#     print("--- API Run Complete. You can now run the testing script. ---")

# if __name__ == "__main__":
#     main()


# /run_fetch_videos.py  (STEP 2 VERSION - SAFE)

import pandas as pd
import sys
from pathlib import Path

# We are COMMENTING OUT the API call function
# from utils.youtube_utils import fetch_for_seed_channels 

# We are ADDING the fingerprint function
from utils.fingerprint_utils import create_fingerprint 

base_dir = Path(__file__).resolve().parent 

def main():
    """
    Main function to run the FINGERPRINTING process from a local CSV.
    This version DOES NOT call the YouTube API.
    """
    print("Loading local video data (API calls are OFF)...")
    
    # --- API CALLS ARE NOW PROPERLY COMMENTED OUT ---
    # print("Loading seed channels...")
    # ... (all API code is gone)
    
    # --- Load directly from your NEW 30-video file ---
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