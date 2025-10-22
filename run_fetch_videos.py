import pandas as pd
import sys
from pathlib import Path
from utils.youtube_utils import fetch_for_seed_channels 

# Get the project root directory
base_dir = Path(__file__).resolve().parent 

def main():
    """
    Main function to run the video fetching process.
    """
    print("Loading seed channels...")
    # 1. Use base_dir to READ the input file
    seed_file_path = base_dir / "seed_channels.csv"
    try:
        seed_df = pd.read_csv(seed_file_path)
    except FileNotFoundError:
        print(f"ERROR: 'seed_channels.csv' not found at {seed_file_path}")
        return

    print(f"Found {len(seed_df)} channels. Fetching videos...")
    
    # Use head(2) for testing to save API quota
    # In production, you'd remove .head(2)
    video_df = fetch_for_seed_channels(seed_df.head(2))

    if video_df.empty:
        print("No videos fetched.")
        return

    # 2. Use base_dir to WRITE the output file (you already did this well)
    output_path = base_dir / "sample_videos.csv"
    video_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ Success! Saved {len(video_df)} videos to 'sample_videos.csv'")
    print("\nSample of data:")
    print(video_df.head())


    df = pd.read_csv("sample_videos.csv")
    print(df.loc[0, "description"])


# 3. Use the __name__ == "__main__" guard
if __name__ == "__main__":
    main()