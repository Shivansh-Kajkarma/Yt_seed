import pandas as pd
import sys
from pathlib import Path
from utils.gemini_utils import create_channel_fingerprint_gemini


base_dir = Path(__file__).resolve().parent

def main():
    """
    Main function to run the FINGERPRINTING process using Gemini from a local CSV.
    This version DOES NOT call the YouTube API.
    """
    print("Loading local video data (API calls are OFF)...")

    # Load from your FULL sample file
    video_file_path = base_dir / "sample_videos_30.csv" # <-- Make sure this is correct

    try:
        video_df = pd.read_csv(video_file_path)
    except FileNotFoundError:
        print(f"ERROR: '{video_file_path.name}' not found.")
        return

    print(f"Successfully loaded {len(video_df)} videos from '{video_file_path.name}'.")

    # Ensure necessary columns exist and fill NaNs
    video_df['description'] = video_df['description'].fillna('')
    video_df['title'] = video_df['title'].fillna('')
    # Ensure video_id column exists for transcript fetching
    if 'video_id' not in video_df.columns:
         print("ERROR: 'video_id' column missing from CSV. Cannot fetch transcripts.")
         return
    video_df['video_id'] = video_df['video_id'].fillna('')


    unique_channels = video_df['Channel_Name'].unique()
    channel_fingerprints = {}

    print("\n--- Generating All Fingerprints using Gemini ---")
    for channel_name in unique_channels:
        print(f"\n--- Processing Channel: {channel_name} ---")

        # Filter the DataFrame for the current channel
        channel_video_df = video_df[video_df['Channel_Name'] == channel_name].copy()

        if channel_video_df.empty:
            print(f"No video data for {channel_name}. Skipping.")
            continue

        print(f"Found {len(channel_video_df)} videos.")

        # --- THIS IS THE MAIN CHANGE ---
        # Get the required data for the Gemini function
        c_video_ids = channel_video_df['video_id'].tolist()
        c_titles = channel_video_df['title'].tolist()
        c_descriptions = channel_video_df['description'].tolist()
        channel_id_placeholder = channel_video_df['Channel_ID'].iloc[0] # Get channel ID for context

        # Call the new Gemini function
        fingerprint = create_channel_fingerprint_gemini(
            channel_id=channel_id_placeholder, # Pass the actual channel ID
            video_ids=c_video_ids,
            video_titles=c_titles,
            video_descriptions=c_descriptions
        )


        print(f"Fingerprint for {channel_name}:")
        print(fingerprint)
        channel_fingerprints[channel_name] = fingerprint

    print("\n\n--- FINAL GEMINI FINGERPRINT REPORT ---")
    for channel, fp in channel_fingerprints.items():
        print(f"\n{channel}: \n{fp}")
    print("\n--- Gemini Fingerprinting test complete on all channels. ---")


if __name__ == "__main__":
    main()