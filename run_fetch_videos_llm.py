# /run_fetch_videos.py (Using Gemini LLM Fingerprinter)

import pandas as pd
import sys
from pathlib import Path
import time # Keep time import if you add delays back later

# --- Import the NEW LLM fingerprint function ---
from utils.fingerprint_llm_utils import create_channel_fingerprint_llm
# --- Remove or comment out old imports ---
# from utils.llm_utils import create_channel_fingerprint_llm # <-- Old file name
# from utils.gemini_utils import create_channel_fingerprint_gemini
# from utils.fingerprint_utils import create_fingerprint

base_dir = Path(__file__).resolve().parent

def main():
    """
    Main function to run LLM FINGERPRINTING from local CSV (Titles + Descriptions).
    This version DOES NOT call YouTube Data API for video lists.
    IT WILL CALL the Google Gemini API for keyword extraction.
    """
    print("Loading local video data (API calls for video lists are OFF)...")
    # Using the FULL dataset
    video_file_path = base_dir / "sample_videos_10.csv"

    try: df = pd.read_csv(video_file_path)
    except FileNotFoundError: print(f"ERROR: '{video_file_path.name}' not found."); return
    except Exception as e: print(f"ERROR: Could not read CSV: {e}"); return
    print(f"Successfully loaded {len(df)} videos from '{video_file_path.name}'.")

    # Ensure necessary columns exist and fill NaNs
    df['description'] = df['description'].fillna('')
    df['title'] = df['title'].fillna('')
    # Ensure Channel_ID exists (use Channel_Name as fallback if missing)
    if 'Channel_ID' not in df.columns:
        print("Warning: 'Channel_ID' column missing. Using 'Channel_Name' as identifier.")
        df['Channel_ID'] = df['Channel_Name'] # Create it if missing

    unique_channels = df['Channel_Name'].unique()
    channel_fingerprints = {}

    print("\n--- Generating All Fingerprints using Gemini LLM ---")
    for channel_name in unique_channels:
        try:
            print(f"\n--- Processing Channel: {channel_name} ---")
            channel_video_df = df[df['Channel_Name'] == channel_name].copy()
            if channel_video_df.empty: print(f"No video data for {channel_name}. Skipping."); continue

            # --- Call the new LLM fingerprint function ---
            # It takes the DataFrame for the channel directly
            fingerprint = create_channel_fingerprint_llm(channel_video_df)
            # The 'top_n' logic is now inside the LLM prompt

            print(f"✅ Fingerprint for {channel_name}:")
            print(fingerprint)
            channel_fingerprints[channel_name] = fingerprint

            # Optional: Add a small delay between channels if hitting API limits
            # time.sleep(2) # e.g., wait 2 seconds

        except Exception as e:
            print(f"\n!!!!!! ERROR PROCESSING CHANNEL: {channel_name} !!!!!!")
            print(f"Error details: {e}")
            channel_fingerprints[channel_name] = ["ERROR_PROCESSING"]

    print("\n\n--- FINAL LLM (Gemini) FINGERPRINT REPORT ---")
    for channel, fp in channel_fingerprints.items():
        print(f"\n{channel}: \n{fp}")
    print("\n--- LLM Fingerprinting test complete on all channels. ---")

if __name__ == "__main__":
    main()