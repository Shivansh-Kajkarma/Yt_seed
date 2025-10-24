# import pandas as pd
# import sys
# from pathlib import Path
# import time # Keep time import if you add delays back later

# # --- Import the NEW LLM fingerprint function ---
# from utils.fingerprint_llm_utils import create_channel_fingerprint_llm

# base_dir = Path(__file__).resolve().parent

# def main():
#     """
#     Main function to run LLM FINGERPRINTING from local CSV (Titles + Descriptions).
#     This version DOES NOT call YouTube Data API for video lists.
#     IT WILL CALL the Google Gemini API for keyword extraction.
#     """
#     print("Loading local video data (API calls for video lists are OFF)...")
#     # Using the FULL dataset
#     video_file_path = base_dir / "sample_videos_new.csv"

#     try: df = pd.read_csv(video_file_path)
#     except FileNotFoundError: print(f"ERROR: '{video_file_path.name}' not found."); return
#     except Exception as e: print(f"ERROR: Could not read CSV: {e}"); return
#     print(f"Successfully loaded {len(df)} videos from '{video_file_path.name}'.")

#     # Ensure necessary columns exist and fill NaNs
#     df['description'] = df['description'].fillna('')
#     df['title'] = df['title'].fillna('')
#     # Ensure Channel_ID exists (use Channel_Name as fallback if missing)
#     if 'Channel_ID' not in df.columns:
#         print("Warning: 'Channel_ID' column missing. Using 'Channel_Name' as identifier.")
#         df['Channel_ID'] = df['Channel_Name'] # Create it if missing

#     unique_channels = df['Channel_Name'].unique()
#     channel_fingerprints = {}

#     print("\n--- Generating All Fingerprints using Gemini LLM ---")
#     for channel_name in unique_channels:
#         try:
#             print(f"\n--- Processing Channel: {channel_name} ---")
#             channel_video_df = df[df['Channel_Name'] == channel_name].copy()
#             if channel_video_df.empty: print(f"No video data for {channel_name}. Skipping."); continue

#             # --- Call the new LLM fingerprint function ---
#             # It takes the DataFrame for the channel directly
#             print("Passing channel name as : ", channel_name)
#             fingerprint = create_channel_fingerprint_llm(channel_video_df, channel_name=channel_name)
#             # The 'top_n' logic is now inside the LLM prompt

#             print(f"✅ Fingerprint for {channel_name}:")
#             print(fingerprint)
#             channel_fingerprints[channel_name] = fingerprint

#             # Optional: Add a small delay between channels if hitting API limits
#             # time.sleep(2) # e.g., wait 2 seconds

#         except Exception as e:
#             print(f"\n!!!!!! ERROR PROCESSING CHANNEL: {channel_name} !!!!!!")
#             print(f"Error details: {e}")
#             channel_fingerprints[channel_name] = ["ERROR_PROCESSING"]

#     print("\n\n--- FINAL LLM (Gemini) FINGERPRINT REPORT ---")
#     for channel, fp in channel_fingerprints.items():
#         print(f"\n{channel}: \n{fp}")
#     print("\n--- LLM Fingerprinting test complete on all channels. ---")

# if __name__ == "__main__":
#     main()

# /run_fetch_videos.py (Using Gemini LLM Fingerprinter with JSON Export)

import pandas as pd
import sys
import json
from pathlib import Path
import time
from datetime import datetime

# Import the LLM fingerprint function
from utils.fingerprint_llm_utils import create_channel_fingerprint_llm

base_dir = Path(__file__).resolve().parent


def main():
    """
    Main function to run LLM FINGERPRINTING from local CSV.
    Saves results to JSON file.
    """
    print("Loading local video data (API calls for video lists are OFF)...")
    video_file_path = base_dir / "sample_videos_new.csv"

    try:
        df = pd.read_csv(video_file_path)
    except FileNotFoundError:
        print(f"ERROR: '{video_file_path.name}' not found.")
        return
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}")
        return
    
    print(f"Successfully loaded {len(df)} videos from '{video_file_path.name}'.")

    # Ensure necessary columns exist
    df['description'] = df['description'].fillna('')
    df['title'] = df['title'].fillna('')
    
    if 'Channel_ID' not in df.columns:
        print("Warning: 'Channel_ID' column missing. Using 'Channel_Name' as identifier.")
        df['Channel_ID'] = df['Channel_Name']

    unique_channels = df['Channel_Name'].unique()
    channel_fingerprints = {}
    
    # Track metadata
    processing_metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_channels": len(unique_channels),
        "total_videos_processed": len(df),
        "model": "gemini-2.0-flash-exp",
        "channels": {}
    }

    print("\n--- Generating All Fingerprints using Gemini LLM ---")
    
    for idx, channel_name in enumerate(unique_channels, 1):
        try:
            print(f"\n[{idx}/{len(unique_channels)}] Processing Channel: {channel_name}")
            channel_video_df = df[df['Channel_Name'] == channel_name].copy()
            
            if channel_video_df.empty:
                print(f"No video data for {channel_name}. Skipping.")
                continue

            # Call LLM fingerprint function
            fingerprint = create_channel_fingerprint_llm(
                channel_video_df, 
                channel_name=channel_name
            )

            print(f"✅ Fingerprint for {channel_name}:")
            print(fingerprint)
            
            # Store results
            channel_fingerprints[channel_name] = fingerprint
            
            # Store detailed metadata
            processing_metadata["channels"][channel_name] = {
                "keywords": fingerprint,
                "video_count": len(channel_video_df),
                "status": "success" if fingerprint else "failed"
            }

        except Exception as e:
            print(f"\n❌ ERROR PROCESSING CHANNEL: {channel_name}")
            print(f"Error details: {e}")
            channel_fingerprints[channel_name] = ["ERROR_PROCESSING"]
            processing_metadata["channels"][channel_name] = {
                "keywords": [],
                "video_count": len(channel_video_df) if 'channel_video_df' in locals() else 0,
                "status": "error",
                "error": str(e)
            }

    # === SAVE TO JSON ===
    output_json_path = base_dir / "channel_fingerprints.json"
    
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processing_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_json_path}")
        
        # Also save simplified version (just channel: keywords)
        simple_json_path = base_dir / "channel_keywords_simple.json"
        with open(simple_json_path, 'w', encoding='utf-8') as f:
            json.dump(channel_fingerprints, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Simple version saved to: {simple_json_path}")
        
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

    # === PRINT SUMMARY ===
    print("\n\n" + "="*60)
    print("FINAL LLM (Gemini) FINGERPRINT REPORT")
    print("="*60)
    
    for channel, fp in channel_fingerprints.items():
        status = "✅" if fp and fp[0] != "ERROR_PROCESSING" else "❌"
        print(f"\n{status} {channel}:")
        print(f"   {fp}")
    
    # Summary stats
    successful = sum(1 for ch in processing_metadata["channels"].values() if ch["status"] == "success")
    failed = len(unique_channels) - successful
    
    print("\n" + "="*60)
    print(f"📊 SUMMARY:")
    print(f"   Total Channels: {len(unique_channels)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total Videos: {len(df)}")
    print("="*60)
    
    print("\n--- LLM Fingerprinting test complete on all channels. ---")


if __name__ == "__main__":
    main()
