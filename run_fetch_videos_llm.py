import pandas as pd
import sys
import json
from pathlib import Path
import time
from datetime import datetime

# --- MODIFIED: Import both functions now ---
from utils.fingerprint_llm_utils import (
    create_channel_fingerprint_llm, 
    extract_niche_llm
)

base_dir = Path(__file__).resolve().parent

# --- ADDED: Set your model choice in ONE place ---
# Change this to "gemini" if you want to use Gemini
MODEL_TO_USE = "gpt"
MODEL_NAME = "gpt-4o" if MODEL_TO_USE == "gpt" else "gemini-2.0-flash-exp"


def main():
    """
    Main function to run LLM FINGERPRINTING from local CSV.
    Implements 2-STEP logic:
    1. Extract Niche
    2. Extract Keywords (guided by Niche)
    """
    print(f"--- Starting LLM Fingerprint Run ({MODEL_NAME}) ---")
    print("Loading local video data (API calls for video lists are OFF)...")
    video_file_path = base_dir / "sample_videos.csv"

    try:
        df = pd.read_csv(video_file_path)
    except FileNotFoundError:
        print(f"ERROR: '{video_file_path.name}' not found.")
        return
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}")
        return
    
    print(f"Successfully loaded {len(df)} videos from '{video_file_path.name}'.")

    # --- MODIFIED: Ensure all required columns exist ---
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    
    # This is the CRITICAL new column from your youtube_utils update
    if 'channel_description' not in df.columns:
        print("ERROR: 'channel_description' column not found in CSV.")
        print("Please re-run your YouTube data fetching script to add it.")
        return
    df['channel_description'] = df['channel_description'].fillna('')
    
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
        "model": MODEL_NAME,  # --- MODIFIED: Use dynamic model name
        "channels": {}
    }

    print(f"\n--- Generating All Fingerprints using {MODEL_NAME} ---")
    
    for idx, channel_name in enumerate(unique_channels, 1):
        niche = "ERROR"
        fingerprint = []
        
        try:
            print(f"\n[{idx}/{len(unique_channels)}] Processing Channel: {channel_name}")
            channel_video_df = df[df['Channel_Name'] == channel_name].copy()
            
            if channel_video_df.empty:
                print(f"No video data for {channel_name}. Skipping.")
                continue

            # --- STEP 1: Get Niche (Using your new HYBRID function) ---
            print(f"  Step 1/2: Extracting niche for {channel_name}...")
            
            # --- STEP 1: Get Niche (Using your new HYBRID function) ---
            print(f"  Step 1/2: Extracting niche for {channel_name}...")

            # The empty check already happened at line 77, so it's safe to access here
            channel_desc = channel_video_df['channel_description'].iloc[0]
            video_titles_list = channel_video_df['title'].tolist()

            niche = extract_niche_llm(
                channel_description=channel_desc,
                video_titles=video_titles_list,
                channel_name=channel_name,
                model_type=MODEL_TO_USE
            )

            # --- STEP 2: Get Keywords (Guided by the Niche) ---
            print(f"  Step 2/2: Extracting keywords (Niche: {niche})...")
            
            fingerprint = create_channel_fingerprint_llm(
                channel_video_df, 
                channel_name=channel_name,
                niche=niche,  # <-- Pass the niche here
                model_type=MODEL_TO_USE
            )

            print(f"✅ Fingerprint for {channel_name}:")
            # print(fingerprint)
            print(json.dumps(fingerprint, indent=2))
            
            # Store results
            channel_fingerprints[channel_name] = fingerprint
            
            # Store detailed metadata
            processing_metadata["channels"][channel_name] = {
                "niche": niche,  # <-- ADDED Niche to metadata
                "keywords": fingerprint,
                "video_count": len(channel_video_df),
                "status": "success" if fingerprint else "failed"
            }

        except Exception as e:
            print(f"\n❌ ERROR PROCESSING CHANNEL: {channel_name}")
            print(f"Error details: {e}")
            channel_fingerprints[channel_name] = ["ERROR_PROCESSING"]
            processing_metadata["channels"][channel_name] = {
                "niche": niche if niche != "ERROR" else "Niche extraction failed",
                "keywords": [],
                "video_count": len(channel_video_df) if 'channel_video_df' in locals() else 0,
                "status": "error",
                "error": str(e)
            }

    # === SAVE TO JSON ===
    output_json_path = base_dir / f"channel_fingerprints_{MODEL_TO_USE}.json"
    
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processing_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_json_path}")
        
        # Also save simplified version (just channel: keywords)
        simple_json_path = base_dir / f"channel_keywords_simple_{MODEL_TO_USE}_29th_oct_4pm.json"
        with open(simple_json_path, 'w', encoding='utf-8') as f:
            json.dump(channel_fingerprints, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Simple version saved to: {simple_json_path}")
        
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

    # === PRINT SUMMARY ===
    print("\n\n" + "="*60)
    print(f"FINAL LLM ({MODEL_NAME}) FINGERPRINT REPORT")
    print("="*60)
    
    # for channel, fp in channel_fingerprints.items():
        # status = "✅" if fp and fp[0] != "ERROR_PROCESSING" else "❌"
        # # --- MODIFIED: Also print the niche in the summary ---
        # niche_found = processing_metadata["channels"][channel].get('niche', 'N/A')
        # print(f"\n{status} {channel} (Niche: {niche_found}):")
        # print(f"   {fp}")

    for channel, fp in channel_fingerprints.items():
        # --- MODIFIED: Check if fp is a list and has content before accessing index ---
        is_error = not fp or (isinstance(fp, list) and fp and fp[0] == "ERROR_PROCESSING")
        status = "❌" if is_error else "✅"
        niche_found = processing_metadata["channels"][channel].get('niche', 'N/A')
        print(f"\n{status} {channel} (Niche: {niche_found}):")
        # Check if fp is a dictionary (new format) or list (old format/error)
        if isinstance(fp, dict):
            # Pretty print the dictionary for the summary
            print(f"   {json.dumps(fp, indent=4)}") # <-- CHANGE: Pretty print dictionary
            # Optional: Calculate total keywords
            # total_kws = sum(len(v) for v in fp.values())
            # print(f"   (Total Keywords: {total_kws})")
        else:
            # Handle the error case or potentially old list format
            print(f"   {fp}") # Print the error list or unexpected format
    
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
    
    print(f"\n--- LLM Fingerprinting test ({MODEL_NAME}) complete. ---")


if __name__ == "__main__":
    main()