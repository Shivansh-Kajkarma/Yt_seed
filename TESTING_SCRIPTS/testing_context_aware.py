import pandas as pd
from pathlib import Path
import json
import sys
import time

# --- Import the NEW One-Shot function ---
try:
    # Make sure your utils file has this new function
    from utils.fingerprint_llm_utils import get_channel_fingerprint_oneshot
    LLM_CHECK_ENABLED = True
    print("✅ 'get_channel_fingerprint_oneshot' function imported.")
except ImportError:
    print("⚠️ WARNING: Could not import 'get_channel_fingerprint_oneshot'.")
    print("   Please add this new function to fingerprint_llm_utils.py first.")
    LLM_CHECK_ENABLED = False
    sys.exit() # Exit if we can't import the function

# === CONFIGURATION ===
base_dir = Path("/home/rareboy/Internship/Kajkarma")
MODEL_PROVIDER = "gpt" # or "gemini"

# --- Define the files and the channel names to test ---
# We will get the channel name from the CSV itself
FILES_TO_TEST = [
    base_dir / "sample_videos_moon.csv",
    base_dir / "sample_videos.csv"  # (for Vox)
]

# ============================================
# === TEST SCRIPT ===
# ============================================
def main():
    print("=" * 70)
    print("🚀 Running One-Shot Fingerprint Test (Context-Aware)")
    print("=" * 70)
    
    if not LLM_CHECK_ENABLED:
        return

    all_fingerprints = {}

    for file_path in FILES_TO_TEST:
        print(f"\n--- Processing File: {file_path.name} ---")
        
        try:
            df = pd.read_csv(file_path)
            # Ensure required columns exist
            required_cols = ['Channel_Name', 'channel_description', 'title', 'description']
            if not all(col in df.columns for col in required_cols):
                print(f"❌ ERROR: CSV is missing required columns. Need: {required_cols}")
                continue
            
            # Clean NAs
            df['channel_description'] = df['channel_description'].fillna("")
            df['title'] = df['title'].fillna("")
            df['description'] = df['description'].fillna("")

        except FileNotFoundError:
            print(f"❌ ERROR: File not found: {file_path.name}")
            continue
        except Exception as e:
            print(f"❌ ERROR loading file: {e}")
            continue
        
        if df.empty:
            print("   File is empty. Skipping.")
            continue
            
        # Get channel name and description from the first row
        channel_name = df['Channel_Name'].iloc[0]
        channel_desc = df['channel_description'].iloc[0]
        
        print(f"✅ Loaded {len(df)} videos for channel: {channel_name}")

        # --- Call the One-Shot Function ---
        # The 'video_df' is the dataframe 'df' itself
        fingerprint_json = get_channel_fingerprint_oneshot(
            channel_name=channel_name,
            channel_description=channel_desc,
            video_df=df, # Pass the whole DataFrame
            model_provider=MODEL_PROVIDER
        )
        
        print("-" * 70)
        print(f"🎉 One-Shot Fingerprint for {channel_name}:")
        if fingerprint_json:
            print(json.dumps(fingerprint_json, indent=2))
            all_fingerprints[channel_name] = fingerprint_json
        else:
            print("   Failed to generate fingerprint.")
        print("-" * 70)
        
        time.sleep(5) # Add a small delay between channels

    print("\n\n" + "=" * 70)
    print("✅ All tests complete.")
    
    # Optional: Save all results to one file
    if all_fingerprints:
        output_path = base_dir / "test_oneshot_fingerprints_4o_ver5.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_fingerprints, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved all test fingerprints to: {output_path.name}")
        except Exception as e:
            print(f"❌ ERROR saving JSON: {e}")
            
    print("=" * 70)

if __name__ == "__main__":
    main()