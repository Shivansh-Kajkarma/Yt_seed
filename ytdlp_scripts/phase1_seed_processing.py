import sys
import os
import json
import time
import random
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- 1. SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import Utils
try:
    from utils.youtube_utils import _load_from_google_sheet, fetch_recent_videos, extract_channel_id
    from utils.yt_dlp_utils import fetch_video_data_ytdlp
    from utils.mongo_utils import save_dataframe_to_mongo, save_json_blob
    from utils.fingerprint_llm_utils import get_channel_fingerprint_oneshot
except ImportError as e:
    print("❌ Error: Could not import from 'utils'.")
    raise e

# Config
OUTPUT_DIR = os.path.join(BASE_DIR, "ytdlp_scripts", "output", "phase1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CHANGED: Default lowered to 3 for testing ---
def process_single_seed(run_tag, channel_name, channel_url, client_format, client_intent, max_videos=10):
    print(f"\nStarted processing seed: {channel_name}")
    print(f"   🎯 Target Format: {client_format} | Intent: {client_intent}")

    # --- STEP 1: Resolve ID & Basic API Fetch ---
    channel_id = extract_channel_id(channel_url)
    if not channel_id:
        print(f"❌ Could not resolve Channel ID for {channel_url}")
        return False

    print(f"   ID: {channel_id} | Fetching recent {max_videos} videos via API...")
    
    api_videos, channel_desc = fetch_recent_videos(
        channel_id, 
        max_results=max_videos, 
        filter_shorts=True, 
        run_tag=run_tag,
        seed_name=channel_name
    )

    if not api_videos:
        print("❌ No long-form videos found via API.")
        return False

    # --- STEP 2: Deep Enrich with yt-dlp ---
    print(f"   ✅ API found {len(api_videos)} videos. Starting Deep Scan...")
    enriched_videos = []
    
    for i, vid in enumerate(api_videos):
        vid_id = vid['video_id']
        print(f"      [{i+1}/{len(api_videos)}] Deep scanning: {vid['title'][:40]}...")

        deep_data = fetch_video_data_ytdlp(vid_id)

        if deep_data:
            merged_data = {**vid, **deep_data}
            # Add metadata for context
            merged_data['Channel_Name'] = channel_name
            merged_data['Channel_ID'] = channel_id
            merged_data['channel_description'] = channel_desc
            merged_data['run_tag'] = run_tag
            
            enriched_videos.append(merged_data)
            has_trans = len(deep_data.get('caption_tracks', '') or '') > 0
            print(f"         -> Success. Transcript: {'Yes' if has_trans else 'No'} | Tags: {len(deep_data.get('tags', []))}")
        else:
            print("         -> Failed to fetch deep data. Skipping.")

        time.sleep(random.uniform(2.0, 4.0)) # Stealth delay

    if not enriched_videos:
        print("❌ All deep scans failed.")
        return False

    # --- STEP 3: Save Output ---
    df_enriched = pd.DataFrame(enriched_videos)
    
    try:
        save_dataframe_to_mongo(
            df_enriched, 
            f"{run_tag.upper()}_phase1", 
            unique_key_column="video_id"
        )
        print(f"   💾 Saved {len(df_enriched)} records to Mongo.")
    except Exception as e:
        print(f"   ⚠️ Mongo Save Error: {e}")

    # --- STEP 4: Generate Fingerprint (The "Super Context" Update) ---
    print("\n   🧠 Generating Targeted Fingerprint...")
    
    df_for_llm = df_enriched.copy()
    
    def combine_text_strict(row):
        """
        Creates a Meta-Block only using data that ACTUALLY exists.
        """
        # 1. Header Info
        title = row.get('title', 'Unknown Title')
        
        # 2. Conditional Metadata (The Fix)
        meta_lines = []
        
        # Category
        cat = row.get('category')
        if cat and cat != "Unknown" and cat != "None":
            meta_lines.append(f"CATEGORY: {cat}")
            
        # Tags (Clean & Filter)
        tags = row.get('tags')
        if tags and isinstance(tags, list) and len(tags) > 0:
            # Filter out generic/empty tags, take top 15
            clean_tags = [str(t).lower() for t in tags[:15] if t]
            if clean_tags:
                meta_lines.append(f"TAGS: {', '.join(clean_tags)}")
        
        # Chapters
        chapters = row.get('chapters')
        if chapters and isinstance(chapters, list) and len(chapters) > 0:
            # Just take the titles to save tokens
            chap_titles = [c.get('title', '') for c in chapters if c.get('title')]
            if chap_titles:
                meta_lines.append(f"CHAPTERS: {', '.join(chap_titles)}")

        # Join metadata lines
        meta_block = "\n".join(meta_lines)

        # 3. Transcript Slicing
        transcript = row.get('caption_tracks', '')
        if not isinstance(transcript, str): transcript = ""
        
        if len(transcript) > 50: # Only add if substantial text exists
            t_len = len(transcript)
            if t_len > 8000:
                head = transcript[:3000]
                mid_start = t_len // 2
                mid = transcript[mid_start:mid_start+2000]
                tail = transcript[-2000:]
                transcript_block = f"TRANSCRIPT SLICE:\n{head}\n...\n{mid}\n...\n{tail}"
            else:
                transcript_block = f"TRANSCRIPT:\n{transcript}"
        else:
            transcript_block = "[NO TRANSCRIPT AVAILABLE]"

        # 4. Final Assembly
        
        return f"""
        === VIDEO START ===
        TITLE: {title}
        {meta_block}
        {transcript_block}
        === VIDEO END ===
        """
    
    df_for_llm['formatted_content'] = df_for_llm.apply(combine_text_strict, axis=1)
    df_for_llm['description'] = df_for_llm['formatted_content']

    # Generate Fingerprint
    fingerprint = get_channel_fingerprint_oneshot(
        channel_name=channel_name,
        channel_description=channel_desc,
        video_df=df_for_llm,
        client_format=client_format, 
        client_intent=client_intent, 
        model_provider="gpt-4o"
    )

    # Save Fingerprint
    fp_wrapper = {
        "metadata": {
            "run_tag": run_tag,
            "created_at": datetime.now().isoformat(),
            "client_constraints": {
                "format": client_format,
                "intent": client_intent
            }
        },
        "channels": {
            channel_id: {
                "channel_name": channel_name,
                "fingerprint": fingerprint
            }
        }
    }
    save_json_blob(fp_wrapper, f"{run_tag.upper()}_phase1_fingerprints", "run_tag", run_tag)
    
    print("   ✅ Fingerprint generated and saved.")
    return True


def main():
    # --- CHANGED: Arg parsing ---
    if len(sys.argv) < 5:
        print("Usage: python ytdlp_scripts/phase1_seed_processing.py <run_tag> <sheet_url> <format> <intent>")
        print('Example: python ... moon "http://..." "Podcast" "Interviews with founders"')
        sys.exit(1)

    run_tag = sys.argv[1]
    input_source = sys.argv[2]
    client_format = sys.argv[3]
    client_intent = sys.argv[4]

    print(f"🚀 STARTING PHASE 1 | Tag: {run_tag}")
    print(f"📋 Constraints: {client_format} ({client_intent})")
    
    df_seeds = _load_from_google_sheet(input_source)
    if df_seeds is None or df_seeds.empty:
        print("❌ No seeds found.")
        return

    for _, row in df_seeds.iterrows():
        try:
            process_single_seed(
                run_tag, 
                row['Channel_Name'], 
                row['Channel_URL'],
                client_format,
                client_intent
            )
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🏁 Phase 1 Complete.")

if __name__ == "__main__":
    main()
