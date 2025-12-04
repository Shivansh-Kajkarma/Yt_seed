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
    from utils.youtube_utils import (
        _load_from_google_sheet,
        fetch_recent_videos,
        extract_channel_id,
    )
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
def process_single_seed(
    run_tag, channel_name, channel_url, client_format, client_intent, max_videos=10, api_key=None
):
    print(f"\nStarted processing seed: {channel_name}")
    print("API Key Present:", bool(api_key))
    print("\nAPI Key: ", api_key)
    print(f"   🎯 Target Format: {client_format} | Intent: {client_intent}")

    # --- STEP 1: Resolve ID & Basic API Fetch ---
    channel_id = extract_channel_id(channel_url, api_key=api_key)
    if not channel_id:
        print(f"❌ Could not resolve Channel ID for {channel_url}")
        return False

    print(f"   ID: {channel_id} | Fetching recent {max_videos} videos via API...")

    api_videos, channel_desc = fetch_recent_videos(
        channel_id,
        max_results=max_videos,
        filter_shorts=True,
        run_tag=run_tag,
        seed_name=channel_name,
        api_key=api_key
    )

    if not api_videos:
        print("❌ No long-form videos found via API.")
        return False

    # --- STEP 2: Deep Enrich with yt-dlp ---
    print(f"   ✅ API found {len(api_videos)} videos. Starting Deep Scan...")
    enriched_videos = []

    for i, vid in enumerate(api_videos):
        vid_id = vid["video_id"]
        print(
            f"      [{i + 1}/{len(api_videos)}] Deep scanning: {vid['title'][:40]}..."
        )

        deep_data = fetch_video_data_ytdlp(vid_id)

        if deep_data:
            merged_data = {**vid, **deep_data}
            # Add metadata for context
            merged_data["Channel_Name"] = channel_name
            merged_data["Channel_ID"] = channel_id
            merged_data["channel_description"] = channel_desc
            merged_data["run_tag"] = run_tag

            enriched_videos.append(merged_data)
            has_trans = len(deep_data.get("caption_tracks", "") or "") > 0
            print(
                f"         -> Success. Transcript: {'Yes' if has_trans else 'No'} | Tags: {len(deep_data.get('tags', []))}"
            )
        else:
            print("         -> Failed to fetch deep data. Skipping.")

        time.sleep(random.uniform(2.0, 4.0))  # Stealth delay

    if not enriched_videos:
        print("❌ All deep scans failed.")
        return False

    # --- STEP 3: Save Output ---
    df_enriched = pd.DataFrame(enriched_videos)

    try:
        save_dataframe_to_mongo(
            df_enriched, f"{run_tag.upper()}_phase1", unique_key_column="video_id"
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
        title = row.get("title", "Unknown Title")

        # 2. Conditional Metadata (The Fix)
        meta_lines = []

        # Category
        cat = row.get("category")
        if cat and cat != "Unknown" and cat != "None":
            meta_lines.append(f"CATEGORY: {cat}")

        # Tags (Clean & Filter)
        tags = row.get("tags")
        if tags and isinstance(tags, list) and len(tags) > 0:
            # Filter out generic/empty tags, take top 15
            clean_tags = [str(t).lower() for t in tags[:15] if t]
            if clean_tags:
                meta_lines.append(f"TAGS: {', '.join(clean_tags)}")

        # Chapters
        chapters = row.get("chapters")
        if chapters and isinstance(chapters, list) and len(chapters) > 0:
            # Just take the titles to save tokens
            chap_titles = [c.get("title", "") for c in chapters if c.get("title")]
            if chap_titles:
                meta_lines.append(f"CHAPTERS: {', '.join(chap_titles)}")

        # Join metadata lines
        meta_block = "\n".join(meta_lines)

        # 3. Transcript Slicing
        transcript = row.get("caption_tracks", "")
        if not isinstance(transcript, str):
            transcript = ""

        if len(transcript) > 50:  # Only add if substantial text exists
            t_len = len(transcript)
            if t_len > 8000:
                head = transcript[:3000]
                mid_start = t_len // 2
                mid = transcript[mid_start : mid_start + 2000]
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


    df_for_llm["formatted_content"] = df_for_llm.apply(combine_text_strict, axis=1)

    # IMPORTANT: We use formatted_content for the LLM, regardless of transcript presence
    # because it cleans up the tags/categories better than raw description.

    safe_channel_desc = str(channel_desc) if pd.notna(channel_desc) else "N/A"
    
    # Use formatted_content if available (Phase 1 logic), else raw text
    if 'formatted_content' in df_for_llm.columns:
        content_blocks = df_for_llm['formatted_content'].astype(str).tolist()
    else:
        content_blocks = (df_for_llm['title'] + "\n" + df_for_llm['description']).astype(str).tolist()

    full_content_str = "\n".join(content_blocks)
    truncated_content = full_content_str[:50000]

    dossier = f"=== TARGET CHANNEL DOSSIER ===\nNAME: {channel_name}\nBIO: {safe_channel_desc[:2000]}\n\n--- CONTENT DATA ---\n{truncated_content}\n"
    
    dossier += f"\n--- CONTENT ANALYSIS DATA ---\n{truncated_content}\n"
    fingerprint = get_channel_fingerprint_oneshot(
        channel_name=channel_name,
        dossier=dossier,
        client_format=client_format,
        client_intent=client_intent,
        model_provider="gpt-4o",
    )

    if not fingerprint:
        print("❌ Failed to generate fingerprint.")
        return False

    # Save Fingerprint to Local File
    fp_wrapper = {
        "metadata": {
            "run_tag": run_tag,
            "created_at": datetime.now().isoformat(),
            "client_constraints": {"format": client_format, "intent": client_intent},
        },
        "channels": {
            channel_id: {"channel_name": channel_name, "fingerprint": fingerprint}
        },
    }
    save_json_blob(
        fp_wrapper, f"{run_tag.upper()}_phase1_fingerprints", "run_tag", run_tag
    )

    print("   ✅ Fingerprint generated and saved.")
    return True


def main():
    # --- CHANGED: Arg parsing - supports both sheet URL and direct channel data ---
    if len(sys.argv) < 6:
        print(
            "Usage: python ytdlp_scripts/phase1_seed_processing.py <run_tag> <channel_name> <channel_url> <format> <intent>"
        )
        print(
            'Example: python ... moon "Lenny\'s Podcast" "https://youtube.com/@lennypodcast" "Podcast" "Interviews with founders"'
        )
        sys.exit(1)

    run_tag = sys.argv[1]
    channel_name = sys.argv[2]
    channel_url = sys.argv[3]
    client_format = sys.argv[4]
    client_intent = sys.argv[5]
    api_key = os.environ.get("YOUTUBE_API_KEY_DYNAMIC")
    if not api_key:
        print("⚠️ No Dynamic API Key found. Falling back to default env loading.")

    print(f"🚀 STARTING PHASE 1 | Tag: {run_tag} | API Key: {'[REDACTED]' if api_key else 'None'} | Key Present: {bool(api_key)}")
    print(f"📋 Seed: {channel_name}")
    print(f"📋 Constraints: {client_format} ({client_intent})")

    # Process single seed directly (no sheet loading needed)
    try:
        success = process_single_seed(
            run_tag,
            channel_name,
            channel_url,
            client_format,
            client_intent,
            max_videos=10,  # Can adjust this
            api_key=api_key
        )

        if success:
            print("\n🏁 Phase 1 Complete - SUCCESS")
        else:
            print("\n⚠️ Phase 1 Complete - No videos found")
            pass  # Exit with error code if no videos

    except Exception as e:
        print(f"❌ Error: {e}")
        # import traceback

        # traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()
