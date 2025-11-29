import sys
import os
import json
import time
import random
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import Utils
try:
    from utils.youtube_utils import extract_channel_id, fetch_recent_videos
    from utils.yt_dlp_utils import fetch_video_data_ytdlp
    from utils.mongo_utils import (
        save_dataframe_to_mongo,
        save_json_blob,
        load_collection_as_df,
    )
    from utils.fingerprint_llm_utils import get_channel_fingerprint_oneshot
except ImportError as e:
    print("❌ Error: Could not import from 'utils'.")
    raise e


def combine_text_strict(row):
    """Creates a Meta-Block using available data."""
    title = row.get("title", "Unknown Title")
    meta_lines = []

    # Category
    cat = row.get("category")
    if cat and cat not in ["Unknown", "None"]:
        meta_lines.append(f"CATEGORY: {cat}")

    # Tags
    tags = row.get("tags")
    if tags and isinstance(tags, list):
        clean_tags = [str(t).lower() for t in tags[:15] if t]
        if clean_tags:
            meta_lines.append(f"TAGS: {', '.join(clean_tags)}")

    # Transcript
    transcript = row.get("caption_tracks", "")
    if not isinstance(transcript, str):
        transcript = ""

    if len(transcript) > 50:
        if len(transcript) > 8000:
            head = transcript[:3000]
            mid_start = len(transcript) // 2
            mid = transcript[mid_start : mid_start + 2000]
            tail = transcript[-2000:]
            transcript_block = f"TRANSCRIPT SLICE:\n{head}\n...\n{mid}\n...\n{tail}"
        else:
            transcript_block = f"TRANSCRIPT:\n{transcript}"
    else:
        transcript_block = "[NO TRANSCRIPT AVAILABLE]"

    return (
        f"=== VIDEO START ===\nTITLE: {title}\n"
        + "\n".join(meta_lines)
        + f"\n{transcript_block}\n=== VIDEO END ==="
    )


def process_single_seed(
    run_tag, channel_name, channel_url, client_format, client_intent, max_videos=10
):
    print(f"\nStarted processing seed: {channel_name}")
    print(f"   🎯 Format: {client_format} | Intent: {client_intent}")

    channel_id = extract_channel_id(channel_url)
    channel_desc = "Description not fetched in test mode"  # Default

    # --- 1. CHECK MONGODB FOR EXISTING DATA (TEST MODE) ---
    collection_name = f"{run_tag.upper()}_phase1"
    print(f"   🔍 Checking Mongo for existing data in '{collection_name}'...")

    try:
        df_existing = load_collection_as_df(collection_name)

        if not df_existing.empty:
            print(f"   ✅ FOUND {len(df_existing)} existing videos. SKIPPING FETCHING.")
            df_enriched = df_existing

            # Try to grab description from the first row if available
            if "channel_description" in df_enriched.columns:
                channel_desc = df_enriched.iloc[0].get("channel_description", "")

        else:
            # --- 2. DATA NOT FOUND -> FETCH (PROD MODE) ---
            print("   ⚠️  No existing data found. Starting Production Fetch...")

            if not channel_id:
                print(f"❌ Could not resolve Channel ID for {channel_url}")
                return False

            api_videos, channel_desc = fetch_recent_videos(
                channel_id,
                max_results=max_videos,
                filter_shorts=True,
                run_tag=run_tag,
                seed_name=channel_name,
            )

            if not api_videos:
                print("❌ No videos found via API.")
                return False

            print(f"   ✅ API found {len(api_videos)} videos. Starting Deep Scan...")
            enriched_videos = []

            for i, vid in enumerate(api_videos):
                vid_id = vid["video_id"]
                print(
                    f"      [{i + 1}/{len(api_videos)}] Deep scanning: {vid['title'][:40]}..."
                )
                deep_data = fetch_video_data_ytdlp(vid_id)

                if deep_data:
                    merged = {**vid, **deep_data}
                    merged["Channel_Name"] = channel_name
                    merged["Channel_ID"] = channel_id
                    merged["channel_description"] = channel_desc
                    merged["run_tag"] = run_tag
                    enriched_videos.append(merged)
                    time.sleep(random.uniform(2.0, 4.0))

            if not enriched_videos:
                return False

            df_enriched = pd.DataFrame(enriched_videos)
            save_dataframe_to_mongo(df_enriched, collection_name, "video_id")

    except Exception as e:
        print(f"❌ Error during data load/fetch: {e}")
        return False

    # --- 3. GENERATE FINGERPRINT (ALWAYS RUNS) ---
    print("\n   🧠 Generating Modular Fingerprint...")

    df_for_llm = df_enriched.copy()

    # Apply your formatting logic
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

    # save_json_blob(fp_wrapper, f"{run_tag.upper()}_phase1_fingerprints", "run_tag", run_tag)


    # Save to local JSON file instead of MongoDB
    output_dir = os.path.join(BASE_DIR, "testing_new", "output")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"{run_tag}_fingerprint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(fp_wrapper, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Fingerprint generated and saved to: {output_file}")
    return True


def main():
    if len(sys.argv) < 6:
        print(
            "Usage: python ... <run_tag> <channel_name> <channel_url> <format> <intent>"
        )
        sys.exit(1)

    run_tag = sys.argv[1]
    channel_name = sys.argv[2]
    channel_url = sys.argv[3]
    client_format = sys.argv[4]
    client_intent = sys.argv[5]

    try:
        process_single_seed(
            run_tag, channel_name, channel_url, client_format, client_intent
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise e


if __name__ == "__main__":
    main()
