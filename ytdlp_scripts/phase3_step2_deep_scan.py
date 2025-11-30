import sys
import os
import json
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo
    from utils.yt_dlp_utils import fetch_video_data_ytdlp
    from utils.fingerprint_llm_utils import get_candidate_fingerprint_independent
except ImportError as e:
    print("❌ Error importing utils.")
    raise e

# CONFIG
VIDEOS_TO_SCAN = 3  # changed 5->3 # We scan 5 to get a representative sample
# REMOVED: MIN_PODCAST_DURATION = 600 (Client says duration doesn't matter)


def get_client_constraints(run_tag):
    """Fetch constraints to pass to SOTA Analyzer."""
    try:
        df = load_collection_as_df(
            f"{run_tag.upper()}_phase1_fingerprints", {"metadata.run_tag": run_tag}
        )
        if df.empty:
            return "General", "General"
        meta = df.iloc[0].get("metadata", {})
        constraints = meta.get("client_constraints", {})
        return constraints.get("format", "General"), constraints.get(
            "intent", "General"
        )
    except:
        return "General", "General"


def combine_text_strict(row):
    """
    Creates a Meta-Block only using data that ACTUALLY exists.
    (Exact copy of Phase 1 Logic)
    """
    # 1. Header Info
    title = row.get("title", "Unknown Title")

    # 2. Conditional Metadata
    meta_lines = []

    # Category
    cat = row.get("category")
    if cat and cat != "Unknown" and cat != "None":
        meta_lines.append(f"CATEGORY: {cat}")

    # Tags (Clean & Filter)
    tags = row.get("tags")
    if tags and isinstance(tags, list) and len(tags) > 0:
        clean_tags = [str(t).lower() for t in tags[:15] if t]
        if clean_tags:
            meta_lines.append(f"TAGS: {', '.join(clean_tags)}")

    # Chapters
    chapters = row.get("chapters")
    if chapters and isinstance(chapters, list) and len(chapters) > 0:
        chap_titles = [c.get("title", "") for c in chapters if c.get("title")]
        if chap_titles:
            meta_lines.append(f"CHAPTERS: {', '.join(chap_titles)}")

    meta_block = "\n".join(meta_lines)

    # 3. Transcript Slicing
    transcript = row.get("caption_tracks", "")
    if not isinstance(transcript, str):
        transcript = ""

    if len(transcript) > 50:
        t_len = len(transcript)
        if t_len > 3000:
            head = transcript[:3000]
            transcript_block = head
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python ytdlp_scripts/phase3_step2_deep_scan.py <run_tag>")
        sys.exit(1)

    run_tag = sys.argv[1]
    print(f"🚀 PHASE 3 STEP B: Deep Scan (yt-dlp) | Tag: {run_tag}")
    print("ℹ️  Constraint Update: Duration Filter REMOVED (Format > Duration).")

    client_format, client_intent = get_client_constraints(run_tag)
    print(f"📋 Constraints: {client_format} | {client_intent}")

    # 1. Load Step 1 Survivors
    collection_in = f"{run_tag.upper()}_phase3_step1"
    try:
        df_candidates = load_collection_as_df(collection_in)
        if df_candidates.empty:
            print("❌ No candidates found in Step 1.")
            return
        print(f"📥 Loaded {len(df_candidates)} candidates to scan.")
    except Exception as e:
        print(f"❌ Error loading candidates: {e}")
        return

    scanned_results = []

    for index, row in df_candidates.iterrows():
        channel_name = row.get("Discovered_Channel_Name")

        # Parse videos from Phase 2 (API data) to get IDs
        videos_json = row.get("Discovered_Videos_JSON", "[]")
        try:
            video_list = json.loads(videos_json)
        except:
            video_list = []

        if not video_list:
            print(f"⚠️ Skipping {channel_name} (No video IDs)")
            continue

        print(f"\n[{index + 1}/{len(df_candidates)}] Deep Scanning: {channel_name}")

        # 3. Stealth Scan (Top 3)
        # We need this deep data to prove the "100% Format Match" in the next step
        target_videos = video_list[:VIDEOS_TO_SCAN]
        deep_data_list = []

        for vid in target_videos:
            vid_id = vid.get("video_id")

            # Call yt-dlp utils (Stealth Mode)
            data = fetch_video_data_ytdlp(vid_id)

            if data:
                deep_data_list.append(data)
                print(
                    f"    ✅ Fetched: {data['title'][:30]}... (Has Trans: {len(data.get('caption_tracks', '') or '') > 0})"
                )
            else:
                print(f"    ❌ Failed to fetch: {vid_id}")

            # Sleep to stay safe
            time.sleep(random.uniform(4.0, 6.0))

        if not deep_data_list:
            print(f"    ⚠️ All scans failed for {channel_name}. Skipping.")
            continue

        # =========================================================
        # 4. GENERATE FINGERPRINT (The New Integration)
        # =========================================================
        print(f"    🧠 Creating Dossier & Calling One-Shot LLM...")

        # A. Convert List to DataFrame
        df_deep = pd.DataFrame(deep_data_list)

        # B. Apply formatting logic (Create Meta-Blocks)
        df_deep["formatted_content"] = df_deep.apply(combine_text_strict, axis=1)

        if "formatted_content" in df_deep.columns:
            content_blocks = df_deep["formatted_content"].astype(str).tolist()
        else:
            content_blocks = (
                (df_deep["title"] + "\n" + df_deep["description"]).astype(str).tolist()
            )

        full_content_str = "\n".join(content_blocks)
        truncated_content = full_content_str[:50000]

        dossier = f"=== TARGET CHANNEL DOSSIER ===\nNAME: {channel_name}\n--- CONTENT DATA ---\n{truncated_content}\n"

        # C. Call the Shared Function
        # Note: We use gpt-4o-mini here to save cost on volume, or gpt-4o if quality is paramount
        fingerprint = get_candidate_fingerprint_independent(
            dossier=dossier,
            model_provider="gpt-4o-mini",
        )

        # =========================================================
        # 5. PREPARE OUTPUT & SAVE
        # =========================================================
        row_dict = row.to_dict()

        # Save raw deep data
        row_dict["Deep_Scan_Data"] = json.dumps(deep_data_list)

        # Save Fingerprint Data (Niche/Intent/Keywords)
        if fingerprint:
            row_dict["generated_niche"] = fingerprint.get("generated_niche", "Unknown")
            row_dict["generated_intent"] = fingerprint.get(
                "generated_intent", "Unknown"
            )
            row_dict["generated_keywords"] = json.dumps(
                fingerprint.get("search_keywords", [])
            )
            print(f"    ✅ Fingerprinted: {row_dict['generated_niche']}")
        else:
            row_dict["generated_niche"] = "Unknown"
            print(f"    ⚠️ Fingerprint Failed")

        # Calculate metrics (For Context only, not Filtering)
        durations = [d.get("duration", 0) for d in deep_data_list]
        avg_duration = np.mean(durations) if durations else 0
        has_chapters_count = sum(1 for d in deep_data_list if d.get("has_chapters"))

        # Add metrics to SAME row_dict (don't recreate!)
        row_dict["step2_avg_duration"] = avg_duration
        row_dict["step2_has_chapters"] = has_chapters_count > 0
        row_dict["step2_status"] = "scanned"

        # Status message
        print(
            f"    ✅ DATA ACQUIRED. Avg Duration: {int(avg_duration)}s | Chapters: {has_chapters_count}/{len(deep_data_list)}"
        )

        scanned_results.append(row_dict)

        # Save-as-you-go
        save_dataframe_to_mongo(
            pd.DataFrame([row_dict]),
            f"{run_tag.upper()}_phase3_step2",
            "Discovered_Channel_ID",
        )

    print("\n🏁 Phase 3 Step B Complete.")


if __name__ == "__main__":
    main()
