import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# --- Make sure utils are importable ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    # --- 1. Import local utils ---
    from utils.fingerprint_llm_utils import (
        embedding_model,
        get_vector_from_texts,
        calculate_cosine_similarity,
        calculate_matrix_average_similarity,
        calculate_embedding_similarity_hybrid,  # <-- IMPORTING YOUR FUNCTION
    )

    print("✅ Successfully imported local embedding model and hybrid score function.")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import 'utils' folder or functions.")
    print(
        "   Make sure 'calculate_embedding_similarity_hybrid' is in your llm_utils.py"
    )
    sys.exit(1)

if not embedding_model:
    print("❌ CRITICAL ERROR: The 'embedding_model' from utils is None.")
    sys.exit(1)

# --- 2. Helper Functions ---


def find_latest_file_path(directory: Path, prefix: str, suffix: str) -> Path | None:
    """Finds the most recent file in a directory matching a pattern."""
    try:
        latest_file = max(directory.glob(f"{prefix}*{suffix}"), key=os.path.getctime)
        return latest_file
    except ValueError:
        return None


# --- 3. Main Triage Function ---
def main(run_tag: str):
    """
    Runs the full Phase 2.5 Embedding Triage pipeline.
    Calculates FIVE different types of embedding scores for comparison.
    """
    print(f"--- 🚀 Starting Phase 2.5: Embedding Triage for '{run_tag}' ---")

    # --- Define file paths ---
    FINGERPRINT_DIR = BASE_DIR / "FINGERPRINTS_ONESHOT"
    DISCOVERY_CACHE_DIR = BASE_DIR / "PHASE_2_DISCOVERY_CACHE"
    TRIAGE_REPORT_DIR = BASE_DIR / "PHASE_2_5_TRIAGE_REPORTS"

    os.makedirs(TRIAGE_REPORT_DIR, exist_ok=True)

    # --- A. Load Seed Channel Fingerprint (from Phase 1) ---
    print(f"\n[STEP 1/4] Loading Seed Fingerprint & Video Data...")
    SEED_FINGERPRINT_FILE = find_latest_file_path(
        FINGERPRINT_DIR, f"fingerprints_oneshot_{run_tag}", ".json"
    )
    SEED_VIDEOS_FILE = find_latest_file_path(
        BASE_DIR / "PHASE_1_OUTPUTS", f"sample_videos_{run_tag}", ".csv"
    )

    if not SEED_FINGERPRINT_FILE or not SEED_VIDEOS_FILE:
        print(f"❌ ERROR: Missing Phase 1 files for tag '{run_tag}'.")
        return

    print(f"  Using Seed Fingerprint: {SEED_FINGERPRINT_FILE.name}")
    print(f"  Using Seed Video Data: {SEED_VIDEOS_FILE.name}")

    try:
        with open(SEED_FINGERPRINT_FILE, "r") as f:
            seed_data = json.load(f)
        first_channel_key = next(iter(seed_data["channels"]))
        seed_channel_desc = (
            seed_data["channels"][first_channel_key]
            .get("fingerprint", {})
            .get("profile", {})
            .get("description", "")
        )
        seed_name = seed_data["channels"][first_channel_key].get(
            "channel_name", run_tag
        )

        df_seed_videos = pd.read_csv(SEED_VIDEOS_FILE)
        seed_video_titles = df_seed_videos["title"].dropna().tolist()
        seed_video_descs = df_seed_videos["description"].dropna().tolist()

    except Exception as e:
        print(f"❌ ERROR: Failed to load or parse seed files: {e}")
        return

    # --- B. Create the Seed Vectors (for Methods 1, 2, 3) ---
    print(f"\n[STEP 2/4] Creating base embedding vectors for Seed '{seed_name}'...")

    seed_vec_titles = get_vector_from_texts(seed_video_titles)
    seed_desc_texts = [seed_channel_desc] + [desc[:300] for desc in seed_video_descs]
    seed_vec_descs = get_vector_from_texts(seed_desc_texts)
    seed_all_texts = (
        [seed_channel_desc]
        + seed_video_titles
        + [desc[:300] for desc in seed_video_descs]
    )
    seed_vec_combined = get_vector_from_texts(seed_all_texts)

    print("✅ Seed vectors created successfully.")

    # --- C. Load Discovered Candidate Data (from Phase 2) ---
    print(f"\n[STEP 3/4] Loading Discovered Candidates...")
    DISCOVERY_CACHE_FILE = (
        DISCOVERY_CACHE_DIR / f"phase2_discovered_raw_data_{run_tag}.csv"
    )
    if not DISCOVERY_CACHE_FILE.exists():
        print(f"❌ ERROR: Phase 2 cache file not found: {DISCOVERY_CACHE_FILE.name}")
        return

    print(f"  Using Cache File: {DISCOVERY_CACHE_FILE.name}")
    df_candidates = pd.read_csv(DISCOVERY_CACHE_FILE)
    df_candidates["Discovered_Channel_Description"] = df_candidates[
        "Discovered_Channel_Description"
    ].fillna("")
    df_candidates["Discovered_Videos_JSON"] = df_candidates[
        "Discovered_Videos_JSON"
    ].fillna("[]")
    print(f"✅ Found {len(df_candidates)} candidates to analyze.")

    # --- D. Loop, Create Vectors, and Score ---
    print(f"\n[STEP 4/4] Starting embedding analysis loop...")

    results = []

    for index, row in df_candidates.iterrows():
        cand_name = row.get("Discovered_Channel_Name")
        if not cand_name:
            continue

        print(f"--- Processing {cand_name} ({index + 1}/{len(df_candidates)}) ---")

        try:
            cand_desc = row.get("Discovered_Channel_Description")
            cand_videos_json = row.get("Discovered_Videos_JSON")

            cand_videos_list = json.loads(cand_videos_json)
            if not cand_videos_list:
                print("  ⚠️ No videos in cache. Skipping.")
                continue
            df_cand_videos = pd.DataFrame(cand_videos_list)
            cand_video_titles = df_cand_videos["title"].dropna().tolist()
            cand_video_descs = df_cand_videos["description"].dropna().tolist()

            # --- Calculate all 5 scores ---

            # 1. AvgVec: Titles
            cand_vec_titles = get_vector_from_texts(cand_video_titles)
            score_titles_avg_vec = calculate_cosine_similarity(
                seed_vec_titles, cand_vec_titles
            )

            # 2. AvgVec: Descriptions
            cand_desc_texts = [cand_desc] + [desc[:300] for desc in cand_video_descs]
            cand_vec_descs = get_vector_from_texts(cand_desc_texts)
            score_descs_avg_vec = calculate_cosine_similarity(
                seed_vec_descs, cand_vec_descs
            )

            # 3. AvgVec: Combined
            cand_all_texts = (
                [cand_desc]
                + cand_video_titles
                + [desc[:300] for desc in cand_video_descs]
            )
            cand_vec_combined = get_vector_from_texts(cand_all_texts)
            score_combined_avg_vec = calculate_cosine_similarity(
                seed_vec_combined, cand_vec_combined
            )

            # 4. Your Hybrid Function (Method 4)
            score_hybrid_titles = calculate_embedding_similarity_hybrid(
                seed_video_titles, cand_video_titles
            )

            # 5. Your New Matrix Avg Idea (Method 5)
            score_matrix_avg = calculate_matrix_average_similarity(
                seed_video_titles, cand_video_titles
            )

            print(
                f"  ✅ Scores: Combined={score_combined_avg_vec:.3f}, Hybrid={score_hybrid_titles:.3f}, MatrixAvg={score_matrix_avg:.3f}"
            )

            # Store Results
            result_row = row.to_dict()
            result_row["Score_Emb_Titles_AvgVec"] = score_titles_avg_vec
            result_row["Score_Emb_Descs_AvgVec"] = score_descs_avg_vec
            result_row["Score_Emb_Combined_AvgVec"] = score_combined_avg_vec
            result_row["Score_Emb_Hybrid_Titles"] = score_hybrid_titles
            result_row["Score_Emb_Matrix_Avg"] = score_matrix_avg
            results.append(result_row)

        except Exception as e:
            print(f"  ❌❌ UNEXPECTED ERROR during processing for {cand_name}: {e}")
            continue

    # --- E. Save Final Report ---
    print(f"\n--- ANALYSIS COMPLETE ---")
    if not results:
        print("No channels were successfully scored.")
        return

    df_report = pd.DataFrame(results)
    df_report = df_report.sort_values(by="Score_Emb_Combined_AvgVec", ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save FULL report (all channels)
    FINAL_TRIAGE_FILE = (
        TRIAGE_REPORT_DIR / f"phase2_5_triage_report_{run_tag}_{timestamp}.csv"
    )
    df_report.to_csv(FINAL_TRIAGE_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ Full Triage Report saved to {FINAL_TRIAGE_FILE.name}")
    print(f"   Total channels: {len(df_report)}")

    # 2. Save FILTERED report (Score_Emb_Titles_AvgVec >= 0.5)
    df_filtered = df_report[df_report["Score_Emb_Combined_AvgVec"] >= 0.5].copy()
    FILTERED_TRIAGE_FILE = (
        TRIAGE_REPORT_DIR / f"phase2_5_triage_report_{run_tag}_filtered_{timestamp}.csv"
    )
    df_filtered.to_csv(FILTERED_TRIAGE_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ Filtered Triage Report saved to {FILTERED_TRIAGE_FILE.name}")
    print(f"   Filtered channels (Score_Emb_Titles_AvgVec >= 0.5): {len(df_filtered)}")
    print(f"   Channels removed: {len(df_report) - len(df_filtered)}")

    try:
        print("\n--- Top 10 Channels (by Combined Score) ---")
        cols_to_show = [
            "Discovered_Channel_Name",
            "Discovered_Subs",
            "Score_Emb_Combined_AvgVec",  # My main method
            "Score_Emb_Hybrid_Titles",  # Your hybrid function
            "Score_Emb_Matrix_Avg",  # Your new idea
        ]

        FINAL_TRIAGE_FILE_META = (
            TRIAGE_REPORT_DIR / f"phase2_5_triage_meta_{run_tag}_{timestamp}.csv"
        )
        df_report[
            cols_to_show + ["Discovered_Video_Count", "Discovered_Channel_URL"]
        ].to_csv(FINAL_TRIAGE_FILE_META, index=False, encoding="utf-8-sig")
        print(df_report[cols_to_show].head(10).to_markdown(index=False, floatfmt=".3f"))
    except Exception as e:
        print(f"Could not print markdown table: {e}")
        print(df_report[cols_to_show].head(10))


if __name__ == "__main__":
    try:
        import tabulate
    except ImportError:
        print(
            "⚠️  Warning: 'tabulate' module not found. `pip install tabulate` for cleaner log output."
        )

    # if len(sys.argv) < 2:
    #     print("❌ ERROR: You must provide a run_tag as an argument.")
    #     print("Example: python SCRIPTS/phase2_5_embedding_triage.py moon")
    #     sys.exit(1)

    # run_tag_arg = sys.argv[1].lower().strip()
    run_tag_arg = "moon"
    main(run_tag_arg)
