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
    # --- Import utils ---
    from utils.fingerprint_llm_utils import (
        embedding_model,
        get_vector_from_texts,
        calculate_cosine_similarity,
        calculate_matrix_average_similarity,
        calculate_embedding_similarity_hybrid,
    )
    from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo

    print("✅ Successfully imported local embedding model and Mongo utils.")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import from utils: {e}")
    sys.exit(1)

if not embedding_model:
    print("❌ CRITICAL ERROR: 'embedding_model' is None.")
    sys.exit(1)

# --- Config ---
run_tag = "moon"  # your seed tag
MONGO_COLLECTION_PREFIX = f"{run_tag.upper()}_phase2_5"

# --- Main function ---
def main(run_tag: str):
    print(f"\n🚀 Starting Phase 2.5: Embedding Triage for '{run_tag}'")

    TRIAGE_REPORT_DIR = BASE_DIR / "PHASE_2_5_TRIAGE_REPORTS"
    os.makedirs(TRIAGE_REPORT_DIR, exist_ok=True)

    print(f"\n[STEP 1/4] Loading Seed Fingerprint & Video Data from Mongo...")

    try:
        # --- Load fingerprint (latest run) ---
        df_fp = load_collection_as_df(
            f"{run_tag.upper()}_phase1_fingerprints",
            {"metadata.run_tag": run_tag}
        )

        if df_fp.empty:
            raise ValueError(f"No fingerprints found for '{run_tag}' in Mongo.")

        # --- Handle missing metadata.created_at safely ---
        if "metadata.created_at" in df_fp.columns:
            df_fp = df_fp.sort_values("metadata.created_at", ascending=False)
        elif "mirrored_at" in df_fp.columns:
            df_fp = df_fp.sort_values("mirrored_at", ascending=False)
        else:
            # fallback to insertion order if no timestamp field
            df_fp = df_fp.sort_index(ascending=False)

        seed_data = df_fp.iloc[0].to_dict()

        # --- Extract seed channel info ---
        first_channel_key = next(iter(seed_data.get("channels", {})))
        seed_channel_desc = (
            seed_data["channels"][first_channel_key]
            .get("fingerprint", {})
            .get("profile", {})
            .get("description", "")
        )
        seed_name = seed_data["channels"][first_channel_key].get("channel_name", run_tag)

        # --- Load seed videos ---
        df_seed_videos = load_collection_as_df(f"{run_tag.upper()}_phase1")
        if df_seed_videos.empty:
            raise ValueError(f"No seed videos found for '{run_tag}' in Mongo.")

        seed_video_titles = df_seed_videos["title"].dropna().tolist()
        seed_video_descs = df_seed_videos["description"].dropna().tolist()

        print(f"✅ Loaded {len(seed_video_titles)} seed videos for '{seed_name}'.")

    except Exception as e:
        print(f"❌ ERROR loading seed fingerprint/video data from Mongo: {e}")
        return

    # ============================================================
    # STEP 2: Create Embedding Vectors
    # ============================================================
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

    # ============================================================
    # STEP 3: Load Discovered Candidates from MongoDB
    # ============================================================
    print(f"\n[STEP 3/4] Loading Discovered Candidates from Mongo...")
    try:
        df_candidates = load_collection_as_df(f"{run_tag.upper()}_phase2")
        if df_candidates.empty:
            raise ValueError(f"No discovered candidates found for '{run_tag}'.")

        df_candidates["Discovered_Channel_Description"] = df_candidates[
            "Discovered_Channel_Description"
        ].fillna("")
        df_candidates["Discovered_Videos_JSON"] = df_candidates[
            "Discovered_Videos_JSON"
        ].fillna("[]")

        print(f"✅ Found {len(df_candidates)} candidates to analyze.")

    except Exception as e:
        print(f"❌ ERROR loading discovered candidates from Mongo: {e}")
        return

    # ============================================================
    # STEP 4: Embedding Analysis Loop
    # ============================================================
    print(f"\n[STEP 4/4] Starting embedding analysis loop...")
    results = []

    for index, row in df_candidates.iterrows():
        cand_name = row.get("Discovered_Channel_Name")
        if not cand_name:
            continue

        print(f"--- Processing {cand_name} ({index + 1}/{len(df_candidates)}) ---")
        try:
            cand_desc = row.get("Discovered_Channel_Description", "")
            cand_videos_json = row.get("Discovered_Videos_JSON", "[]")

            cand_videos_list = json.loads(cand_videos_json)
            if not cand_videos_list:
                print("  ⚠️ No videos in cache. Skipping.")
                continue

            df_cand_videos = pd.DataFrame(cand_videos_list)
            cand_video_titles = df_cand_videos.get("title", pd.Series()).dropna().tolist()
            cand_video_descs = df_cand_videos.get("description", pd.Series()).dropna().tolist()

            # Compute all scores
            cand_vec_titles = get_vector_from_texts(cand_video_titles)
            score_titles_avg_vec = calculate_cosine_similarity(seed_vec_titles, cand_vec_titles)

            cand_desc_texts = [cand_desc] + [desc[:300] for desc in cand_video_descs]
            cand_vec_descs = get_vector_from_texts(cand_desc_texts)
            score_descs_avg_vec = calculate_cosine_similarity(seed_vec_descs, cand_vec_descs)

            cand_all_texts = [cand_desc] + cand_video_titles + [desc[:300] for desc in cand_video_descs]
            cand_vec_combined = get_vector_from_texts(cand_all_texts)
            score_combined_avg_vec = calculate_cosine_similarity(seed_vec_combined, cand_vec_combined)

            score_hybrid_titles = calculate_embedding_similarity_hybrid(seed_video_titles, cand_video_titles)
            score_matrix_avg = calculate_matrix_average_similarity(seed_video_titles, cand_video_titles)

            print(
                f"  ✅ Scores: Combined={score_combined_avg_vec:.3f}, Hybrid={score_hybrid_titles:.3f}, MatrixAvg={score_matrix_avg:.3f}"
            )

            # Save scores
            result_row = row.to_dict()
            result_row.update({
                "Score_Emb_Titles_AvgVec": score_titles_avg_vec,
                "Score_Emb_Descs_AvgVec": score_descs_avg_vec,
                "Score_Emb_Combined_AvgVec": score_combined_avg_vec,
                "Score_Emb_Hybrid_Titles": score_hybrid_titles,
                "Score_Emb_Matrix_Avg": score_matrix_avg,
            })
            results.append(result_row)

        except Exception as e:
            print(f"  ❌ ERROR processing {cand_name}: {e}")
            continue

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print(f"\n--- ANALYSIS COMPLETE ---")
    if not results:
        print("No channels were successfully scored.")
        return

    df_report = pd.DataFrame(results).sort_values(by="Score_Emb_Combined_AvgVec", ascending=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    FINAL_TRIAGE_FILE = TRIAGE_REPORT_DIR / f"phase2_5_triage_report_{run_tag}_{timestamp}.csv"
    df_report.to_csv(FINAL_TRIAGE_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ Full Triage Report saved: {FINAL_TRIAGE_FILE.name}")

    # Save to MongoDB
    try:
        df_report["run_tag"] = run_tag
        df_report["mirrored_at"] = datetime.now().isoformat()
        save_dataframe_to_mongo(
            df_report,
            collection_name=f"{run_tag.upper()}_phase2_5",
            unique_key_column="Discovered_Channel_ID",
        )
        print(f"✅ Mongo: {len(df_report)} scored channels upserted to '{run_tag.upper()}_phase2_5'.")
    except Exception as e:
        print(f"❌ Mongo push failed for Phase 2.5: {e}")

    # Save filtered report
    df_filtered = df_report[df_report["Score_Emb_Combined_AvgVec"] >= 0.45].copy()
    FILTERED_FILE = TRIAGE_REPORT_DIR / f"phase2_5_triage_filtered_{run_tag}_{timestamp}.csv"
    df_filtered.to_csv(FILTERED_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ Filtered report saved: {FILTERED_FILE.name} ({len(df_filtered)} channels)")

    # Optional top 10 printout
    try:
        print("\n--- Top 10 Channels ---")
        cols = ["Discovered_Channel_Name", "Discovered_Subs",
                "Score_Emb_Combined_AvgVec", "Score_Emb_Hybrid_Titles", "Score_Emb_Matrix_Avg"]
        print(df_report[cols].head(10).to_markdown(index=False, floatfmt=".3f"))
    except Exception as e:
        print(f"⚠️ Could not print top 10 table: {e}")

# --- Entrypoint ---
if __name__ == "__main__":
    try:
        import tabulate
    except ImportError:
        print("⚠️ Install 'tabulate' for prettier logs.")
    main("moon")
