import pandas as pd
import os, sys, json, time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from utils.fingerprint_llm_utils import gpt_client, get_channel_tier_gpt
    from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo
    print("✅ Successfully imported GPT + Mongo utils.")
except Exception as e:
    print(f"❌ Import error: {e}")
    raise e


def main(run_tag: str):
    print(f"\n🚀 Starting Phase 3 (GPT Tier Scoring) for '{run_tag}'")
    # ---------------- CONFIG ----------------
    if not gpt_client:
        print("❌ CRITICAL: GPT client not initialized. Cannot run Phase 3.")
        # This will fail the task, but not kill the worker
        raise ValueError("CRITICAL: GPT client not initialized.")

    MONGO_COLLECTION_PHASE1_FP = f"{run_tag.upper()}_phase1_fingerprints"
    MONGO_COLLECTION_PHASE2_5 = f"{run_tag.upper()}_phase2_5"
    MONGO_COLLECTION_PHASE3 = f"{run_tag.upper()}_phase3"
    FINAL_REPORT_DIR = BASE_DIR / "PHASE_3_REPORTS"
    FINAL_REPORT_DIR.mkdir(exist_ok=True)
    # ----------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    JSON_OUTPUT_FILE = FINAL_REPORT_DIR / f"client_tier_analysis_{run_tag}_{timestamp}.json"
    INTERNAL_CSV_FILE = FINAL_REPORT_DIR / f"phase3_INTERNAL_report_{run_tag}_{timestamp}.csv"

    # ======================================================
    # STEP 1. Load Seed Profile + Keywords from Mongo
    # ======================================================
    print(f"\n[STEP 1/3] Loading Seed Profile & Keywords from Mongo...")
    try:
        df_fp = load_collection_as_df(MONGO_COLLECTION_PHASE1_FP, {"metadata.run_tag": run_tag})
        if df_fp.empty:
            raise ValueError("No fingerprints found in Mongo.")
        seed_data = df_fp.iloc[0].to_dict()

        first_channel_key = next(iter(seed_data["channels"]))
        fingerprint = seed_data["channels"][first_channel_key].get("fingerprint", {})
        seed_profile_obj = fingerprint.get("profile")
        seed_keywords_obj = fingerprint.get("keywords")
        seed_channel_name = seed_data["channels"][first_channel_key].get("channel_name", run_tag)

        seed_profile_str = json.dumps(seed_profile_obj, indent=2)
        all_keywords = []
        for cat, kws in seed_keywords_obj.items():
            all_keywords.append(f"\nCategory: {cat}")
            all_keywords.extend([f"- {kw}" for kw in kws])
        seed_keywords_str = "\n".join(all_keywords)

        print(f"✅ Loaded profile for '{seed_channel_name}' ({len(seed_keywords_obj)} categories)")
    except Exception as e:
        print(f"❌ ERROR loading seed profile: {e}")
        return

    # ======================================================
    # STEP 2. Load Filtered Candidates from Mongo
    # ======================================================
    print(f"\n[STEP 2/3] Loading filtered candidates from Mongo ({MONGO_COLLECTION_PHASE2_5})...")
    try:
        df_candidates = load_collection_as_df(
            MONGO_COLLECTION_PHASE2_5,
            {"Score_Emb_Combined_AvgVec": {"$gte": 0.45}}
        )
        if df_candidates.empty:
            print("⚠️ No filtered candidates found (Score_Emb_Combined_AvgVec ≥ 0.45).")
            return
        print(f"✅ Loaded {len(df_candidates)} candidates to score.")
    except Exception as e:
        print(f"❌ ERROR loading candidates: {e}")
        return

    # ======================================================
    # STEP 3. LLM Tier Analysis Loop
    # ======================================================
    json_results = {}
    new_data_list = []

    for index, row in df_candidates.iterrows():
        channel_name = row.get("Discovered_Channel_Name")
        channel_id = row.get("Discovered_Channel_ID")
        print(f"\n--- Processing: {channel_name} ({index+1}/{len(df_candidates)}) ---")

        try:
            videos_json = row.get("Discovered_Videos_JSON", "[]")
            videos_list = json.loads(videos_json)

            if not videos_list:
                print("⚠️  No videos in cache. Skipping.")
                tier_data = {"tier": -1, "reason": "No videos in cache"}
            else:
                video_titles = [v.get("title", "") for v in videos_list if v.get("title")]
                tier_data = get_channel_tier_gpt(
                    seed_name=seed_channel_name,
                    candidate_name=channel_name,
                    candidate_titles=video_titles,
                    seed_profile_str=seed_profile_str,
                    seed_categories_str="",
                    seed_keywords_str=seed_keywords_str
                )

            print(f"  ✅ Tier: {tier_data.get('tier')} | Reason: {tier_data.get('reason')[:80]}")

            tier_data["Discovered_Channel_ID"] = channel_id
            tier_data["Discovered_Channel_Name"] = channel_name
            tier_data["run_tag"] = run_tag
            tier_data["mirrored_at"] = datetime.now().isoformat()

            json_results[channel_name] = tier_data
            new_data_list.append(tier_data)

            # Save-as-you-go Mongo upsert (in case of crashes)
            save_dataframe_to_mongo(pd.DataFrame([tier_data]), MONGO_COLLECTION_PHASE3, "Discovered_Channel_ID")
            time.sleep(1)

        except Exception as e:
            print(f"❌ ERROR processing {channel_name}: {e}")
            continue

    # ======================================================
    # STEP 4. Final Saving
    # ======================================================
    print("\n--- ANALYSIS COMPLETE ---")
    if not new_data_list:
        print("No channels were successfully tiered.")
        return

    df_tiers = pd.DataFrame(new_data_list)
    df_tiers = df_tiers.sort_values(by="tier", ascending=True)
    df_tiers.to_csv(INTERNAL_CSV_FILE, index=False, encoding="utf-8-sig")
    with open(JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"✅ CSV saved: {INTERNAL_CSV_FILE.name}")
    print(f"✅ JSON saved: {JSON_OUTPUT_FILE.name}")
    print(f"✅ Mongo: {len(df_tiers)} documents mirrored into '{MONGO_COLLECTION_PHASE3}'")

    print("\n🎯 Phase 3 complete! Ready for FINAL_OUTPUT merging stage.")

# -----------------
if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "DEFAULT"
    main(tag)