import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo
    from utils.fingerprint_llm_utils import gpt_client
except ImportError as e:
    print("❌ Error importing utils.")
    raise e

# CONFIG
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "ytdlp_scripts", "output", "final_reports")
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# --- CALIBRATED THRESHOLDS ---
THRES_TIER_2_AUTO = 0.82  # Auto Tier 2 if format failed but content score high
THRES_LLM_CHECK = 0.70  # Min score to qualify for LLM re-check


def get_client_constraints(run_tag):
    """Fetch Format AND Intent (Niche) from Phase 1."""
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


# --- LLM FORMAT RE-CHECK (For borderline cases) ---
def run_format_recheck_llm(row_dict, client_format):
    """Re-checks format for channels with False format_match but good content score."""
    name = row_dict.get("Discovered_Channel_Name")
    deep_json = row_dict.get("Deep_Scan_Data", "[]")
    try:
        deep_data = json.loads(deep_json)
    except:
        return False, "No data"

    dossier = ""
    for i, vid in enumerate(deep_data[:2]):
        raw_trans = vid.get("caption_tracks")
        transcript = (raw_trans or "")[:1500]
        dossier += f"VIDEO {i + 1}: {vid.get('title')}\nTRANSCRIPT: {transcript}\n---\n"

    prompt = f"""You are a Format Verification Expert.
    
    CONTEXT:
    Channel "{name}" failed initial format check but has VERY HIGH content similarity (score > 0.7).
    We need a second opinion.
    
    TARGET FORMAT: "{client_format}"
    
    YOUR TASK:
    Review the transcripts below. Does this channel ACTUALLY match the format?
    
    CRITICAL RULES:
    1. Focus on FORMAT structure, not just topic relevance.
    2. For "Podcast": Look for dialogue, interviews, conversational flow.
    3. For "Tutorial": Look for instructional language, step-by-step guidance.
    4. Be strict but fair - high content similarity suggests they cover similar topics.
    
    DOSSIER:
    {dossier}
    
    DECISION:
    Return JSON: {{ "is_match": true/false, "reason": "Brief explanation of format assessment." }}
    """
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        res = json.loads(response.choices[0].message.content)
        return res.get("is_match", False), res.get("reason", "Format recheck")
    except:
        return False, "LLM Error"


def main():
    if len(sys.argv) < 2:
        print("Usage: python ytdlp_scripts/phase4_final_ranking.py <run_tag>")
        sys.exit(1)

    run_tag = sys.argv[1]
    pipeline_execution_id = sys.argv[2] if len(sys.argv) > 2 else "Unknown"
    
    print(f"🚀 PHASE 4 ... | Tag: {run_tag} | Pipeline ID: {pipeline_execution_id}")

    client_format, client_intent = get_client_constraints(run_tag)
    print(f"📋 Constraints: Format='{client_format}', Niche='{client_intent}'")
    print(f"📊 Thresholds: Auto T2>={THRES_TIER_2_AUTO} | LLM Check>={THRES_LLM_CHECK}")

    # 1. Load Data
    try:
        df_llm = load_collection_as_df(f"{run_tag.upper()}_phase3_step3a")
        df_emb = load_collection_as_df(f"{run_tag.upper()}_phase3_step3b")
        if df_llm.empty or df_emb.empty:
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. Merge (Include all sub-scores from Phase 3b)
    df_merged = pd.merge(
        df_llm,  # Keep all columns from phase3_step3a
        df_emb[
            [
                "Discovered_Channel_ID",
                "score_similarity",
                "sub_score_niche",
                "sub_score_intent",
                "sub_score_keywords",
                "sub_score_content",
            ]
        ],
        on="Discovered_Channel_ID",
        how="inner",
    )

    print(f"🔗 Merged {len(df_merged)} candidates. Starting New Ranking Logic...")

    final_results = []
    llm_check_queue = []

    # 3. NEW SIMPLIFIED LOGIC
    for index, row in df_merged.iterrows():
        fmt_match = row["score_format_match"]
        content_score = row["sub_score_content"]
        row_dict = row.to_dict()
        row_dict["pipeline_execution_id"] = pipeline_execution_id

        # RULE 1: Format Match = True → Tier 1 (sorted by content score)
        if fmt_match:
            row_dict["Final_Tier"] = 1
            row_dict["Final_Status"] = "Tier 1 (Format Match)"
            final_results.append(row_dict)
            print(
                f"  ✅ Tier 1: {row_dict['Discovered_Channel_Name']} (content: {content_score:.3f})"
            )

        # RULE 2: Format Match = False BUT content >= 0.82 → Auto Tier 2
        elif not fmt_match and content_score >= THRES_TIER_2_AUTO:
            row_dict["Final_Tier"] = 2
            row_dict["Final_Status"] = "Tier 2 (High Content Score)"
            final_results.append(row_dict)
            print(
                f"  ⚡ Tier 2 (Auto): {row_dict['Discovered_Channel_Name']} (content: {content_score:.3f})"
            )

        # RULE 3: Format Match = False BUT content >= 0.70 → LLM Check
        elif not fmt_match and content_score >= THRES_LLM_CHECK:
            llm_check_queue.append(row_dict)

        # RULE 4: Below 0.70 → Tier 4 (Rejected)
        else:
            row_dict["Final_Tier"] = 4
            row_dict["Final_Status"] = "Tier 4 (Low Content Score)"
            final_results.append(row_dict)

    # 4. Process LLM Check Queue (0.70 - 0.82 range)
    print(
        f"\n🧠 LLM Format Re-check for {len(llm_check_queue)} borderline candidates..."
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_row = {
            executor.submit(run_format_recheck_llm, row_dict, client_format): row_dict
            for row_dict in llm_check_queue
        }

        for future in as_completed(future_to_row):
            row_dict = future_to_row[future]
            name = row_dict["Discovered_Channel_Name"]
            content_score = row_dict["sub_score_content"]

            is_match, reason = future.result()
            row_dict["LLM_Recheck_Reason"] = reason

            if is_match:
                row_dict["Final_Tier"] = 2
                row_dict["Final_Status"] = "Tier 2 (LLM Saved)"
                print(f"  🎯 LLM Saved → Tier 2: {name} (content: {content_score:.3f})")
            else:
                row_dict["Final_Tier"] = 4
                row_dict["Final_Status"] = "Tier 4 (LLM Rejected)"
                print(f"  ❌ LLM Rejected: {name}")

            final_results.append(row_dict)

    # 5. Sort & Save
    df_final = pd.DataFrame(final_results)

    # Sort: Tier 1 by content score DESC, then Tier 2 by content score DESC
    df_final = df_final.sort_values(
        by=["Final_Tier", "sub_score_content"], ascending=[True, False]
    )

    # Keep all columns except Deep_Scan_Data (too large for CSV)
    df_csv = df_final.drop(columns=["Deep_Scan_Data"], errors="ignore")

    # Save
    collection_out = f"{run_tag.upper()}_final_ranked"
    save_dataframe_to_mongo(df_final, collection_out, "Discovered_Channel_ID")

    # timestamp = datetime.now().strftime("%Y%m%d")
    # csv_qual = os.path.join(
    #     OUTPUT_CSV_DIR, f"Final_Report_QUALIFIED_{run_tag}_{timestamp}.csv"
    # )

    # df_qual = df_csv[df_csv["Final_Tier"].isin([1, 2, 3])]
    # df_qual.to_csv(csv_qual, index=False)

    # print(f"\n🏆 DONE! Tier 1-3 Count: {len(df_qual)}")
    # print(f"   📄 CSV: {csv_qual}")


if __name__ == "__main__":
    main()
