# import sys
# import os
# import json
# import pandas as pd
# from pathlib import Path
# from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # --- SETUP PATHS ---
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(BASE_DIR))

# try:
#     from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo
#     from utils.fingerprint_llm_utils import gpt_client
# except ImportError as e:
#     print("❌ Error importing utils.")
#     raise e

# # CONFIG
# OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "ytdlp_scripts", "output", "final_reports")
# os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# # --- CALIBRATED THRESHOLDS (OpenAI Embeddings) ---
# # OpenAI scores are generally higher. 0.75 is the new 0.60.
# THRES_TIER_1 = 0.78  # Direct Competitor (Clone)
# THRES_TIER_2 = 0.68  # Niche Competitor (Related)
# THRES_RE_EVAL = 0.75 # Only Re-Eval if Similarity is EXTREMELY high

# def get_client_constraints(run_tag):
#     try:
#         df = load_collection_as_df(f"{run_tag.upper()}_phase1_fingerprints", {"metadata.run_tag": run_tag})
#         if df.empty: return "General"
#         meta = df.iloc[0].get("metadata", {})
#         constraints = meta.get("client_constraints", {})
#         return constraints.get("format", "General")
#     except:
#         return "General"

# def run_tie_breaker_llm(row_dict, client_format):
#     """
#     The 'Format Auditor': Strict re-evaluation.
#     """
#     name = row_dict.get('Discovered_Channel_Name')
#     deep_json = row_dict.get('Deep_Scan_Data', '[]')
#     try:
#         deep_data = json.loads(deep_json)
#     except:
#         return False, "No data"

#     # Quick dossier (Intro only)
#     dossier = ""
#     for i, vid in enumerate(deep_data[:2]): 
#         transcript = vid.get('caption_tracks', '')[:1500]
#         dossier += f"VIDEO {i+1}: {vid.get('title')}\nTRANSCRIPT START: {transcript}\n---\n"

#     prompt = f"""You are a Strict Format Auditor.
    
#     CONTEXT:
#     The channel "{name}" was REJECTED by the previous filter because it did not look like a "{client_format}".
#     However, it has high topic relevance, so we are double-checking.
    
#     YOUR TASK:
#     Audit the transcripts below. Does this channel ACTUALLY match the format: "{client_format}"?
    
#     CRITICAL RULES:
#     1. **IGNORE TOPIC**: I do not care if they talk about the right subject. If the format is wrong, REJECT IT.
#     2. **STRICT DEFINITIONS**:
#        - If Target = "Podcast": Must have dialogue, interviews, "Welcome to the show", "Guest". 
#          (REJECT if it is a solo monologue, tutorial, or scripted video essay).
#        - If Target = "Tutorial": Must have instructions.
#     3. **BURDEN OF PROOF**: If it looks like a "Video Essay" or "Vlog" pretending to be a Podcast, REJECT IT.
    
#     DOSSIER:
#     {dossier}
    
#     DECISION:
#     Return JSON: {{ "is_match": true/false, "reason": "Brief explanation focusing ONLY on format structure." }}
#     """
    
#     try:
#         response = gpt_client.chat.completions.create(
#             model="gpt-4o-mini",
#             response_format={"type": "json_object"},
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0
#         )
#         res = json.loads(response.choices[0].message.content)
#         return res.get('is_match', False), res.get('reason', 'Tie-breaker decision')
#     except:
#         return False, "LLM Error"

# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python ytdlp_scripts/phase4_final_ranking.py <run_tag>")
#         sys.exit(1)
        
#     run_tag = sys.argv[1]
#     print(f"🚀 PHASE 4: Final Ranking & Report | Tag: {run_tag}")
    
#     client_format = get_client_constraints(run_tag)
#     print(f"📋 Enforcing Format: {client_format}")
#     print(f"📊 Thresholds: T1>={THRES_TIER_1} | T2>={THRES_TIER_2} | ReEval>={THRES_RE_EVAL}")

#     # 1. Load Data
#     try:
#         df_llm = load_collection_as_df(f"{run_tag.upper()}_phase3_step3a")
#         df_emb = load_collection_as_df(f"{run_tag.upper()}_phase3_step3b")
        
#         if df_llm.empty or df_emb.empty:
#             print("❌ Missing scoring data.")
#             return
            
#         print(f"📥 Loaded Data: LLM ({len(df_llm)}), Emb ({len(df_emb)})")
        
#     except Exception as e:
#         print(f"❌ Error loading data: {e}")
#         return

#     # 2. Merge
#     df_merged = pd.merge(
#         df_llm[['Discovered_Channel_ID', 'Discovered_Channel_Name', 'Discovered_Channel_URL', 
#                 'Discovered_Subs', 'Deep_Scan_Data', 
#                 'score_format_match', 'score_format_confidence', 'score_format_reason']],
#         df_emb[['Discovered_Channel_ID', 'score_similarity']],
#         on='Discovered_Channel_ID',
#         how='inner'
#     )
    
#     final_results = []
#     re_eval_count = 0
    
#     # 3. Logic Loop
#     print("⚖️  Applying Logic...")
    
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         future_to_row = {}
        
#         for index, row in df_merged.iterrows():
#             fmt_match = row['score_format_match']
#             sim_score = row['score_similarity']
#             row_dict = row.to_dict()
            
#             tier = 4
#             status = "Tier 4 (Irrelevant)"
            
#             # Case A: Format Approved
#             if fmt_match:
#                 if sim_score >= THRES_TIER_1:
#                     tier = 1
#                     status = "Tier 1 (Direct Competitor)"
#                 elif sim_score >= THRES_TIER_2:
#                     tier = 2
#                     status = "Tier 2 (Niche Competitor)"
#                 else:
#                     tier = 3
#                     status = "Tier 3 (Format Match / Different Topic)"
            
#             # Case B: Re-Eval Zone (Format Failed, but Text is Identical)
#             elif not fmt_match and sim_score >= THRES_RE_EVAL:
#                 re_eval_count += 1
#                 future = executor.submit(run_tie_breaker_llm, row_dict, client_format)
#                 future_to_row[future] = (row_dict, sim_score)
#                 continue 
            
#             # Case C: Trash
#             else:
#                 tier = 4
#                 status = "Tier 4 (Mismatch)"

#             row_dict['Final_Tier'] = tier
#             row_dict['Final_Status'] = status
#             row_dict['Tie_Breaker_Used'] = False
#             row_dict['Tie_Breaker_Reason'] = "N/A"
#             final_results.append(row_dict)

#         # Process Tie-Breakers
#         if re_eval_count > 0:
#             print(f"⚡ Running {re_eval_count} Tie-Breakers...")
            
#         for future in as_completed(future_to_row):
#             row_dict, sim_score = future_to_row[future]
#             name = row_dict['Discovered_Channel_Name']
            
#             is_match, reason = future.result()
#             row_dict['Tie_Breaker_Used'] = True
#             row_dict['Tie_Breaker_Reason'] = reason
            
#             if is_match:
#                 print(f"  ✨ Tie-Breaker SAVED: {name} (Sim: {sim_score:.2f})")
#                 # Re-rank saved channel
#                 if sim_score >= THRES_TIER_1:
#                     row_dict['Final_Tier'] = 1
#                     row_dict['Final_Status'] = "Tier 1 (Saved by Auditor)"
#                 elif sim_score >= THRES_TIER_2:
#                     row_dict['Final_Tier'] = 2
#                     row_dict['Final_Status'] = "Tier 2 (Saved by Auditor)"
#                 else:
#                     row_dict['Final_Tier'] = 3
#                     row_dict['Final_Status'] = "Tier 3 (Saved by Auditor)"
#             else:
#                 print(f"  💀 Tie-Breaker KILLED: {name}")
#                 row_dict['Final_Tier'] = 4
#                 row_dict['Final_Status'] = "Tier 4 (Confirmed Format Mismatch)"
            
#             final_results.append(row_dict)

#     # 4. Final Sort & Clean
#     df_final = pd.DataFrame(final_results)
#     df_final = df_final.sort_values(by=['Final_Tier', 'score_similarity'], ascending=[True, False])
#     df_csv = df_final.drop(columns=['Deep_Scan_Data'], errors='ignore')
    
#     # Reorder Cols
#     desired_order = [
#         "Discovered_Channel_ID", "Discovered_Channel_Name", "Discovered_Channel_URL", "Discovered_Subs",
#         "Final_Tier", "Final_Status", "score_similarity", "score_format_match", "score_format_confidence",
#         "score_format_reason", "Tie_Breaker_Used", "Tie_Breaker_Reason"
#     ]
#     cols = [c for c in desired_order if c in df_csv.columns]
#     remaining = [c for c in df_csv.columns if c not in cols]
#     df_csv = df_csv[cols + remaining]

#     # 5. Save
#     # Mongo
#     collection_out = f"{run_tag.upper()}_final_ranked"
#     save_dataframe_to_mongo(df_final, collection_out, "Discovered_Channel_ID")
    
#     # CSVs
#     timestamp = datetime.now().strftime('%Y%m%d')
#     csv_full = os.path.join(OUTPUT_CSV_DIR, f"Final_Report_FULL_{run_tag}_{timestamp}.csv")
#     df_csv.to_csv(csv_full, index=False)
    
#     df_qual = df_csv[df_csv['Final_Tier'].isin([1, 2, 3])].copy()
#     csv_qual = os.path.join(OUTPUT_CSV_DIR, f"Final_Report_QUALIFIED_{run_tag}_{timestamp}.csv")
#     df_qual.to_csv(csv_qual, index=False)
    
#     print(f"\n🏆 DONE! Processed {len(df_final)} candidates.")
#     print(f"   ✅ Tier 1-3: {len(df_qual)}")
#     print(f"   📄 Qualified CSV: {csv_qual}")

# if __name__ == "__main__":
#     main()


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
THRES_TIER_1 = 0.78  # Direct Clone
THRES_TIER_2 = 0.68  # Niche Zone (Requires Verification)
THRES_RE_EVAL = 0.75 # Only Re-Eval Format if Sim is VERY high

def get_client_constraints(run_tag):
    """Fetch Format AND Intent (Niche) from Phase 1."""
    try:
        df = load_collection_as_df(f"{run_tag.upper()}_phase1_fingerprints", {"metadata.run_tag": run_tag})
        if df.empty: return "General", "General"
        meta = df.iloc[0].get("metadata", {})
        constraints = meta.get("client_constraints", {})
        return constraints.get("format", "General"), constraints.get("intent", "General")
    except:
        return "General", "General"

# --- AUDITOR 1: FORMAT CHECK ---
def run_tie_breaker_format(row_dict, client_format):
    """Checks if the FORMAT is valid (e.g. Is this really a Podcast?)."""
    name = row_dict.get('Discovered_Channel_Name')
    deep_json = row_dict.get('Deep_Scan_Data', '[]')
    try:
        deep_data = json.loads(deep_json)
    except:
        return False, "No data"

    dossier = ""
    for i, vid in enumerate(deep_data[:2]):
        raw_trans = vid.get('caption_tracks')
        transcript = (raw_trans or "")[:1500]
        dossier += f"VIDEO {i+1}: {vid.get('title')}\nTRANSCRIPT START: {transcript}\n---\n"

    prompt = f"""You are a Strict Format Auditor.
    
    CONTEXT:
    The channel "{name}" was REJECTED by the previous filter because it did not look like a "{client_format}".
    However, it has VERY HIGH topic similarity, so we are double-checking.
    
    YOUR TASK:
    Audit the transcripts below. Does this channel ACTUALLY match the format: "{client_format}"?
    
    CRITICAL RULES:
    1. **IGNORE TOPIC**: I do not care if they talk about the right subject. If the format is wrong, REJECT IT.
    2. **STRICT DEFINITIONS**:
       - If Target = "Podcast": Must have dialogue, interviews, "Welcome to the show", "Guest". 
         (REJECT if it is a solo monologue, tutorial, or scripted video essay).
       - If Target = "Tutorial": Must have instructions.
       - If Target = "Documentary": Must be narrative.
    3. **BURDEN OF PROOF**: If it looks like a "Video Essay" or "Vlog" pretending to be a Podcast, REJECT IT.
    
    DOSSIER:
    {dossier}
    
    DECISION:
    Return JSON: {{ "is_match": true/false, "reason": "Brief explanation focusing ONLY on format structure." }}
    """
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        res = json.loads(response.choices[0].message.content)
        return res.get('is_match', False), res.get('reason', 'Audit decision')
    except:
        return False, "LLM Error"

# --- AUDITOR 2: NICHE CHECK (NEW!) ---
def run_niche_check_llm(row_dict, client_intent):
    """Checks if the NICHE matches (e.g. Is this about SaaS/Tech?)."""
    name = row_dict.get('Discovered_Channel_Name')
    deep_json = row_dict.get('Deep_Scan_Data', '[]')
    try:
        deep_data = json.loads(deep_json)
    except:
        return False, "No data"

    # For niche, titles are often enough, but we add transcript snippets for depth
    dossier = ""
    for i, vid in enumerate(deep_data[:3]):
        title = vid.get('title', '')

        raw_trans = vid.get('caption_tracks')
        transcript = (raw_trans or "")[:500]

        dossier += f"- {title} (Context: {transcript}...)\n"

    prompt = f"""You are a Niche Alignment Judge.
    
    TARGET NICHE/INTENT: "{client_intent}"
    CANDIDATE CHANNEL: "{name}"
    
    CONTENT SAMPLES:
    {dossier}
    
    TASK: Does this channel cover the SAME Niche/Topic as the target?
    
    RULES:
    - If Target is "SaaS Marketing" and Candidate is "Crypto News" -> NO.
    - If Target is "B2B Sales" and Candidate is "General Motivation" -> NO.
    - If loosely related but not the same audience -> NO.
    
    JSON: {{ "is_niche_match": true/false, "reason": "..." }}
    """
    
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        res = json.loads(response.choices[0].message.content)
        return res.get('is_niche_match', False), res.get('reason', 'Niche check')
    except:
        return False, "LLM Error"

def main():
    if len(sys.argv) < 2:
        print("Usage: python ytdlp_scripts/phase4_final_ranking.py <run_tag>")
        sys.exit(1)
        
    run_tag = sys.argv[1]
    print(f"🚀 PHASE 4: Final Ranking & Report | Tag: {run_tag}")
    
    client_format, client_intent = get_client_constraints(run_tag)
    print(f"📋 Constraints: Format='{client_format}', Niche='{client_intent}'")
    print(f"📊 Thresholds: T1>={THRES_TIER_1} | T2>={THRES_TIER_2}")

    # 1. Load Data
    try:
        df_llm = load_collection_as_df(f"{run_tag.upper()}_phase3_step3a")
        df_emb = load_collection_as_df(f"{run_tag.upper()}_phase3_step3b")
        if df_llm.empty or df_emb.empty: return
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. Merge
    df_merged = pd.merge(
        df_llm[['Discovered_Channel_ID', 'Discovered_Channel_Name',  
                'Discovered_Subs', 'Deep_Scan_Data', 
                'score_format_match', 'score_format_confidence', 'score_format_reason', 'Discovered_Channel_URL']],
        df_emb[['Discovered_Channel_ID', 'score_similarity']],
        on='Discovered_Channel_ID',
        how='inner'
    )
    
    print(f"🔗 Merged {len(df_merged)} candidates. Starting Audit...")
    
    final_results = []
    
    # 3. Complex Logic Loop
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_row = {}
        
        for index, row in df_merged.iterrows():
            fmt_match = row['score_format_match']
            sim_score = row['score_similarity']
            row_dict = row.to_dict()
            
            # --- DECISION TREE ---
            
            # PATH A: Passed Initial Format Check
            if fmt_match:
                if sim_score >= THRES_TIER_1:
                    # Tier 1 (High Sim + Format Pass)
                    row_dict['Final_Tier'] = 1
                    row_dict['Final_Status'] = "Tier 1 (Direct Competitor)"
                    final_results.append(row_dict)
                
                elif sim_score >= THRES_TIER_2:
                    # Tier 2 Zone (Mid Sim + Format Pass) -> NEEDS NICHE CHECK
                    future = executor.submit(run_niche_check_llm, row_dict, client_intent)
                    future_to_row[future] = ('niche_check', row_dict)
                    
                else:
                    # Tier 3 (Low Sim + Format Pass)
                    row_dict['Final_Tier'] = 3
                    row_dict['Final_Status'] = "Tier 3 (Format Match / Low Sim)"
                    final_results.append(row_dict)
            
            # PATH B: Failed Initial Check (Re-Eval Zone)
            elif not fmt_match and sim_score >= THRES_RE_EVAL:
                # Needs Format Check
                future = executor.submit(run_tie_breaker_format, row_dict, client_format)
                future_to_row[future] = ('format_check', row_dict)
                
            # PATH C: Failed and Low Sim
            else:
                row_dict['Final_Tier'] = 4
                row_dict['Final_Status'] = "Tier 4 (Irrelevant)"
                final_results.append(row_dict)

        # Process Futures
        for future in as_completed(future_to_row):
            check_type, row_dict = future_to_row[future]
            name = row_dict['Discovered_Channel_Name']
            sim_score = row_dict['score_similarity']
            
            # Handle NICHE CHECK (For Tier 2 Candidates)
            if check_type == 'niche_check':
                is_niche, reason = future.result()
                row_dict['Niche_Check_Reason'] = reason
                
                if is_niche:
                    row_dict['Final_Tier'] = 2
                    row_dict['Final_Status'] = "Tier 2 (Niche Verified)"
                    print(f"  ✅ Tier 2 Confirmed: {name}")
                else:
                    row_dict['Final_Tier'] = 3
                    row_dict['Final_Status'] = "Tier 3 (Format Match / Niche Mismatch)"
                    print(f"  ⬇️  Demoted to Tier 3: {name} (Niche Mismatch)")
            
            # Handle FORMAT CHECK (For Re-Evals)
            elif check_type == 'format_check':
                is_fmt, reason = future.result()
                row_dict['Tie_Breaker_Reason'] = reason
                row_dict['Tie_Breaker_Used'] = True
                
                if is_fmt:
                    # Passed Format Audit
                    if sim_score >= THRES_TIER_1:
                        row_dict['Final_Tier'] = 2 # Demote T1 -> T2 because it failed initially
                        row_dict['Final_Status'] = "Tier 2 (Saved by Auditor)"
                        print(f"  ⚠️ Saved (Tier 2): {name}")
                    else:
                        row_dict['Final_Tier'] = 3 # Demote T2 -> T3
                        row_dict['Final_Status'] = "Tier 3 (Saved by Auditor)"
                        print(f"  ⚠️ Saved (Tier 3): {name}")
                else:
                    row_dict['Final_Tier'] = 4
                    row_dict['Final_Status'] = "Tier 4 (Confirmed Format Mismatch)"
                    # print(f"  💀 Killed: {name}")
            
            final_results.append(row_dict)

    # 4. Sort & Save
    df_final = pd.DataFrame(final_results)
    df_final = df_final.sort_values(by=['Final_Tier', 'score_similarity'], ascending=[True, False])
    
    # Clean columns
    cols_to_keep = [
        "Discovered_Channel_ID", "Discovered_Channel_Name", "Discovered_Channel_URL",
        "Final_Tier", "Final_Status", "score_similarity", 
        "score_format_match", "Niche_Check_Reason", "Tie_Breaker_Reason"
    ]
    # Keep existing + extra, remove massive JSON
    df_csv = df_final.drop(columns=['Deep_Scan_Data'], errors='ignore')
    df_csv = df_csv[ [c for c in cols_to_keep if c in df_csv.columns] ]

    # Save
    collection_out = f"{run_tag.upper()}_final_ranked"
    save_dataframe_to_mongo(df_final, collection_out, "Discovered_Channel_ID")
    
    timestamp = datetime.now().strftime('%Y%m%d')
    csv_qual = os.path.join(OUTPUT_CSV_DIR, f"Final_Report_QUALIFIED_{run_tag}_{timestamp}.csv")
    
    df_qual = df_csv[df_csv['Final_Tier'].isin([1, 2, 3])]
    df_qual.to_csv(csv_qual, index=False)
    
    print(f"\n🏆 DONE! Tier 1-3 Count: {len(df_qual)}")
    print(f"   📄 CSV: {csv_qual}")

if __name__ == "__main__":
    main()