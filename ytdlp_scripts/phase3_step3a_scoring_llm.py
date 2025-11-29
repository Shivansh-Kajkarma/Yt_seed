import sys
import os
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo
    from utils.fingerprint_llm_utils import gpt_client, verify_format_llm
except ImportError as e:
    print("❌ Error importing utils.")
    raise e

# CONFIG
MAX_WORKERS = 5 
def get_client_constraints(run_tag):
    try:
        df = load_collection_as_df(f"{run_tag.upper()}_phase1_fingerprints", {"metadata.run_tag": run_tag})
        if df.empty: return "General"
        meta = df.iloc[0].get("metadata", {})
        constraints = meta.get("client_constraints", {})
        return constraints.get("format", "General")
    except:
        return "General"
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python ytdlp_scripts/phase3_step3a_scoring_llm.py <run_tag>")
        sys.exit(1)
        
    run_tag = sys.argv[1]
    print(f"🚀 PHASE 3 STEP C-1: LLM Format Check | Tag: {run_tag}")
    
    client_format = get_client_constraints(run_tag)
    print(f"📋 Constraint: Must be '{client_format}'")
    
    # Load from Step 2 (Deep Scan)
    collection_in = f"{run_tag.upper()}_phase3_step2"
    try:
        df_candidates = load_collection_as_df(collection_in)
    except:
        print(f"❌ Collection {collection_in} not found.")
        return
    
    if df_candidates.empty:
        print("❌ No candidates found from Step B.")
        return

    scored_results = []
    
    print(f"🧠 Scoring {len(df_candidates)} candidates with LLM (Smart Slicing)...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {
            executor.submit(verify_format_llm, row.to_dict(), client_format): row 
            for _, row in df_candidates.iterrows()
        }

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            row_dict = row.to_dict()
            name = row_dict['Discovered_Channel_Name']
            
            try:
                llm_result = future.result()
                row_dict['score_format_match'] = llm_result['is_format_match']
                row_dict['score_format_confidence'] = llm_result['confidence']
                row_dict['score_format_reason'] = llm_result['reason']
                
                icon = "✅" if llm_result['is_format_match'] else "❌"
                print(f"  {icon} {name[:30]}... | Conf: {llm_result['confidence']}")
                
            except Exception as e:
                print(f"  ⚠️ LLM Fail {name}: {e}")
                row_dict['score_format_match'] = False
                row_dict['score_format_confidence'] = 0.0
            
            scored_results.append(row_dict)

    # Save to Step 3a collection
    collection_out = f"{run_tag.upper()}_phase3_step3a"
    save_dataframe_to_mongo(pd.DataFrame(scored_results), collection_out, "Discovered_Channel_ID")
    print(f"\n💾 Saved to '{collection_out}'")

if __name__ == "__main__":
    main()