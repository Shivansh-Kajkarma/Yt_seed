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
    from utils.fingerprint_llm_utils import gpt_client
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

# --- NEW: SMART SLICER (Phase 1 Logic) ---
def get_smart_slice(text, chunk_size=1500):
    """
    Takes Start, Middle, and End to give the LLM a full picture of the format.
    """
    if not text: return "[NO TRANSCRIPT]"
    if len(text) < chunk_size * 3: return text # Short video? Return all.
    
    head = text[:chunk_size]
    
    mid_start = len(text) // 2
    mid = text[mid_start : mid_start + chunk_size]
    
    tail = text[-chunk_size:]
    
    return f"[START]\n{head}\n...\n[MIDDLE]\n{mid}\n...\n[END]\n{tail}"

def verify_format_llm(candidate_data, client_format):
    """
    Strictly checks if the content matches the Client Format using Deep Data.
    Feeds 3 Videos x (Start + Mid + End).
    """
    name = candidate_data.get('Discovered_Channel_Name')
    deep_json = candidate_data.get('Deep_Scan_Data', '[]')
    try:
        deep_data = json.loads(deep_json)
    except:
        deep_data = []

    if not deep_data:
        return {"is_format_match": False, "reason": "No scan data", "confidence": 0.0}

    # Create a Dossier of the 3 scanned videos
    dossier = ""
    for i, vid in enumerate(deep_data):
        title = vid.get('title', 'Unknown')
        duration = vid.get('duration', 0)
        has_chapters = vid.get('has_chapters', False)
        
        # --- USE SMART SLICE ---
        raw_transcript = vid.get('caption_tracks', '')
        transcript_sample = get_smart_slice(raw_transcript)
        
        dossier += f"""
        === VIDEO {i+1} ===
        TITLE: {title}
        DURATION: {duration}s
        HAS CHAPTERS: {has_chapters}
        TRANSCRIPT SAMPLES:
        {transcript_sample}
        ===================
        """

    # print(dossier)

    prompt = f"""You are a Strict Format Validator.
    
    CLIENT REQUIREMENT: "{client_format}"
    
    TASK: Analyze the 3 video samples below. Does this channel STRICTLY produce "{client_format}" content?
    
    RULES:
    1. **Podcast**: Look for dialogue markers (>>), "Welcome back", "Guest", "Episode". 
       - Check [MIDDLE]: Is it a conversation or a monologue?
       - Check [END]: Do they sign off like a show?
    2. **Documentary**: Look for narrative flow, "The story of...", music cues.
    3. **Tutorial**: Look for "Step 1", "How to", "In this guide".
    
    DOSSIER:
    {dossier}
    
    OUTPUT JSON:
    {{
        "is_format_match": true/false,
        "confidence": 0.0 to 1.0,
        "reason": "Brief explanation citing specific video evidence"
    }}
    """
    
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"is_format_match": False, "reason": f"LLM Error: {e}", "confidence": 0.0}

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