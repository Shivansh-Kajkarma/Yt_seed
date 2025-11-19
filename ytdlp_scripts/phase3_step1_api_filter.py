import sys
import os
import json
import time
import pandas as pd
from datetime import datetime
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

# Config
MODEL_NAME = "gpt-4o-mini" # Cheap & Fast for filtering
MAX_WORKERS = 5 # Parallel LLM calls

def get_client_constraints(run_tag):
    """
    Fetches the Format/Intent constraints from Phase 1 Fingerprints.
    """
    try:
        df = load_collection_as_df(f"{run_tag.upper()}_phase1_fingerprints", {"metadata.run_tag": run_tag})
        if df.empty: return "General", "General"
        
        # Extract metadata from the first record
        meta = df.iloc[0].get("metadata", {})
        constraints = meta.get("client_constraints", {})
        
        return constraints.get("format", "General"), constraints.get("intent", "General")
    except Exception as e:
        print(f"⚠️ Could not load constraints: {e}")
        return "General", "General"

def check_relevance_gpt_mini(channel_data, client_format, client_intent):
    """
    Uses API Metadata (Titles/Desc) to filter out obvious non-matches.
    """
    name = channel_data.get('Discovered_Channel_Name', 'Unknown')
    desc = channel_data.get('Discovered_Channel_Description', '')
    
    # Parse the JSON string of videos from Phase 2
    videos_json = channel_data.get('Discovered_Videos_JSON', '[]')
    try:
        videos = json.loads(videos_json)
    except:
        videos = []
        
    # Get last 5 titles for context
    titles = [v.get('title', '') for v in videos]
    titles_blob = "\n- ".join(titles)

    # print("\n\n titles_blob:", titles_blob)

    prompt = f"""You are a Content Filter.
    
    User is looking for channels matching this profile:
    FORMAT: "{client_format}"
    INTENT: "{client_intent}"
    
    Evaluate this Candidate:
    NAME: {name}
    BIO: {desc[:300]}...
    RECENT VIDEOS:
    - {titles_blob}
    
    TASK: Return JSON.
    "keep": true/false
    "reason": "short reason"
    
    RULES:
    1. If the content is OBVIOUSLY irrelevant INTENT (e.g., Gaming, Music, Vlog, News Clips when user wants Podcast), set "keep": false.
    2. If it looks promising OR ambiguous, set "keep": true (Pass it to the next deep scan step).
    3. Be lenient. Only drop confirmed mismatches.
    """

    try:
        response = gpt_client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠️ LLM Error for {name}: {e}")
        return {"keep": True, "reason": "Error fallback"}

def main():
    if len(sys.argv) < 2:
        print("Usage: python ytdlp_scripts/phase3_step1_api_filter.py <run_tag>")
        sys.exit(1)
        
    run_tag = sys.argv[1]
    print(f"🚀 PHASE 3 STEP A: API Metadata Filter | Tag: {run_tag}")
    
    # 1. Get Constraints
    fmt, intent = get_client_constraints(run_tag)
    print(f"📋 Constraints: Format='{fmt}', Intent='{intent}'")

    # 2. Load Phase 2 Candidates
    collection_in = f"{run_tag.upper()}_phase2"
    try:
        df_candidates = load_collection_as_df(collection_in)
        if df_candidates.empty:
            print("❌ No candidates found in Phase 2.")
            return
        print(f"📥 Loaded {len(df_candidates)} candidates from Phase 2.")
    except Exception as e:
        print(f"❌ Error loading candidates: {e}")
        return

    # 3. Run Parallel Filter
    filtered_candidates = []
    kept_count = 0
    dropped_count = 0
    
    print(f"🧠 Filtering with {MODEL_NAME} (Workers: {MAX_WORKERS})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map futures to rows
        future_to_row = {
            executor.submit(check_relevance_gpt_mini, row.to_dict(), fmt, intent): row 
            for _, row in df_candidates.iterrows()
        }
        
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            name = row['Discovered_Channel_Name']
            
            try:
                result = future.result()
                is_keep = result.get('keep', True)
                reason = result.get('reason', 'N/A')
                
                if is_keep:
                    # Add metadata for next steps
                    row_dict = row.to_dict()
                    row_dict['step1_filter_reason'] = reason
                    row_dict['step1_passed'] = True
                    filtered_candidates.append(row_dict)
                    print(f"  ✅ KEEP: {name}")
                    kept_count += 1
                else:
                    print(f"  🗑️  DROP: {name} ({reason})")
                    dropped_count += 1
                    
            except Exception as e:
                print(f"  ⚠️ Error processing {name}: {e}")
                # Keep on error to be safe
                filtered_candidates.append(row.to_dict())
                kept_count += 1

    # 4. Save Results
    if not filtered_candidates:
        print("❌ All candidates were filtered out!")
        return

    df_out = pd.DataFrame(filtered_candidates)
    collection_out = f"{run_tag.upper()}_phase3_step1"
    
    try:
        save_dataframe_to_mongo(
            df_out, 
            collection_out, 
            unique_key_column="Discovered_Channel_ID"
        )
        print(f"\n💾 Saved {len(df_out)} qualified candidates to '{collection_out}'")
        print(f"📊 Summary: {kept_count} Kept | {dropped_count} Dropped")
        print("✅ Ready for Step B (Deep Scan).")
        
    except Exception as e:
        print(f"❌ Save Error: {e}")

if __name__ == "__main__":
    main()