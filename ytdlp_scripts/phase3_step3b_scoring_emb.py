import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo
    from utils.fingerprint_llm_utils import get_openai_embedding
except ImportError as e:
    print("❌ Error importing utils.")
    raise e

# --- WEIGHT CONFIGURATION (The SOTA Recipe) ---
WEIGHT_NICHE = 0.5    # Dealbreaker: Identity must match
WEIGHT_INTENT = 0.3  # Vibe Check: Goal must match
WEIGHT_KEYWORDS = 0.2 # Vocabulary: Supports identity

def get_mega_slice(text):
    """Slices text to fit OpenAI Embedding Limit."""
    if not text: return ""
    TOTAL_TARGET = 25000
    if len(text) < TOTAL_TARGET: return text
    chunk_size = TOTAL_TARGET // 3
    head = text[:chunk_size]
    mid_start = len(text) // 2 - (chunk_size // 2)
    mid_end = mid_start + chunk_size
    mid = text[mid_start:mid_end]
    tail = text[-chunk_size:]
    return f"{head} ... {mid} ... {tail}"

def get_seed_fingerprint(run_tag):
    """Load Seed Fingerprint (Niche, Intent, Keywords) from Phase 1"""
    try:
        df_fp = load_collection_as_df(f"{run_tag.upper()}_phase1_fingerprints", {"metadata.run_tag": run_tag})
        if df_fp.empty: return None
        fp_blob = df_fp.iloc[0].to_dict()
        channels_dict = fp_blob.get("channels", {})
        for _, details in channels_dict.items():
            fp = details.get("fingerprint", {})
            return {
                "channel_name": details.get("channel_name", "Unknown"),
                "niche": fp.get("generated_niche", "Unknown"),
                "intent": fp.get("generated_intent", "Unknown"),
                "keywords": fp.get("search_keywords", [])
            }
    except:
        return None

def calculate_text_similarity(text1, text2):
    """Calculates cosine similarity between two strings using OpenAI."""
    if not text1 or not text2: return 0.0
    try:
        vec1 = get_openai_embedding([text1])
        vec2 = get_openai_embedding([text2])
        if vec1 is not None and vec2 is not None:
            return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])
    except:
        pass
    return 0.0

def calculate_list_similarity(list1, list2):
    """Calculates similarity between two lists of keywords (as single string blocks)."""
    if not list1 or not list2: return 0.0
    str1 = ", ".join(list1)
    str2 = ", ".join(list2)
    return calculate_text_similarity(str1, str2)

def get_seed_content_vector(run_tag):
    """Calculates Seed Vector from Phase 1 Deep Data (Raw Content)"""
    df = load_collection_as_df(f"{run_tag.upper()}_phase1", {"run_tag": run_tag})
    if df.empty: return None
    texts = []
    for _, row in df.iterrows():
        raw_text = row.get('caption_tracks', '') or row.get('description', '')
        sliced = get_mega_slice(raw_text)
        if sliced: texts.append(sliced)
    if not texts: return None
    print(f"   Calculating Seed Content Vector from {len(texts)} videos...")
    return get_openai_embedding(texts)

def main():
    if len(sys.argv) < 2:
        print("Usage: python ytdlp_scripts/phase3_step3b_scoring_emb.py <run_tag>")
        sys.exit(1)
        
    run_tag = sys.argv[1]
    print(f"🚀 PHASE 3 STEP C-2: SOTA Weighted Similarity | Tag: {run_tag}")
    
    # 1. Load Seed Data
    seed_fp = get_seed_fingerprint(run_tag)
    if not seed_fp:
        print("❌ Critical Error: Could not load Seed Fingerprint.")
        return

    seed_content_vector = get_seed_content_vector(run_tag)
    
    print(f"\n📌 SEED: {seed_fp['channel_name']}")
    print(f"   Niche: {seed_fp['niche']}")
    print(f"   Intent: {seed_fp['intent'][:50]}...")

    # 2. Load Candidates
    collection_in = f"{run_tag.upper()}_phase3_step2"
    try:
        df_candidates = load_collection_as_df(collection_in)
    except:
        print(f"❌ Collection {collection_in} not found.")
        return
    
    if df_candidates.empty:
        print("❌ No candidates found.")
        return

    scored_results = []
    print(f"\n🧠 Scoring {len(df_candidates)} candidates...\n")

    for index, row in df_candidates.iterrows():
        row_dict = row.to_dict()
        name = row_dict.get('Discovered_Channel_Name', 'Unknown')
        
        # --- A. Extract Candidate Data ---
        cand_niche = row_dict.get('generated_niche', '')
        cand_intent = row_dict.get('generated_intent', '')
        cand_kws_raw = row_dict.get('generated_keywords', '[]')
        try:
            cand_keywords = json.loads(cand_kws_raw) if isinstance(cand_kws_raw, str) else cand_kws_raw
        except:
            cand_keywords = []

        print(f"[{index+1}/{len(df_candidates)}] 🎯 {name}")

        # --- B. Calculate Component Scores ---
        
        # 1. Niche Score (Identity)
        score_niche = calculate_text_similarity(seed_fp['niche'], cand_niche)
        
        # 2. Intent Score (Goal)
        score_intent = calculate_text_similarity(seed_fp['intent'], cand_intent)
        
        # 3. Keyword Score (Vocabulary)
        score_keywords = calculate_list_similarity(seed_fp['keywords'], cand_keywords)
        
        # 4. Content Score (Raw Text - The "Blender")
        score_content = 0.0
        deep_json = row_dict.get('Deep_Scan_Data', '[]')
        try:
            deep_data = json.loads(deep_json)
            cand_texts = []
            for v in deep_data:
                txt = v.get('caption_tracks', '') or v.get('description', '')
                sliced = get_mega_slice(txt)
                if sliced: cand_texts.append(sliced)
            
            if cand_texts and seed_content_vector is not None:
                cand_vec = get_openai_embedding(cand_texts)
                if cand_vec is not None:
                    score_content = float(cosine_similarity(seed_content_vector.reshape(1, -1), cand_vec.reshape(1, -1))[0][0])
        except Exception as e:
            print(f"  ⚠️ Content Embedding Error: {e}")

        # --- C. Calculate Weighted Final Score ---
        final_score = (
            (score_niche * WEIGHT_NICHE) +
            (score_intent * WEIGHT_INTENT) +
            (score_keywords * WEIGHT_KEYWORDS)
        )

        print(f"   Scores -> Niche: {score_niche:.2f} | Intent: {score_intent:.2f} | KWs: {score_keywords:.2f} | Content: {score_content:.2f}")
        print(f"   🏆 FINAL WEIGHTED SCORE: {final_score:.3f}")

        # --- D. Save All Metrics ---
        row_dict['score_similarity'] = final_score # This is what Phase 4 uses
        row_dict['sub_score_niche'] = score_niche
        row_dict['sub_score_intent'] = score_intent
        row_dict['sub_score_keywords'] = score_keywords
        row_dict['sub_score_content'] = score_content
        
        scored_results.append(row_dict)

    # Save
    collection_out = f"{run_tag.upper()}_phase3_step3b"
    save_dataframe_to_mongo(pd.DataFrame(scored_results), collection_out, "Discovered_Channel_ID")
    print(f"\n💾 Saved to '{collection_out}'")
    print("✅ Ready for Step D (Final Ranking).")

if __name__ == "__main__":
    main()