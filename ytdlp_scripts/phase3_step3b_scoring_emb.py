# import sys
# import os
# import json
# import pandas as pd
# from pathlib import Path
# from sklearn.metrics.pairwise import cosine_similarity

# # --- SETUP PATHS ---
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(BASE_DIR))

# try:
#     from utils.mongo_utils import load_collection_as_df, save_dataframe_to_mongo
#     from utils.fingerprint_llm_utils import get_vector_from_texts
# except ImportError as e:
#     print("❌ Error importing utils.")
#     raise e

# def get_compressed_slice(text):
#     """
#     Slices text to fit into standard Embedding Models (approx 512 tokens / 2000 chars).
#     Strategy: Take Intro (Context) + Middle (Depth).
#     """
#     if not text: return ""
#     if len(text) < 2000: return text
    
#     # Take first 1000 chars (Intro)
#     head = text[:1000]
#     # Take 1000 chars from the middle
#     mid_start = len(text) // 2
#     mid = text[mid_start : mid_start + 1000]
    
#     return f"{head} ... {mid}"

# def get_seed_vector(run_tag):
#     """Calculates Seed Vector from Phase 1 Deep Data"""
#     # Load Phase 1 data (which has transcripts)
#     df = load_collection_as_df(f"{run_tag.upper()}_phase1", {"run_tag": run_tag})
#     if df.empty: return None
    
#     texts = []
#     for _, row in df.iterrows():
#         # Use the compressed slice strategy for Seed too!
#         trans = get_compressed_slice(row.get('caption_tracks', ''))
#         if trans: texts.append(trans)
        
#     if not texts: return None
#     return get_vector_from_texts(texts)

# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python ytdlp_scripts/phase3_step3b_scoring_emb.py <run_tag>")
#         sys.exit(1)
        
#     run_tag = sys.argv[1]
#     print(f"🚀 PHASE 3 STEP C-2: Embedding Similarity | Tag: {run_tag}")
    
#     # 1. Calculate Seed Vector
#     seed_vector = get_seed_vector(run_tag)
#     if seed_vector is None:
#         print("❌ Could not calculate Seed Vector. (Did Phase 1 save deep data?)")
#         return
#     print("✅ Seed Vector Calculated.")

#     # 2. Load Candidates from Step 2 (Deep Scan)
#     collection_in = f"{run_tag.upper()}_phase3_step2" # <--- CORRECTED INPUT
#     df_candidates = load_collection_as_df(collection_in)
    
#     if df_candidates.empty:
#         print(f"❌ No candidates found in {collection_in}")
#         return

#     scored_results = []
#     print(f"🧠 Calculating Embeddings for {len(df_candidates)} candidates...")

#     for index, row in df_candidates.iterrows():
#         row_dict = row.to_dict()
#         name = row_dict.get('Discovered_Channel_Name', 'Unknown')
        
#         # Extract texts from Deep Scan Data
#         deep_json = row_dict.get('Deep_Scan_Data', '[]')
#         try:
#             deep_data = json.loads(deep_json)
            
#             # Apply the "Compressed Slice" to candidate videos
#             cand_texts = [
#                 get_compressed_slice(v.get('caption_tracks', '')) 
#                 for v in deep_data 
#                 if v.get('caption_tracks')
#             ]
            
#             if not cand_texts: 
#                 # Fallback to description if no transcript
#                 cand_texts = [v.get('description', '') for v in deep_data]
            
#             cand_vector = get_vector_from_texts(cand_texts)
            
#             if cand_vector is not None:
#                 similarity = cosine_similarity(seed_vector.reshape(1, -1), cand_vector.reshape(1, -1))[0][0]
#                 row_dict['score_similarity'] = float(similarity)
#             else:
#                 row_dict['score_similarity'] = 0.0
                
#         except Exception as e:
#             print(f"  ⚠️ Fail {name}: {e}")
#             row_dict['score_similarity'] = 0.0
            
#         print(f"  📊 {name[:20]}... | Sim: {row_dict['score_similarity']:.3f}")
#         scored_results.append(row_dict)

#     # Save to Step 3b collection
#     collection_out = f"{run_tag.upper()}_phase3_step3b"
#     save_dataframe_to_mongo(pd.DataFrame(scored_results), collection_out, "Discovered_Channel_ID")
#     print(f"\n💾 Saved to '{collection_out}'")
#     print("✅ Ready for Step D (Final Ranking).")

# if __name__ == "__main__":
#     main()


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

def get_mega_slice(text):
    """
    Slices text to fit OpenAI Embedding Limit (~8191 tokens / ~32k chars).
    Strategy: Take 3 massive chunks (Start, Mid, End) to capture full context.
    """
    if not text: return ""
    
    # Safety buffer: Target ~25k chars total to be safe under 8k tokens
    TOTAL_TARGET = 25000
    
    if len(text) < TOTAL_TARGET: 
        return text
    
    # Calculate chunks (approx 8k each)
    chunk_size = TOTAL_TARGET // 3
    
    # 1. Head (Intro/Hook)
    head = text[:chunk_size]
    
    # 2. Middle (Deep Dive)
    mid_start = len(text) // 2 - (chunk_size // 2)
    mid_end = mid_start + chunk_size
    mid = text[mid_start:mid_end]
    
    # 3. Tail (Conclusion/Outro)
    tail = text[-chunk_size:]
    
    # Combine
    return f"{head} ... [CONTENT SKIPPED] ... {mid} ... [CONTENT SKIPPED] ... {tail}"

def get_seed_vector(run_tag):
    """Calculates Seed Vector from Phase 1 Deep Data using OpenAI"""
    # Load Phase 1 data (which has transcripts)
    df = load_collection_as_df(f"{run_tag.upper()}_phase1", {"run_tag": run_tag})
    if df.empty: return None
    
    texts = []
    for _, row in df.iterrows():
        # Get transcript
        raw_text = row.get('caption_tracks', '')
        if not raw_text:
            raw_text = row.get('description', '')
        
        # Apply Mega Slice
        sliced_text = get_mega_slice(raw_text)
        
        if sliced_text: 
            texts.append(sliced_text)
        
    if not texts: return None
    
    print(f"   Calculating Seed Vector from {len(texts)} videos (Mega-Sliced)...")
    return get_openai_embedding(texts)

def main():
    if len(sys.argv) < 2:
        print("Usage: python ytdlp_scripts/phase3_step3b_scoring_emb.py <run_tag>")
        sys.exit(1)
        
    run_tag = sys.argv[1]
    print(f"🚀 PHASE 3 STEP C-2: OpenAI Embedding Similarity | Tag: {run_tag}")
    
    # 1. Calculate Seed Vector (OpenAI)
    seed_vector = get_seed_vector(run_tag)
    if seed_vector is None:
        print("❌ Could not calculate Seed Vector. (Did Phase 1 save deep data?)")
        return
    print("✅ Seed Vector Calculated (OpenAI).")

    # 2. Load Candidates from Step 2 (Deep Scan)
    collection_in = f"{run_tag.upper()}_phase3_step2"
    try:
        df_candidates = load_collection_as_df(collection_in)
    except:
        print(f"❌ Collection {collection_in} not found.")
        return
    
    if df_candidates.empty:
        print(f"❌ No candidates found in {collection_in}")
        return

    scored_results = []
    print(f"🧠 Calculating Embeddings for {len(df_candidates)} candidates...")

    for index, row in df_candidates.iterrows():
        row_dict = row.to_dict()
        name = row_dict.get('Discovered_Channel_Name', 'Unknown')
        
        deep_json = row_dict.get('Deep_Scan_Data', '[]')
        try:
            deep_data = json.loads(deep_json)
            
            # 3. Collect SLICED TEXTS for this candidate
            cand_texts = []
            for v in deep_data:
                # Priority: Transcript > Description
                text = v.get('caption_tracks', '')
                if not text:
                    text = v.get('description', '')
                
                # Apply Mega Slice
                processed_text = get_mega_slice(text)
                if processed_text:
                    cand_texts.append(processed_text)
            
            # 4. Get Vector (Avg of 3 videos)
            if cand_texts:
                cand_vector = get_openai_embedding(cand_texts)
            else:
                cand_vector = None
            
            if cand_vector is not None:
                # Calculate Cosine Similarity
                similarity = cosine_similarity(seed_vector.reshape(1, -1), cand_vector.reshape(1, -1))[0][0]
                row_dict['score_similarity'] = float(similarity)
            else:
                row_dict['score_similarity'] = 0.0
                
        except Exception as e:
            print(f"  ⚠️ Fail {name}: {e}")
            row_dict['score_similarity'] = 0.0
            
        print(f"  📊 {name[:20]}... | Sim: {row_dict['score_similarity']:.3f}")
        scored_results.append(row_dict)

    # Save to Step 3b collection
    collection_out = f"{run_tag.upper()}_phase3_step3b"
    save_dataframe_to_mongo(pd.DataFrame(scored_results), collection_out, "Discovered_Channel_ID")
    print(f"\n💾 Saved to '{collection_out}'")
    print("✅ Ready for Step D (Final Ranking).")

if __name__ == "__main__":
    main()