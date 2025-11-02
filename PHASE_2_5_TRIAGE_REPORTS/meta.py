import pandas as pd
from pathlib import Path
import os
import sys

run_tag = "moon1"
cols_to_show = [
            'Discovered_Channel_Name', 
            'Discovered_Subs',
            'Score_Emb_Combined_AvgVec', # My main method
            'Score_Emb_Hybrid_Titles', # Your hybrid function
            'Score_Emb_Matrix_Avg'  # Your new idea
        ]
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
timestamp = "20251102_134720"
TRIAGE_REPORT_DIR = BASE_DIR / "PHASE_2_5_TRIAGE_REPORTS"
FINAL_TRIAGE_FILE_META = TRIAGE_REPORT_DIR / f"phase2_5_triage_meta_{run_tag}_{timestamp}.csv"

df_report = pd.read_csv(r"/home/rareboy/Internship/Kajkarma/PHASE_2_5_TRIAGE_REPORTS/phase2_5_triage_report_moon_20251102_134720.csv")
# df1 = 
# df_report[df_report["Discovered_Video_Count"]<2500][cols_to_show+["Discovered_Video_Count","Discovered_Channel_URL"]].to_csv(FINAL_TRIAGE_FILE_META, index=False, encoding='utf-8-sig')

# ✅ FIX: Import from umap.umap_ instead of just umap
import umap.umap_ as umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
# ✅ FIX: Use df_report instead of undefined 'df'
embeds = df_report[['Score_Emb_Combined_AvgVec', 'Score_Emb_Matrix_Avg']].values
projection = reducer.fit_transform(embeds)

plt.scatter(projection[:,0], projection[:,1], c=df_report['Score_Emb_Combined_AvgVec'], cmap='viridis')
plt.title("Semantic Competitor Map - MagnatesMedia")
plt.colorbar(label='Score_Emb_Combined_AvgVec')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()


# import pandas as pd
# import numpy as np
# import os
# import sys
# import json
# from pathlib import Path
# import time
# from datetime import datetime
# import umap.umap_ as umap
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# # --- Make sure utils are importable ---
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(BASE_DIR))

# try:
#     from utils.fingerprint_llm_utils import (
#         embedding_model,
#         preprocess_text_for_llm
#     )
#     print("✅ Successfully imported local embedding model.")
# except ImportError as e:
#     print(f"❌ CRITICAL ERROR: Could not import 'utils' folder or functions.")
#     sys.exit(1)

# if not embedding_model:
#     print("❌ CRITICAL ERROR: The 'embedding_model' from utils is None.")
#     sys.exit(1)

# # --- Helper Functions ---

# def find_latest_file_path(directory: Path, prefix: str, suffix: str) -> Path | None:
#     try:
#         latest_file = max(
#             directory.glob(f"{prefix}*{suffix}"),
#             key=os.path.getctime
#         )
#         return latest_file
#     except ValueError:
#         return None

# def get_vector_from_texts(texts: list[str]) -> np.ndarray | None:
#     """
#     Takes a list of text strings, cleans them, gets embeddings,
#     and returns the single averaged vector.
#     """
#     if not texts:
#         return None
#     cleaned_texts = [preprocess_text_for_llm(text) for text in texts if text]
#     if not cleaned_texts:
#         return None
#     try:
#         embeddings = embedding_model.encode(cleaned_texts)
#         avg_vector = np.mean(embeddings, axis=0)
#         return avg_vector
#     except Exception as e:
#         print(f"  ⚠️ Error encoding texts: {e}")
#         return None

# # --- Main Plotting Function ---
# def main(run_tag: str):
#     print(f"--- 🚀 Starting Semantic Map Plotting for '{run_tag}' ---")
    
#     # --- Define file paths ---
#     FINGERPRINT_DIR = BASE_DIR / "FINGERPRINTS_ONESHOT"
#     SEED_VIDEOS_DIR = BASE_DIR / "PHASE_1_OUTPUTS"
#     TRIAGE_REPORT_DIR = BASE_DIR / "PHASE_2_5_TRIAGE_REPORTS"

#     # --- A. Load Seed Channel Data ---
#     print(f"\n[STEP 1/4] Loading Seed Channel '{run_tag}' Data...")
#     SEED_FINGERPRINT_FILE = find_latest_file_path(FINGERPRINT_DIR, f"fingerprints_oneshot_{run_tag}", ".json")
#     SEED_VIDEOS_FILE = find_latest_file_path(SEED_VIDEOS_DIR, f"sample_videos_{run_tag}", ".csv")

#     if not SEED_FINGERPRINT_FILE or not SEED_VIDEOS_FILE:
#         print(f"❌ ERROR: Missing Phase 1 files for tag '{run_tag}'.")
#         return
        
#     try:
#         with open(SEED_FINGERPRINT_FILE, 'r') as f:
#             seed_data = json.load(f)
#         first_channel_key = next(iter(seed_data["channels"]))
#         seed_channel_desc = seed_data["channels"][first_channel_key].get("fingerprint", {}).get("profile", {}).get("description", "")
#         seed_name = seed_data["channels"][first_channel_key].get("channel_name", run_tag)
        
#         df_seed_videos = pd.read_csv(SEED_VIDEOS_FILE)
#         seed_video_titles = df_seed_videos['title'].dropna().tolist()
#         seed_video_descs = df_seed_videos['description'].dropna().tolist()
        
#     except Exception as e:
#         print(f"❌ ERROR: Failed to load or parse seed files: {e}")
#         return

#     # --- B. Load Triage Report (from Phase 2.5) ---
#     print(f"\n[STEP 2/4] Loading Triage Report...")
#     TRIAGE_REPORT_FILE = find_latest_file_path(TRIAGE_REPORT_DIR, f"phase2_5_triage_report_{run_tag}", ".csv")
    
#     if not TRIAGE_REPORT_FILE:
#         print(f"❌ ERROR: No Phase 2.5 triage report file found for tag '{run_tag}'.")
#         print("   Please run Phase 2.5 first.")
#         return

#     print(f"  Using Triage File: {TRIAGE_REPORT_FILE.name}")
#     try:
#         df_report = pd.read_csv(TRIAGE_REPORT_FILE)
#         df_report['Discovered_Channel_Description'] = df_report['Discovered_Channel_Description'].fillna("")
#         df_report['Discovered_Videos_JSON'] = df_report['Discovered_Videos_JSON'].fillna("[]")
#         print(f"✅ Found {len(df_report)} candidates to plot.")
#     except Exception as e:
#         print(f"❌ ERROR: Failed to load triage report: {e}")
#         return

#     # --- C. Re-generate ALL Embedding Vectors ---
#     print(f"\n[STEP 3/4] Generating embedding vectors for all channels...")
#     all_vectors = []
#     labels = []
#     scores = []
    
#     # 1. Add Seed Channel
#     seed_all_texts = [seed_channel_desc] + seed_video_titles + [desc[:300] for desc in seed_video_descs]
#     seed_vec = get_vector_from_texts(seed_all_texts)
#     all_vectors.append(seed_vec)
#     labels.append(seed_name) # This is our seed
#     scores.append(1.0) # Seed has a score of 1.0 (max)
    
#     # 2. Add Candidate Channels
#     for index, row in df_report.iterrows():
#         try:
#             cand_desc = row.get('Discovered_Channel_Description')
#             cand_videos_json = row.get('Discovered_Videos_JSON')
#             cand_videos_list = json.loads(cand_videos_json)
            
#             if not cand_videos_list:
#                 continue
                
#             df_cand_videos = pd.DataFrame(cand_videos_list)
#             cand_video_titles = df_cand_videos['title'].dropna().tolist()
#             cand_video_descs = df_cand_videos['description'].dropna().tolist()

#             cand_all_texts = [cand_desc] + cand_video_titles + [desc[:300] for desc in cand_video_descs]
#             cand_vec = get_vector_from_texts(cand_all_texts)
            
#             if cand_vec is not None:
#                 all_vectors.append(cand_vec)
#                 labels.append(row.get('Discovered_Channel_Name'))
#                 scores.append(row.get('Score_Emb_Combined_AvgVec'))
                
#         except Exception as e:
#             print(f"  ⚠️ Skipping {row.get('Discovered_Channel_Name')}: {e}")
            
#     print(f"✅ Generated {len(all_vectors)} total vectors.")
    
#     # Convert to a NumPy array for UMAP
#     # This is your HIGH-DIMENSIONAL data (e.g., 200 channels x 384 dimensions)
#     high_dim_embeddings = np.array(all_vectors)
    
#     # --- D. Run UMAP and Plot ---
#     print(f"\n[STEP 4/4] Running UMAP and plotting...")
    
#     reducer = umap.UMAP(
#         n_neighbors=15,  # How many neighbors to look at. Default 15
#         min_dist=0.1,    # How clumped points are. Default 0.1
#         metric='cosine'  # Use cosine similarity, since we are using text
#     )
    
#     # This takes your (200, 384) data and turns it into (200, 2) data
#     projection = reducer.fit_transform(high_dim_embeddings)

#     plt.figure(figsize=(12, 8))
#     scatter = plt.scatter(
#         projection[:,0], 
#         projection[:,1], 
#         c=scores, # Color by the score
#         cmap='viridis',
#         s=15 # Size of dots
#     )
    
#     # Highlight the Seed Channel
#     plt.scatter(
#         projection[0,0], # X of the first point (seed)
#         projection[0,1], # Y of the first point (seed)
#         c='red',         # Make it red
#         s=100,           # Make it bigger
#         marker='*',      # Make it a star
#         label=f'Seed: {seed_name}'
#     )
    
#     plt.title(f"Semantic Competitor Map - Seed: {seed_name}")
#     plt.colorbar(scatter, label='Score_Emb_Combined_AvgVec')
#     plt.xlabel('UMAP Dimension 1')
#     plt.ylabel('UMAP Dimension 2')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.show()


# if __name__ == "__main__":

#     run_tag_arg = "moon"
        
#     # run_tag_arg = sys.argv[1].lower().strip()
#     main(run_tag_arg)