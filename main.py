# from fastapi import FastAPI
# from celery_worker import run_phase_pipeline, celery_app
# from celery.result import AsyncResult
# import pandas as pd  # <-- ADDED
# from typing import List # <-- ADDED

# # --- ADDED: We need these two functions to get the data ---
# from utils.mongo_utils import load_collection_as_df
# from utils.pipeline_wrapper import normalize_run_tag

# app = FastAPI(title="Kajkarma AI Pipeline API")

# @app.on_event("startup")
# async def startup_event():
#     print(f"🚀 Celery Broker: {celery_app.conf.broker_url}")
#     print(f"🚀 Celery Backend: {celery_app.conf.result_backend}")

# @app.post("/start_pipeline")
# def start_pipeline(sheet_url: str):
    
#     #  call the task in "Loader" mode.
#     # Arg 1: sheet_url = The URL from the user
#     # Arg 2: seed_dict = None
#     task = run_phase_pipeline.delay(sheet_url, None) 
    
#     return {"task_id": task.id, "message": "Pipeline 'Loader' task started."}

# @app.get("/status/{task_id}")
# def get_task_status(task_id: str):
#     res = AsyncResult(task_id, app=celery_app)
#     return {
#         "task_id": task_id,
#         "status": res.status,
#         "result": res.result if res.status == "SUCCESS" else None,
#     }

# @app.get("/progress")
# def get_pipeline_progress():
#     """
#     Gets the status of all processed seeds from the run_progress collection.
#     Returns a list of seeds that are completed or paused.
#     """
#     try:
#         df_progress = load_collection_as_df("run_progress")
#         if df_progress.empty:
#             return {"status": "empty", "runs": []}
        
#         # We only care about the latest status for each run_tag
#         df_progress = df_progress.sort_values("updated_at", ascending=False)
#         df_progress = df_progress.drop_duplicates(subset=["run_tag"])
        
#         # Get all completed or paused runs
#         runs = df_progress[df_progress["status"].isin(["completed", "paused_due_to_quota"])]
        
#         return {
#             "status": "success",
#             "runs": runs.to_dict('records') # Send the full data to the frontend
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# # --- 2. NEW ENDPOINT: DOWNLOAD ---
# @app.get("/download_tier1_2")
# def download_all_tier1_and_2_channels():
#     """
#     Fetches ALL Tier 1 and Tier 2 channels from ALL
#     completed seeds, combines them, de-duplicates,
#     and returns the final master list.
#     """
#     try:
#         # 1. Get all *completed* run_tags
#         df_progress = load_collection_as_df("run_progress", {"status": "completed"})
#         if df_progress.empty:
#             return {"status": "empty", "message": "No runs have completed yet."}
            
#         completed_tags = df_progress["run_tag"].unique()
        
#         all_results_dfs = []
        
#         # 2. Loop through each completed run and get its T1/T2 results
#         for tag in completed_tags:
#             collection_name = f"{tag.upper()}_phase3"
            
#             # This is your "smart query" idea
#             df_tier1_2 = load_collection_as_df(
#                 collection_name,
#                 {"tier": {"$in": [1, 2]}} # Only get Tiers 1 and 2
#             )
            
#             if not df_tier1_2.empty:
#                 all_results_dfs.append(df_tier1_2)

#         if not all_results_dfs:
#             return {"status": "empty", "message": "No Tier 1 or 2 channels found."}

#         # 3. Combine, de-duplicate, and return
#         df_master_list = pd.concat(all_results_dfs, ignore_index=True)
        
#         # This is your manager's "40 + 60" logic
#         df_master_list = df_master_list.sort_values(by="tier", ascending=True)
#         df_master_list = df_master_list.drop_duplicates(subset=["Discovered_Channel_ID"])
        
#         # Only return the columns the client cares about
#         final_columns = [
#             "tier", 
#             "Discovered_Channel_Name", 
#             "Discovered_Channel_URL", 
#             "reason", 
#             "run_tag"
#         ]
#         # Filter to only columns that exist
#         final_columns = [col for col in final_columns if col in df_master_list.columns]
        
#         return {
#             "status": "success",
#             "total_channels": len(df_master_list),
#             "channels": df_master_list[final_columns].to_dict('records')
#         }
        
#     except Exception as e:
#         return {"status": "error", "message": str(e)}


from fastapi import FastAPI
from celery_worker import run_phase_pipeline, celery_app
from celery.result import AsyncResult
import pandas as pd
from typing import List

# --- ADDED: We need load_collection_as_df ---
from utils.mongo_utils import load_collection_as_df
# We DON'T need normalize_run_tag here, you were right.

app = FastAPI(title="Kajkarma AI Pipeline API")

@app.on_event("startup")
async def startup_event():
    print(f"🚀 Celery Broker: {celery_app.conf.broker_url}")
    print(f"🚀 Celery Backend: {celery_app.conf.result_backend}")

@app.post("/start_pipeline")
def start_pipeline(sheet_url: str):
    # This is correct as a POST.
    # We pass None for seed_dict to trigger "Loader" mode.
    task = run_phase_pipeline.delay(sheet_url, None) 
    return {"task_id": task.id, "message": "Pipeline 'Loader' task started."}

@app.get("/status/{task_id}")
def get_task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": res.status,
        "result": res.result if res.status == "SUCCESS" else None,
    }

@app.get("/progress")
def get_pipeline_progress():
    """
    Returns ALL run_tags grouped by their latest status.
    Example:
    {
        "completed": ["COFFEEZILLA", "MAGNETASMEDIA"],
        "skipped": ["MIKE_SWATRZ"],
        "started": ["SCAMMER_PAYBACK"],
        "failed": []
    }
    """
    try:
        df = load_collection_as_df("run_progress", {})
        if df.empty:
            return {"status": "empty", "data": {}}

        # 1. Sort newest first
        df = df.sort_values("updated_at", ascending=False)

        # 2. Get only the LATEST record per run_tag
        latest = df.drop_duplicates(subset=["run_tag"])

        # 3. Build a dict: status -> list of run_tags (UPPERCASE)
        status_map = {}

        for _, row in latest.iterrows():
            status = row["status"]
            tag = row["run_tag"].upper()

            if status not in status_map:
                status_map[status] = []

            status_map[status].append(tag)

        # 4. Return everything
        return {
            "status": "success",
            "data": status_map
        }

    except Exception as e:
        return {"status": 'error', "message": str(e)}



# --- 2. NEW ENDPOINT: DOWNLOAD ---
@app.get("/download_tier1_2")
def download_all_tier1_and_2_channels():
    """
    Fetches ALL Tier 1 and Tier 2 channels from ALL
    completed seeds, combines them, de-duplicates,
    and returns the final master list.
    (This is your manager's "40 + 60" logic)
    """
    try:
        # 1. Get all *completed* run_tags
        df_progress = load_collection_as_df("run_progress", {"status": "completed"})
        if df_progress.empty:
            return {"status": "empty", "message": "No runs have completed yet."}
            
        completed_tags = df_progress["run_tag"].unique()
        
        all_results_dfs = []
        
        # 2. Loop through each completed run and get its T1/T2 results
        for tag in completed_tags:
            collection_name = f"{tag.upper()}_phase3"
            
            # This is your "smart query" idea
            df_tier1_2 = load_collection_as_df(
                collection_name,
                {"tier": {"$in": [1, 2]}} # Only get Tiers 1 and 2
            )
            
            if not df_tier1_2.empty:
                all_results_dfs.append(df_tier1_2)

        if not all_results_dfs:
            return {"status": "empty", "message": "No Tier 1 or 2 channels found."}

        # 3. Combine, de-duplicate, and return
        df_master_list = pd.concat(all_results_dfs, ignore_index=True)
        
        # This is for previous outputs to go as well if clicked again!
        df_master_list = df_master_list.sort_values(by="tier", ascending=True)
        df_master_list = df_master_list.drop_duplicates(subset=["Discovered_Channel_ID"])
        
        # Only return the columns the client cares about
        final_columns = [
            "tier", 
            "Discovered_Channel_Name", 
            "Discovered_Channel_URL", 
            "reason", 
            "Discovered_From_Run" # This column comes from the feedback loop
        ]
        
        # Add 'run_tag' as a fallback if 'Discovered_From_Run' isn't there
        if "Discovered_From_Run" not in df_master_list.columns and "run_tag" in df_master_list.columns:
            final_columns.append("run_tag")
            
        # Filter to only columns that actually exist
        final_columns_exists = [col for col in final_columns if col in df_master_list.columns]
        
        return {
            "status": "success",
            "total_channels": len(df_master_list),
            "channels": df_master_list[final_columns_exists].to_dict('records')
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}