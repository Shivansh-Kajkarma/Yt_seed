from fastapi import FastAPI
from celery_worker import run_phase_pipeline, celery_app
from celery.result import AsyncResult
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List

# --- ADDED: We need load_collection_as_df ---
from utils.mongo_utils import load_collection_as_df
# We DON'T need normalize_run_tag here, you were right.

app = FastAPI(title="Kajkarma AI Pipeline API")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    print(f"🚀 Celery Broker: {celery_app.conf.broker_url}")
    print(f"🚀 Celery Backend: {celery_app.conf.result_backend}")


@app.post("/start_pipeline")
def start_pipeline(
    sheet_url: str,
    input_format: str = "Podcast",
    clients_intent: str = "Podcast",
):
    # This is correct as a POST.
    # We pass None for seed_dict to trigger "Loader" mode.
    # NEW: Added input_format and clients_intent parameters
    task = run_phase_pipeline.delay(sheet_url, None, input_format, clients_intent)
    return {
        "task_id": task.id,
        "message": f"Pipeline started with format='{input_format}', intent='{clients_intent}'",
    }


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
    Returns ALL run_tags grouped by their latest status with phase details.
    Example:
    {
        "completed": [{"run_tag": "COFFEEZILLA", "current_phase": "all_phases_complete"}],
        "in_progress": [{"run_tag": "SCAMMER_PAYBACK", "current_phase": "phase3_step2_done"}],
        "skipped": [{"run_tag": "MIKE_SWATRZ", "current_phase": null}],
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

        # 3. Build a dict: status -> list of {run_tag, current_phase, progress_percentage}
        status_map = {}

        for _, row in latest.iterrows():
            status = row["status"]
            tag = row["run_tag"].upper()
            current_phase = row.get("current_phase", None)
            progress_percentage = row.get("progress_percentage", 0)

            if status not in status_map:
                status_map[status] = []

            status_map[status].append(
                {
                    "run_tag": tag,
                    "current_phase": current_phase,
                    "progress_percentage": progress_percentage,
                }
            )

        # 4. Return everything
        return {"status": "success", "data": status_map}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- 2. NEW ENDPOINT: DOWNLOAD ---
@app.get("/download_tier1_2")
def download_all_tier1_and_2_channels():
    """
    Fetches ALL Tier 1 and Tier 2 channels from ALL
    completed seeds, combines them, de-duplicates,
    and returns the final master list.
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
            # NEW: Load from phase4 final_ranked collection
            collection_name = f"{tag.upper()}_final_ranked"

            # This is your "smart query" idea
            df_tier1_2 = load_collection_as_df(
                collection_name,
                {"Final_Tier": {"$in": [1, 2]}},  # Only get Tiers 1 and 2 from phase4
            )

            if not df_tier1_2.empty:
                all_results_dfs.append(df_tier1_2)

        if not all_results_dfs:
            return {"status": "empty", "message": "No Tier 1 or 2 channels found."}

        # 3. Combine, de-duplicate, and return
        df_master_list = pd.concat(all_results_dfs, ignore_index=True)

        # This is for previous outputs to go as well if clicked again!
        df_master_list = df_master_list.sort_values(by="Final_Tier", ascending=True)
        df_master_list = df_master_list.drop_duplicates(
            subset=["Discovered_Channel_ID"]
        )

        # Only return the columns the client cares about
        final_columns = [
            "Discovered_Channel_ID",
            "Discovered_Channel_Name",
            "Discovered_Channel_URL",
            "Final_Tier",
            "Final_Status",
            "LLM_Recheck_Reason",
            "run_tag",
        ]

        # Filter to only columns that actually exist
        final_columns_exists = [
            col for col in final_columns if col in df_master_list.columns
        ]

        # CRITICAL FIX: Clean NaN/Inf values before JSON serialization
        df_output = df_master_list[final_columns_exists].copy()

        # Replace NaN with None (null in JSON)
        df_output = df_output.fillna("")

        # Replace inf/-inf with None
        df_output = df_output.replace([float("inf"), float("-inf")], "")

        return {
            "status": "success",
            "total_channels": len(df_output),
            "channels": df_output.to_dict("records"),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
