from fastapi import FastAPI
from celery_worker import run_phase_pipeline, celery_app
from celery.result import AsyncResult
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uuid
from typing import List

# --- Utils ---
from utils.mongo_utils import load_collection_as_df

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
    clients_intent: str = "General",
):
    # 1. Generate a UNIQUE ID for this run (This is your "Job ID")
    pipeline_execution_id = str(uuid.uuid4())

    # 2. DECIDE THE LANE (Queue Routing)
    target_queue = "default"
    format_lower = input_format.lower()

    if "podcast" in format_lower:
        target_queue = "podcast_queue"
    elif "doc" in format_lower:
        target_queue = "doc_queue"
    elif "talking" in format_lower:
        target_queue = "talking_head_queue"

    print(
        f"🚦 New Run: {pipeline_execution_id} | Format: {input_format} -> Queue: {target_queue}"
    )

    # 3. Fire Task
    task = run_phase_pipeline.apply_async(
        args=[sheet_url, None, input_format, clients_intent, pipeline_execution_id],
        queue=target_queue,
    )

    return {
        "task_id": task.id,
        "pipeline_execution_id": pipeline_execution_id,
        "queue": target_queue,
        "message": f"Pipeline started in lane '{target_queue}'",
    }


@app.get("/progress")
def get_pipeline_progress():
    """
    Returns ALL runs. Frontend uses this to build the 'Tabs'.
    """
    try:
        df = load_collection_as_df("run_progress", {})
        if df.empty:
            return {"status": "empty", "data": {}}

        # Sort by latest
        df = df.sort_values("updated_at", ascending=False)

        # Get latest status per run_tag
        latest = df.drop_duplicates(subset=["run_tag"])

        # Convert to list for easier Frontend parsing
        progress_list = []
        for _, row in latest.iterrows():
            # FIX: Safely get progress_percentage and handle NaN/Inf
            progress_pct = row.get("progress_percentage", 0)
            if (
                pd.isna(progress_pct)
                or progress_pct == float("inf")
                or progress_pct == float("-inf")
            ):
                progress_pct = 0
            else:
                progress_pct = int(progress_pct)

            item = {
                "run_tag": str(row.get("run_tag", "")),
                "status": str(row.get("status", "unknown")),
                "pipeline_execution_id": str(row.get("pipeline_execution_id", "N/A")),
                "current_phase": str(row.get("current_phase", "unknown")),
                "progress_percentage": progress_pct,
                "updated_at": str(row.get("updated_at", "")),
            }
            progress_list.append(item)

        return {"status": "success", "data": progress_list}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/download_run_results/{pipeline_execution_id}")
def download_run_results(pipeline_execution_id: str):
    """
    Downloads Tier 1 & 2 results ONLY for a specific Pipeline Run UUID.
    """
    try:
        print(f"📥 Searching for results with UUID: {pipeline_execution_id}")

        # 1. Find which run_tags belong to this UUID
        df_runs = load_collection_as_df(
            "run_progress", {"pipeline_execution_id": pipeline_execution_id}
        )

        if df_runs.empty:
            return {"status": "empty", "message": "No runs found for this ID."}

        relevant_tags = df_runs["run_tag"].unique()
        all_results_dfs = []

        # 2. Fetch results for each tag
        for tag in relevant_tags:
            collection_name = f"{tag.upper()}_final_ranked"
            try:
                # Query for Tier 1/2 AND ensure the UUID matches (double safety)
                df_tier = load_collection_as_df(
                    collection_name,
                    {
                        "Final_Tier": {"$in": [1, 2]},
                        "pipeline_execution_id": pipeline_execution_id,
                    },
                )
                if not df_tier.empty:
                    all_results_dfs.append(df_tier)
            except Exception:
                continue

        if not all_results_dfs:
            return {"status": "empty", "message": "No qualified channels found yet."}

        # 3. Combine & Clean
        df_master = pd.concat(all_results_dfs, ignore_index=True)
        # df_master = df_master.drop(columns=["Deep_Scan_Data", "Discovered_Videos_JSON"], errors="ignore")
        df_master = df_master.drop_duplicates(subset=["Discovered_Channel_ID"])

        desired_columns = [
            "Discovered_Channel_ID",
            "Discovered_Channel_Name",
            "Discovered_Channel_URL",
            "run_tag",
            "Final_Tier",
            "Final_Status",
            "Seed_Channel_Name"
        ]
        
        # Only keep columns that actually exist in the dataframe (safety check)
        existing_cols = [c for c in desired_columns if c in df_master.columns]
        df_master = df_master[existing_cols]
        
        if "Final_Tier" in df_master.columns:
            df_master = df_master.sort_values("Final_Tier")

        df_master = df_master.fillna("").replace([float("inf"), float("-inf")], "")

        return {
            "status": "success",
            "pipeline_id": pipeline_execution_id,
            "total_channels": len(df_master),
            "channels": df_master.to_dict("records"),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
