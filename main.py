from fastapi import FastAPI
from celery_worker import run_phase_pipeline, celery_app
from celery.result import AsyncResult
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uuid
from datetime import datetime, timedelta
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
            "Seed_Channel_Name",
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


@app.get("/queue_status_all")
def get_queue_status_dashboard():
    """
    Multi-Lane Dashboard Status (Mission Control).
    Returns the status of Podcast, Documentary, and Talking Head lanes simultaneously.
    Shows countdown timer for 25-hour batch cooldown.
    """
    lanes = ["Podcast", "Documentary", "Talking Head"]
    dashboard = {}

    # Lane key mapping for flexible matching
    lane_key_map = {
        "Podcast": ["podcast"],
        "Documentary": ["doc", "documentary", "documentry"],
        "Talking Head": ["talking", "talkinghead", "talking_head"],
    }

    try:
        # Load all run progress
        df = load_collection_as_df("run_progress", {})

        if df.empty:
            for lane in lanes:
                dashboard[lane] = {
                    "status": "idle",
                    "message": "No history",
                    "timer_seconds": 0,
                }
            return {"status": "success", "dashboard": dashboard}

        # Pre-process dates
        df["updated_at_dt"] = pd.to_datetime(df["updated_at"], errors="coerce")
        if "started_at" in df.columns:
            df["started_at_dt"] = pd.to_datetime(df["started_at"], errors="coerce")
        else:
            df["started_at_dt"] = pd.NaT

        for lane in lanes:
            keys = lane_key_map[lane]

            # Filter runs for this lane
            lane_runs = pd.DataFrame()

            if "input_format" in df.columns:
                # Primary: Match by input_format column
                for key in keys:
                    mask = (
                        df["input_format"]
                        .astype(str)
                        .str.lower()
                        .str.contains(key, na=False)
                    )
                    lane_runs = pd.concat([lane_runs, df[mask]])

            # Fallback: If no input_format matches, try run_tag
            if lane_runs.empty:
                for key in keys:
                    mask = (
                        df["run_tag"]
                        .astype(str)
                        .str.lower()
                        .str.contains(key, na=False)
                    )
                    lane_runs = pd.concat([lane_runs, df[mask]])

            # Deduplicate
            if not lane_runs.empty:
                lane_runs = lane_runs.drop_duplicates(subset=["run_tag"])

            if lane_runs.empty:
                dashboard[lane] = {
                    "status": "idle",
                    "message": "Ready to start",
                    "timer_seconds": 0,
                }
                continue

            # Get the LATEST run for this lane
            latest_run = lane_runs.sort_values("updated_at_dt", ascending=False).iloc[0]

            status = latest_run["status"]
            run_tag = latest_run["run_tag"]
            current_phase = latest_run.get("current_phase", "unknown")
            progress_pct = latest_run.get("progress_percentage", 0)

            # Handle NaN in progress
            if pd.isna(progress_pct):
                progress_pct = 0
            else:
                progress_pct = int(progress_pct)

            # --- LOGIC: RUNNING vs COOLDOWN vs IDLE ---

            # A. RUNNING
            if status in ["started", "in_progress"]:
                dashboard[lane] = {
                    "status": "running",
                    "run_tag": run_tag,
                    "current_phase": str(current_phase),
                    "progress_percentage": progress_pct,
                    "message": f"Processing {run_tag}...",
                    "timer_seconds": 0,
                }

            # B. COOLDOWN (The 25h Timer)
            elif status in ["completed", "failed", "skipped"]:
                # Calculate: Target = Started_At + 25 Hours
                started_at = latest_run.get("started_at_dt")

                # Fallback to updated_at if started_at wasn't saved
                if pd.isna(started_at):
                    base_time = latest_run["updated_at_dt"]
                else:
                    base_time = started_at

                # 25 Hour Rule (or use BATCH_COOLDOWN from config)
                target_resume_time = base_time + timedelta(minutes=20)  # Testing: 20 minutes
                now = datetime.now()
                remaining = target_resume_time - now

                if remaining.total_seconds() > 0:
                    # Timer is active - show countdown
                    timer_secs = int(remaining.total_seconds())

                    # Human-readable format
                    hours = timer_secs // 3600
                    mins = (timer_secs % 3600) // 60
                    display = f"{hours}h {mins}m remaining"

                    dashboard[lane] = {
                        "status": "cooldown",
                        "run_tag": run_tag,
                        "message": "Batch Cooldown Active",
                        "timer_seconds": timer_secs,
                        "timer_display": display,
                        "resumes_at": target_resume_time.isoformat(),
                    }
                else:
                    # Timer expired - ready for next batch
                    dashboard[lane] = {
                        "status": "idle",
                        "message": "Cooldown complete. Ready.",
                        "timer_seconds": 0,
                    }

            else:
                dashboard[lane] = {
                    "status": "idle",
                    "message": "Ready",
                    "timer_seconds": 0,
                }

        return {"status": "success", "dashboard": dashboard}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}
