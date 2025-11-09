# main.py
from fastapi import FastAPI
from celery_worker import run_phase_pipeline, celery_app
from celery.result import AsyncResult

app = FastAPI(title="Kajkarma AI Pipeline API")

@app.on_event("startup")
async def startup_event():
    print(f"🚀 Celery Broker: {celery_app.conf.broker_url}")
    print(f"🚀 Celery Backend: {celery_app.conf.result_backend}")

@app.post("/start_pipeline")
def start_pipeline(sheet_url: str):
    task = run_phase_pipeline.delay(sheet_url)
    return {"task_id": task.id, "message": "Pipeline started."}

@app.get("/status/{task_id}")
def get_task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": res.status,
        "result": res.result if res.status == "SUCCESS" else None,
    }
