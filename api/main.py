import asyncio, json, logging, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from benchmark.config import get_settings
from benchmark.schemas import BenchmarkRequest
from orchestration.controller import execute
from profiling.gpu_telemetry import snapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference-optimizer")
app = FastAPI(title="Agentic Inference Optimizer", version="0.1.0")
runs = {}
background_tasks = set()

async def run_task(run_id, request):
    try:
        result = await execute(request)
        runs[run_id] = {"status": "completed", "result": result}
    except Exception as exc:
        logger.exception("Benchmark %s failed", run_id)
        runs[run_id] = {"status": "failed", "error": str(exc)[:1000]}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/system/gpus")
async def gpu_status():
    try:
        return await asyncio.to_thread(snapshot)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/v1/benchmarks", status_code=202)
async def start_benchmark(request: BenchmarkRequest):
    run_id = str(uuid.uuid4())
    runs[run_id] = {"status": "queued"}
    task = asyncio.create_task(run_task(run_id, request))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return {"run_id": run_id, "status": "queued"}

@app.get("/v1/benchmarks")
async def list_benchmarks():
    return [{"run_id": key, "status": value["status"]} for key, value in runs.items()]

@app.get("/v1/benchmarks/{run_id}")
async def get_benchmark(run_id: str):
    if run_id in runs:
        return runs[run_id]
    try:
        parsed_id = str(uuid.UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")
    path = Path(get_settings().results_dir) / f"{parsed_id}.json"
    if path.exists():
        return {"status": "completed", "result": json.loads(path.read_text())}
    raise HTTPException(status_code=404, detail="Run ID not found")
