import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Request
from api.security import Principal, require_role, write_audit_event
from benchmark.config import get_settings
from benchmark.schemas import BenchmarkRequest
from orchestration.controller import execute
from orchestration.storage import RunStore
from profiling.gpu_telemetry import snapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference-optimizer")
background_tasks: set[asyncio.Task] = set()
_store: RunStore | None = None


def get_store() -> RunStore:
    global _store
    db_path = get_settings().database_path
    if _store is None or str(_store.db_path) != str(Path(db_path)):
        _store = RunStore(db_path)
    return _store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    store = get_store()
    recovered = store.recover_stale_runs()
    if recovered:
        logger.warning("Recovered %d interrupted benchmark run(s) on startup", recovered)
    yield


app = FastAPI(title="Agentic Inference Optimizer", version="0.2.0", lifespan=lifespan)


async def run_task(run_id: str, request: BenchmarkRequest) -> None:
    store = get_store()
    store.mark_running(run_id)
    try:
        result = await execute(request, run_id=run_id)
        result_path = str(Path(get_settings().results_dir) / f"{run_id}.json")
        store.mark_completed(run_id, result["aggregate"], result_path)
    except Exception as exc:
        logger.exception("Benchmark %s failed", run_id)
        store.mark_failed(run_id, str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    try:
        active = get_store().count_active_runs()
        return {"status": "ready", "active_runs": active}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Storage unavailable: {exc}")


@app.get("/v1/system/gpus")
async def gpu_status(
    request: Request,
    principal: Principal = Depends(require_role("viewer")),
):
    try:
        data = await asyncio.to_thread(snapshot)
        write_audit_event("gpu_status", principal, 200, request=request)
        return data
    except Exception as exc:
        write_audit_event("gpu_status", principal, 503, request=request, detail=str(exc))
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/v1/benchmarks", status_code=202)
async def start_benchmark(
    payload: BenchmarkRequest,
    request: Request,
    principal: Principal = Depends(require_role("operator")),
):
    settings = get_settings()
    store = get_store()
    if store.count_active_runs() >= settings.max_concurrent_runs:
        write_audit_event("start_benchmark", principal, 429, request=request, detail="Concurrent run quota exceeded")
        raise HTTPException(status_code=429, detail="Concurrent benchmark quota exceeded")

    run_id = str(uuid.uuid4())
    store.create_run(run_id, principal.actor_id, payload.model_dump())
    task = asyncio.create_task(run_task(run_id, payload))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    write_audit_event("start_benchmark", principal, 202, request=request, run_id=run_id)
    return {"run_id": run_id, "status": "queued"}


@app.get("/v1/benchmarks")
async def list_benchmarks(
    request: Request,
    limit: int = 100,
    principal: Principal = Depends(require_role("viewer")),
):
    bounded_limit = max(1, min(limit, 500))
    runs = get_store().list_runs(limit=bounded_limit)
    write_audit_event("list_benchmarks", principal, 200, request=request)
    return runs


@app.get("/v1/benchmarks/{run_id}")
async def get_benchmark(
    run_id: str,
    request: Request,
    principal: Principal = Depends(require_role("viewer")),
):
    try:
        parsed_id = str(uuid.UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    record = get_store().get_run(parsed_id)
    if record:
        response = {
            "run_id": record["run_id"],
            "status": record["status"],
            "owner": record["owner"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "error": record["error"],
        }
        if record["status"] == "completed" and record.get("result_path"):
            path = Path(record["result_path"])
            if path.exists():
                response["result"] = json.loads(path.read_text(encoding="utf-8"))
        write_audit_event("get_benchmark", principal, 200, request=request, run_id=parsed_id)
        return response

    path = Path(get_settings().results_dir) / f"{parsed_id}.json"
    if path.exists():
        write_audit_event("get_benchmark", principal, 200, request=request, run_id=parsed_id)
        return {"run_id": parsed_id, "status": "completed", "result": json.loads(path.read_text(encoding="utf-8"))}

    write_audit_event("get_benchmark", principal, 404, request=request, run_id=parsed_id)
    raise HTTPException(status_code=404, detail="Run ID not found")
