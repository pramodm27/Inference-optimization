import asyncio, json, os
from pathlib import Path
from benchmark.config import get_settings
from benchmark.engine_adapter import VLLMOpenAIAdapter
from benchmark.runner import run_benchmark
from profiling.gpu_telemetry import snapshot
from agents.diagnosis_agent import diagnose

async def execute(request):
    settings = get_settings()
    adapter = VLLMOpenAIAdapter(settings.vllm_base_url, settings.vllm_model,
                                settings.vllm_api_key, settings.benchmark_timeout_seconds)
    result = await run_benchmark(adapter, request)
    try:
        telemetry = await asyncio.to_thread(snapshot)
    except Exception as exc:
        telemetry = {"error": str(exc)}
    result["diagnosis"] = diagnose(result, telemetry)
    directory = Path(settings.results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (result["run_id"] + ".json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    with (directory / "runs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"run_id": result["run_id"], "aggregate": result["aggregate"]}) + "\n")
    return result
