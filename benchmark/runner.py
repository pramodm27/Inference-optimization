import asyncio
import time
import uuid
import httpx
from benchmark.metrics import summarize

def build_prompt(approx_tokens):
    # Approximate only. Replace with a tokenizer-backed fixed corpus for rigorous benchmarking.
    sentence = "Describe reliability and performance tradeoffs in distributed model serving. "
    return (sentence * max(1, approx_tokens // 10))[:max(1, approx_tokens * 5)]

async def run_batch(adapter, count, concurrency, input_tokens, output_tokens, temperature):
    semaphore = asyncio.Semaphore(concurrency)
    prompt = build_prompt(input_tokens)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        async def invoke(index):
            async with semaphore:
                try:
                    item = await adapter.generate(client, prompt, output_tokens, temperature)
                    item.update({"request_id": str(uuid.uuid4()), "index": index})
                    return item
                except Exception as exc:
                    return {"ok": False, "request_id": str(uuid.uuid4()), "index": index,
                            "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        started = time.perf_counter()
        records = await asyncio.gather(*(invoke(i) for i in range(count)))
        elapsed = time.perf_counter()-started
    return summarize(records, elapsed), records

async def run_benchmark(adapter, request):
    warmup = None
    if request.warmup:
        warmup, _ = await run_batch(adapter, request.warmup,
            min(request.concurrency, request.warmup), request.input_tokens,
            request.output_tokens, request.temperature)
    trials, all_records = [], []
    for index in range(request.trials):
        summary, records = await run_batch(adapter, request.requests, request.concurrency,
            request.input_tokens, request.output_tokens, request.temperature)
        summary["trial"] = index+1
        trials.append(summary)
        all_records.extend(records)
    total_wall = sum(x["wall_seconds"] for x in trials)
    aggregate = summarize(all_records, total_wall)
    return {
        "run_id": str(uuid.uuid4()), "workload": request.model_dump(),
        "warmup": warmup, "trials": trials, "aggregate": aggregate,
        "samples": all_records,
        "notes": [
            "Closed-loop concurrency, not open-loop arrival-rate testing.",
            "Prompt length is approximate unless a tokenizer-backed corpus is supplied.",
            "TTFT is first non-empty text delta; SSE chunks are not treated as tokens.",
            "Token throughput and TPOT require server-provided usage metadata.",
        ],
    }
