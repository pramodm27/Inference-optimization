import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Optional
import httpx
from benchmark.metrics import summarize


def build_prompt(approx_tokens: int) -> str:
    # Approximate only when no prompt_corpus_path is supplied.
    sentence = "Describe reliability and performance tradeoffs in distributed model serving. "
    return (sentence * max(1, approx_tokens // 10))[:max(1, approx_tokens * 5)]


def load_prompt_corpus(corpus_path: Optional[str]) -> Optional[list[str]]:
    """Load prompts from a .jsonl ({"prompt": ...}) or plain-text (one per line) file."""
    if not corpus_path:
        return None
    path = Path(corpus_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Prompt corpus file not found: {corpus_path}")
    prompts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if path.suffix == ".jsonl":
            obj = json.loads(line)
            text = obj.get("prompt")
            if isinstance(text, str) and text.strip():
                prompts.append(text.strip())
        else:
            prompts.append(line)
    if not prompts:
        raise ValueError(f"Prompt corpus file is empty: {corpus_path}")
    return prompts


def _select_prompt(index: int, approx_tokens: int, corpus: Optional[list[str]]) -> str:
    if corpus:
        return corpus[index % len(corpus)]
    return build_prompt(approx_tokens)


async def run_batch(
    adapter,
    count: int,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    temperature: float,
    mode: str = "closed_loop",
    arrival_rate_rps: Optional[float] = None,
    corpus: Optional[list[str]] = None,
):
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(limits=limits) as client:
        async def invoke(index: int, delay_s: float = 0.0):
            if delay_s > 0.0:
                await asyncio.sleep(delay_s)
            prompt = _select_prompt(index, input_tokens, corpus)
            async with semaphore:
                try:
                    item = await adapter.generate(client, prompt, output_tokens, temperature)
                    item.update({"request_id": str(uuid.uuid4()), "index": index})
                    return item
                except Exception as exc:
                    return {
                        "ok": False,
                        "request_id": str(uuid.uuid4()),
                        "index": index,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }

        if mode == "open_loop" and arrival_rate_rps:
            rng = random.Random(42)
            offsets = []
            acc = 0.0
            for i in range(count):
                if i > 0:
                    acc += rng.expovariate(arrival_rate_rps)
                offsets.append(acc)
            started = time.perf_counter()
            records = await asyncio.gather(*(invoke(i, offsets[i]) for i in range(count)))
            elapsed = time.perf_counter() - started
        else:
            started = time.perf_counter()
            records = await asyncio.gather(*(invoke(i) for i in range(count)))
            elapsed = time.perf_counter() - started

    return summarize(records, elapsed), records


async def run_benchmark(adapter, request, run_id: Optional[str] = None):
    corpus = load_prompt_corpus(getattr(request, "prompt_corpus_path", None))
    mode = getattr(request, "mode", "closed_loop")
    arrival_rate_rps = getattr(request, "arrival_rate_rps", None)

    warmup = None
    if request.warmup:
        warmup, _ = await run_batch(
            adapter,
            request.warmup,
            min(request.concurrency, request.warmup),
            request.input_tokens,
            request.output_tokens,
            request.temperature,
            mode="closed_loop",
            corpus=corpus,
        )

    trials, all_records = [], []
    for index in range(request.trials):
        summary, records = await run_batch(
            adapter,
            request.requests,
            request.concurrency,
            request.input_tokens,
            request.output_tokens,
            request.temperature,
            mode=mode,
            arrival_rate_rps=arrival_rate_rps,
            corpus=corpus,
        )
        summary["trial"] = index + 1
        trials.append(summary)
        all_records.extend(records)

    total_wall = sum(x["wall_seconds"] for x in trials)
    aggregate = summarize(all_records, total_wall)
    notes = [
        (
            f"Open-loop Poisson arrival-rate testing at {arrival_rate_rps} req/s (max concurrency cap {request.concurrency})."
            if mode == "open_loop"
            else "Closed-loop concurrency, not open-loop arrival-rate testing."
        ),
        (
            f"Prompts loaded from corpus: {request.prompt_corpus_path}"
            if corpus
            else "Prompt length is approximate unless a tokenizer-backed corpus is supplied."
        ),
        "TTFT is first non-empty text delta; SSE chunks are not treated as tokens.",
        "Token throughput and TPOT require server-provided usage metadata.",
    ]
    return {
        "run_id": run_id or str(uuid.uuid4()),
        "workload": request.model_dump(),
        "warmup": warmup,
        "trials": trials,
        "aggregate": aggregate,
        "samples": all_records,
        "notes": notes,
    }
