import json
import pytest
from pydantic import ValidationError
from benchmark.runner import load_prompt_corpus, run_benchmark
from benchmark.schemas import BenchmarkRequest


class FakeAdapter:
    def __init__(self):
        self.prompts = []

    async def generate(self, client, prompt, output_tokens, temperature):
        self.prompts.append(prompt)
        return {
            "ok": True,
            "latency_s": 0.02,
            "ttft_s": 0.005,
            "tpot_s": 0.001,
            "prompt_tokens": 10,
            "completion_tokens": output_tokens,
        }


def test_schema_validation_closed_and_open_loop():
    with pytest.raises(ValidationError):
        BenchmarkRequest(requests=2, concurrency=4, warmup=4, mode="closed_loop")
    with pytest.raises(ValidationError):
        BenchmarkRequest(requests=4, concurrency=2, mode="open_loop", arrival_rate_rps=None)
    req = BenchmarkRequest(requests=4, concurrency=2, mode="open_loop", arrival_rate_rps=50.0)
    assert req.mode == "open_loop"


@pytest.mark.asyncio
async def test_open_loop_and_prompt_corpus(tmp_path):
    corpus_file = tmp_path / "prompts.jsonl"
    corpus_file.write_text(
        json.dumps({"prompt": "Prompt A"}) + "\n" + json.dumps({"prompt": "Prompt B"}) + "\n",
        encoding="utf-8",
    )
    loaded = load_prompt_corpus(str(corpus_file))
    assert loaded == ["Prompt A", "Prompt B"]

    req = BenchmarkRequest(
        requests=4,
        concurrency=2,
        warmup=0,
        trials=1,
        output_tokens=8,
        mode="open_loop",
        arrival_rate_rps=200.0,
        prompt_corpus_path=str(corpus_file),
    )
    adapter = FakeAdapter()
    result = await run_benchmark(adapter, req, run_id="custom-run-id")
    assert result["run_id"] == "custom-run-id"
    assert result["aggregate"]["requests_succeeded"] == 4
    assert adapter.prompts == ["Prompt A", "Prompt B", "Prompt A", "Prompt B"]
