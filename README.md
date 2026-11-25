# Agentic Inference Optimizer

Production-oriented starter for benchmarking, diagnosing, and safely tuning LLM inference on a dedicated Ubuntu/NVIDIA GPU host (target: H100 + vLLM).

Included:
- Async OpenAI-compatible streaming client.
- Bounded-concurrency benchmark with warmup and repeated trials.
- TTFT, TPOT, request latency percentiles, output tokens/sec, and error rate.
- Server-provided token usage; SSE chunk boundaries are never mistaken for tokens.
- `nvidia-smi` telemetry and safe Nsight Systems/Compute wrappers.
- Evidence-labelled heuristic diagnosis.
- Pydantic allow-list for candidate vLLM settings.
- FastAPI control plane and atomic JSON result persistence.
- Unit tests.

This is a strong engineering foundation, not a claim of production certification or measured performance gains. It uses local JSON files and in-process API tasks; before multi-user production use, add a durable queue/database, authentication, authorization, quotas, audit logging, and deployment isolation.

## Host validation

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
nvcc --version || true
python3 --version
uname -a
nvidia-smi topo -m
```

`nvidia-smi`'s CUDA Version is the maximum CUDA level supported by the driver; it does not prove the CUDA Toolkit (`nvcc`) is installed.

## Install

Use a fresh Python 3.12 environment. Install vLLM according to the current official instructions for your driver/Python combination; CUDA wheel compatibility changes over time.

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git curl jq
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv venv --python 3.12 .venv
source .venv/bin/activate

# Example vLLM flow; verify against current official docs for your host:
uv pip install vllm --torch-backend=auto
vllm download-kernels

uv pip install -r requirements.txt
cp .env.example .env
```

Do not install an unrelated standalone PyTorch wheel into the same vLLM environment first. vLLM's supported wheel determines compatible PyTorch/CUDA dependencies.

Edit `.env` and set the model name and API key. Start vLLM on a private interface:

```bash
export VLLM_API_KEY='your-secret'
vllm serve YOUR_MODEL_ID --host 127.0.0.1 --port 8000 --dtype auto --api-key "$VLLM_API_KEY"
```

Do not expose an unauthenticated inference endpoint publicly. Use TLS, authentication, network policy, and a secret manager.

## Run

```bash
set -a && source .env && set +a
python -m benchmark.gpu_info
python -m benchmark.run_cli --requests 80 --concurrency 8 --input-tokens 512 --output-tokens 128 --warmup 8 --trials 3
uvicorn api.main:app --host 127.0.0.1 --port 8080
pytest -q
```

API endpoints:
- `GET /health`
- `GET /v1/system/gpus`
- `POST /v1/benchmarks`
- `GET /v1/benchmarks/{run_id}`
- `GET /v1/benchmarks`

Example:
```bash
curl -s http://127.0.0.1:8080/v1/benchmarks \
  -H 'content-type: application/json' \
  -d '{"requests":40,"concurrency":4,"input_tokens":256,"output_tokens":128,"warmup":4,"trials":2}'
```

## Metric definitions and limitations

- TTFT: request start to first non-empty generated text delta.
- TPOT: `(latency - TTFT) / (completion_tokens - 1)`, only when server usage is returned.
- Streaming chunks are not tokens. If the server omits usage, token-dependent metrics are unavailable rather than guessed.
- Prompt length generator is approximate. Use a tokenizer-backed, fixed prompt corpus for rigorous tests.
- Runner is closed-loop concurrency. Add an open-loop arrival-rate generator for capacity/SLO testing.
- Run at least three repetitions under identical model, engine version, workload, server flags, clocks and power policy.
- Profiling adds overhead. Capture Nsight on an isolated deployment, not while serving production traffic.
- Diagnosis rules are hypotheses, not a learned classifier or ground truth.

## Safe optimization loop

1. Baseline benchmark and record engine/model/hardware metadata.
2. Repeat baseline and inspect variance.
3. Correlate request metrics with GPU telemetry and Nsight traces.
4. Generate candidate values only from `optimizer/constraints.py`.
5. Apply candidate to a separate deployment (this project does not mutate a running vLLM process).
6. Validate correctness, quality, error rate, TTFT, TPOT, throughput, memory, and tail latency.
7. Require human review before promotion.

Nsight tools are installed separately from NVIDIA's official packages. Cloud providers may restrict performance counters. Check `nsys --version`, `ncu --version`, and provider permissions before profiling.
