import argparse
import asyncio
import json
from benchmark.schemas import BenchmarkRequest
from orchestration.controller import execute


def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmark trials")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--mode", choices=["closed_loop", "open_loop"], default="closed_loop")
    parser.add_argument("--arrival-rate-rps", type=float, default=None)
    parser.add_argument("--prompt-corpus-path", type=str, default=None)
    args = parser.parse_args()

    req = BenchmarkRequest(
        requests=args.requests,
        concurrency=args.concurrency,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        warmup=args.warmup,
        trials=args.trials,
        mode=args.mode,
        arrival_rate_rps=args.arrival_rate_rps,
        prompt_corpus_path=args.prompt_corpus_path,
    )
    result = asyncio.run(execute(req))
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "aggregate": result["aggregate"],
                "diagnosis": result["diagnosis"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
