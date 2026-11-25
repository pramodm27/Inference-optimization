import math

def percentile(values, p):
    data = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not data:
        return None
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    rank = (len(data) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return data[lo] if lo == hi else data[lo] * (hi-rank) + data[hi] * (rank-lo)

def summarize(samples, wall_seconds):
    good = [x for x in samples if x.get("ok")]
    lat = [x["latency_s"] for x in good]
    ttft = [x["ttft_s"] for x in good if x.get("ttft_s") is not None]
    tpot = [x["tpot_s"] for x in good if x.get("tpot_s") is not None]
    completion = sum(x.get("completion_tokens") or 0 for x in good)
    token_usage_available = bool(good) and all(x.get("completion_tokens") is not None for x in good)
    return {
        "requests_total": len(samples),
        "requests_succeeded": len(good),
        "requests_failed": len(samples)-len(good),
        "error_rate": (len(samples)-len(good))/len(samples) if samples else 0,
        "wall_seconds": wall_seconds,
        "request_latency_s": {k: percentile(lat, p) for k,p in [("p50",50),("p95",95),("p99",99)]},
        "ttft_s": {k: percentile(ttft, p) for k,p in [("p50",50),("p95",95),("p99",99)]},
        "tpot_s": {k: percentile(tpot, p) for k,p in [("p50",50),("p95",95),("p99",99)]},
        "completion_tokens": completion if token_usage_available else None,
        "output_tokens_per_second": completion/wall_seconds if (token_usage_available and wall_seconds > 0) else None,
        "token_usage_available": token_usage_available,
    }
