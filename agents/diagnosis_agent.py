def diagnose(benchmark, telemetry=None):
    agg = benchmark["aggregate"]
    hypotheses = []
    error_rate = agg.get("error_rate", 1.0)
    if error_rate > 0.01:
        hypotheses.append({
            "category": "reliability_or_overload", "confidence": "medium",
            "reason": f"Observed error rate {error_rate:.2%} exceeds 1%.",
            "next_step": "Inspect engine logs, queueing, timeout policy, and offered load."
        })
    ttft = (agg.get("ttft_s") or {}).get("p95")
    tpot = (agg.get("tpot_s") or {}).get("p95")
    if ttft is not None and tpot is not None:
        if ttft > 2.0 and tpot <= 0.08:
            hypotheses.append({
                "category": "prefill_or_queueing", "confidence": "low",
                "reason": "P95 TTFT is high relative to P95 TPOT.",
                "next_step": "Separate queue time from prefill; inspect batch mix and prefill kernels."
            })
        elif tpot > 0.08:
            hypotheses.append({
                "category": "decode_path", "confidence": "low",
                "reason": "P95 TPOT is elevated for this workload.",
                "next_step": "Inspect decode kernels, KV-cache pressure, batch size, and GPU clocks."
            })
    if not hypotheses:
        hypotheses.append({
            "category": "insufficient_evidence", "confidence": "low",
            "reason": "No configured heuristic fired.",
            "next_step": "Repeat trials and correlate request metrics with engine metrics and Nsight."
        })
    return {
        "type": "heuristic_hypotheses_not_ground_truth",
        "hypotheses": hypotheses,
        "gpu_telemetry": telemetry,
        "limitations": [
            "A single GPU utilization snapshot cannot establish a bottleneck.",
            "TTFT includes network, queueing, scheduling, and prefill.",
            "Confirm hypotheses with engine metrics and targeted Nsight captures."
        ]
    }
