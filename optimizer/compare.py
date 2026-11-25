def compare_runs(baseline, candidate, max_error_rate=0.01):
    b, c = baseline["aggregate"], candidate["aggregate"]
    bp = (b.get("request_latency_s") or {}).get("p95")
    cp = (c.get("request_latency_s") or {}).get("p95")
    bt, ct = b.get("output_tokens_per_second"), c.get("output_tokens_per_second")
    checks = {
        "error_rate_ok": c.get("error_rate", 1.0) <= max_error_rate,
        "latency_data_available": bp is not None and cp is not None,
        "throughput_data_available": bt is not None and ct is not None,
        "no_latency_regression": bp is not None and cp is not None and cp <= bp,
        "no_throughput_regression": bt is not None and ct is not None and ct >= bt,
    }
    improved = bool(
        bp is not None
        and cp is not None
        and bt is not None
        and ct is not None
        and cp <= bp
        and ct >= bt
        and (cp < bp or ct > bt)
    )
    return {
        "accepted_for_human_review": all(checks.values()) and improved,
        "checks": checks,
        "baseline_p95_s": bp,
        "candidate_p95_s": cp,
        "baseline_output_tokens_s": bt,
        "candidate_output_tokens_s": ct,
        "warning": "Not an automatic production promotion. Validate quality, variance, memory, TTFT/TPOT, and repeatability.",
    }
