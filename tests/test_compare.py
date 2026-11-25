from optimizer.compare import compare_runs


def test_compare_runs_rejects_regression():
    baseline = {
        "aggregate": {
            "error_rate": 0.0,
            "request_latency_s": {"p95": 0.20},
            "output_tokens_per_second": 200.0,
        }
    }
    regressed_latency = {
        "aggregate": {
            "error_rate": 0.0,
            "request_latency_s": {"p95": 0.30},
            "output_tokens_per_second": 250.0,
        }
    }
    better_both = {
        "aggregate": {
            "error_rate": 0.0,
            "request_latency_s": {"p95": 0.15},
            "output_tokens_per_second": 240.0,
        }
    }
    assert compare_runs(baseline, regressed_latency)["accepted_for_human_review"] is False
    assert compare_runs(baseline, better_both)["accepted_for_human_review"] is True
