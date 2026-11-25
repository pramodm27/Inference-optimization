from benchmark.metrics import percentile, summarize

def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5

def test_summary_counts_failures():
    out = summarize([
        {"ok": True, "latency_s": 1, "ttft_s": .2, "tpot_s": .1, "completion_tokens": 5},
        {"ok": False}
    ], 2)
    assert out["requests_succeeded"] == 1
    assert out["requests_failed"] == 1
    assert out["output_tokens_per_second"] == 2.5
