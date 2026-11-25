from agents.diagnosis_agent import diagnose

def test_diagnosis_is_heuristic():
    result = diagnose({"aggregate": {"error_rate": 0,
        "ttft_s": {"p95": 2.4}, "tpot_s": {"p95": .04}}})
    assert result["type"] == "heuristic_hypotheses_not_ground_truth"
    assert result["hypotheses"][0]["category"] == "prefill_or_queueing"
