import pytest
from pydantic import ValidationError
from optimizer.constraints import validate_candidate

def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        validate_candidate({"max_num_seqs": 64, "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": .9, "enable_chunked_prefill": True, "shell": "whoami"})

def test_bounds_enforced():
    with pytest.raises(ValidationError):
        validate_candidate({"max_num_seqs": 9999, "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": .9, "enable_chunked_prefill": True})
