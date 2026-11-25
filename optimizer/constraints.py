from pydantic import BaseModel, ConfigDict, Field

class VLLMCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_num_seqs: int = Field(ge=1, le=512)
    max_num_batched_tokens: int = Field(ge=128, le=65536)
    gpu_memory_utilization: float = Field(ge=0.50, le=0.95)
    enable_chunked_prefill: bool

def validate_candidate(payload):
    return VLLMCandidate.model_validate(payload)
