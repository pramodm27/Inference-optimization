from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class BenchmarkRequest(BaseModel):
    requests: int = Field(default=40, ge=1, le=10000)
    concurrency: int = Field(default=4, ge=1, le=512)
    input_tokens: int = Field(default=256, ge=1, le=32768)
    output_tokens: int = Field(default=128, ge=1, le=8192)
    warmup: int = Field(default=4, ge=0, le=1000)
    trials: int = Field(default=3, ge=1, le=20)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    mode: Literal["closed_loop", "open_loop"] = "closed_loop"
    arrival_rate_rps: Optional[float] = Field(default=None, gt=0.0, le=1000.0)
    prompt_corpus_path: Optional[str] = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def validate_workload(self):
        if self.mode == "closed_loop" and self.requests < self.concurrency:
            raise ValueError("concurrency cannot exceed requests in closed_loop mode")
        if self.mode == "open_loop" and self.arrival_rate_rps is None:
            raise ValueError("arrival_rate_rps is required when mode is open_loop")
        return self
