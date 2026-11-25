from pydantic import BaseModel, Field, model_validator

class BenchmarkRequest(BaseModel):
    requests: int = Field(default=40, ge=1, le=10000)
    concurrency: int = Field(default=4, ge=1, le=512)
    input_tokens: int = Field(default=256, ge=1, le=32768)
    output_tokens: int = Field(default=128, ge=1, le=8192)
    warmup: int = Field(default=4, ge=0, le=1000)
    trials: int = Field(default=3, ge=1, le=20)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def check_concurrency(self):
        if self.warmup + self.requests < self.concurrency:
            raise ValueError("concurrency cannot exceed warmup + requests")
        return self
