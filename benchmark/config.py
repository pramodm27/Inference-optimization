from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_model: str = "replace-with-served-model-name"
    vllm_api_key: str = "replace-me"
    results_dir: str = "results"
    benchmark_timeout_seconds: float = 180.0

@lru_cache
def get_settings() -> Settings:
    return Settings()
