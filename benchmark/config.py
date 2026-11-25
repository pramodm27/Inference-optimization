from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_model: str = "replace-with-served-model-name"
    vllm_api_key: str = "replace-me"
    results_dir: str = "results"
    benchmark_timeout_seconds: float = 180.0

    # Production control-plane settings
    database_path: str = "results/runs.db"
    audit_log_path: str = "results/audit.jsonl"
    api_auth_enabled: bool = True
    # Comma-separated key:role pairs, e.g. "secret-admin:admin,secret-op:operator,secret-view:viewer"
    api_keys: str = "dev-admin-key:admin"
    rate_limit_rpm: int = 60
    max_concurrent_runs: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
