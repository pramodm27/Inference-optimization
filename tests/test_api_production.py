import json
from fastapi.testclient import TestClient
from api import main as api_main
from api.security import rate_limiter
from benchmark.config import get_settings
from orchestration.storage import RunStore


def _configure_env(monkeypatch, tmp_path, rate_limit_rpm=60, max_concurrent_runs=4):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "results" / "runs.db"))
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "results" / "audit.jsonl"))
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEYS", "admin-secret:admin,op-secret:operator,view-secret:viewer")
    monkeypatch.setenv("RATE_LIMIT_RPM", str(rate_limit_rpm))
    monkeypatch.setenv("MAX_CONCURRENT_RUNS", str(max_concurrent_runs))
    get_settings.cache_clear()
    rate_limiter.reset()
    api_main._store = None


def test_auth_rbac_and_audit_log(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)

    async def fake_execute(request, run_id=None):
        return {
            "run_id": run_id,
            "aggregate": {"error_rate": 0.0, "output_tokens_per_second": 100.0},
            "diagnosis": {"type": "heuristic_hypotheses_not_ground_truth", "hypotheses": []},
        }

    monkeypatch.setattr(api_main, "execute", fake_execute)

    with TestClient(api_main.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        # Missing key -> 401
        assert client.get("/v1/benchmarks").status_code == 401
        # Invalid key -> 401
        assert client.get("/v1/benchmarks", headers={"X-API-Key": "wrong"}).status_code == 401
        # Viewer cannot POST -> 403
        resp = client.post(
            "/v1/benchmarks",
            headers={"X-API-Key": "view-secret"},
            json={"requests": 4, "concurrency": 2, "warmup": 0, "trials": 1},
        )
        assert resp.status_code == 403
        # Operator can POST -> 202
        resp = client.post(
            "/v1/benchmarks",
            headers={"Authorization": "Bearer op-secret"},
            json={"requests": 4, "concurrency": 2, "warmup": 0, "trials": 1},
        )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        # Viewer can list & get
        assert len(client.get("/v1/benchmarks", headers={"X-API-Key": "view-secret"}).json()) == 1
        detail = client.get(f"/v1/benchmarks/{run_id}", headers={"X-API-Key": "view-secret"})
        assert detail.status_code == 200
        assert detail.json()["run_id"] == run_id

    audit_lines = (tmp_path / "results" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) >= 3
    first = json.loads(audit_lines[0])
    assert "actor" in first and "action" in first


def test_rate_limit_and_stale_recovery(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path, rate_limit_rpm=2)
    store = RunStore(str(tmp_path / "results" / "runs.db"))
    store.create_run("11111111-1111-1111-1111-111111111111", "key:test", {"requests": 2})

    with TestClient(api_main.app) as client:
        recovered = store.get_run("11111111-1111-1111-1111-111111111111")
        assert recovered["status"] == "failed"
        assert "restart" in recovered["error"].lower()

        assert client.get("/v1/benchmarks", headers={"X-API-Key": "view-secret"}).status_code == 200
        assert client.get("/v1/benchmarks", headers={"X-API-Key": "view-secret"}).status_code == 200
        assert client.get("/v1/benchmarks", headers={"X-API-Key": "view-secret"}).status_code == 429
