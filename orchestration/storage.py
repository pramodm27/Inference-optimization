import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


class RunStore:
    """Durable SQLite-backed store for benchmark run lifecycle and summaries."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    workload_json TEXT NOT NULL,
                    aggregate_json TEXT,
                    result_path TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_created ON benchmark_runs(created_at DESC)"
            )

    def create_run(self, run_id: str, owner: str, workload: dict[str, Any]) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_runs (run_id, status, owner, workload_json, created_at, updated_at)
                VALUES (?, 'queued', ?, ?, ?, ?)
                """,
                (run_id, owner, json.dumps(workload), now, now),
            )

    def mark_running(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE benchmark_runs SET status = 'running', updated_at = ? WHERE run_id = ?",
                (time.time(), run_id),
            )

    def mark_completed(self, run_id: str, aggregate: dict[str, Any], result_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE benchmark_runs
                SET status = 'completed', aggregate_json = ?, result_path = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (json.dumps(aggregate), result_path, time.time(), run_id),
            )

    def mark_failed(self, run_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE benchmark_runs
                SET status = 'failed', error = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (error[:1000], time.time(), run_id),
            )

    def recover_stale_runs(self) -> int:
        """Mark leftover queued/running jobs from a previous process crash as failed."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE benchmark_runs
                SET status = 'failed', error = 'Interrupted by server restart', updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (time.time(),),
            )
            return cur.rowcount

    def count_active_runs(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM benchmark_runs WHERE status IN ('queued', 'running')"
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, status, owner, created_at, updated_at, error
                FROM benchmark_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM benchmark_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["workload"] = json.loads(data.pop("workload_json"))
        agg = data.pop("aggregate_json")
        data["aggregate"] = json.loads(agg) if agg else None
        return data
