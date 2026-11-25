import hashlib
import hmac
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from fastapi import Header, HTTPException, Request
from benchmark.config import get_settings

ROLE_RANK = {"viewer": 1, "operator": 2, "admin": 3}


@dataclass(frozen=True)
class Principal:
    actor_id: str
    role: str


class RateLimiter:
    """In-memory sliding-window rate limiter keyed by actor_id."""

    def __init__(self) -> None:
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit_rpm: int) -> None:
        if limit_rpm <= 0:
            return
        now = time.monotonic()
        cutoff = now - 60.0
        window = self._windows[key]
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= limit_rpm:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        window.append(now)

    def reset(self) -> None:
        self._windows.clear()


rate_limiter = RateLimiter()


def _parse_api_keys(raw: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            key, role = item.split(":", 1)
            role = role.strip() if role.strip() in ROLE_RANK else "operator"
        else:
            key, role = item, "operator"
        if key.strip():
            entries.append((key.strip(), role))
    return entries


def _extract_token(x_api_key: Optional[str], authorization: Optional[str]) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def authenticate_request(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Principal:
    settings = get_settings()
    if not settings.api_auth_enabled:
        principal = Principal(actor_id="anonymous", role="admin")
        rate_limiter.check(principal.actor_id, settings.rate_limit_rpm)
        return principal

    token = _extract_token(x_api_key, authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")

    matched_role: Optional[str] = None
    for configured_key, role in _parse_api_keys(settings.api_keys):
        if hmac.compare_digest(token.encode("utf-8"), configured_key.encode("utf-8")):
            matched_role = role
            break

    if matched_role is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    key_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    principal = Principal(actor_id=f"key:{key_hash}", role=matched_role)
    rate_limiter.check(principal.actor_id, settings.rate_limit_rpm)
    return principal


def require_role(min_role: str):
    min_rank = ROLE_RANK[min_role]

    def _dependency(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
    ) -> Principal:
        principal = authenticate_request(request, x_api_key=x_api_key, authorization=authorization)
        if ROLE_RANK.get(principal.role, 0) < min_rank:
            raise HTTPException(status_code=403, detail=f"Requires {min_role} role")
        return principal

    return _dependency


def write_audit_event(
    action: str,
    principal: Principal,
    status_code: int,
    request: Optional[Request] = None,
    run_id: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    settings = get_settings()
    path = Path(settings.audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": time.time(),
        "action": action,
        "actor": principal.actor_id,
        "role": principal.role,
        "status_code": status_code,
        "run_id": run_id,
        "client_ip": request.client.host if request and request.client else None,
        "detail": detail,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
