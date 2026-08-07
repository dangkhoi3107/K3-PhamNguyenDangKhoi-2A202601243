"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id/user_id.

        Returns the request_id (generated if not supplied) so the same ID
        can be passed to record_output(), keeping one request traceable
        end-to-end.
        """
        request_id = request_id or str(uuid.uuid4())
        self._open[request_id] = time.monotonic()
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "direction": "input",
            "text": text,
            "layer": None,
            "blocked": False,
            "decision": None,
            "latency_ms": None,
            "timestamp": utc_now_iso(),
        })
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        decision: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs.

        Args:
            layer: which layer produced this outcome, e.g. "input_guardrail",
                "output_guardrail", "rate_limiter", "egress", "hitl".
            decision: reviewer/action decision, e.g. "auto_send", "approve",
                "reject", "timeout".
        """
        start = self._open.pop(request_id, None)
        latency_ms = (time.monotonic() - start) * 1000 if start is not None else None
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "direction": "output",
            "text": text,
            "layer": layer,
            "blocked": blocked,
            "decision": decision,
            "latency_ms": latency_ms,
            "timestamp": utc_now_iso(),
        })

    def find_by_request_id(self, request_id: str) -> list[dict]:
        """Return every log entry (input + output) tied to one request_id."""
        return [entry for entry in self.logs if entry["request_id"] == request_id]

    def export_json(self, filepath: str = "outputs/audit_log.json"):
        """Write logs to disk (JSON array)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
