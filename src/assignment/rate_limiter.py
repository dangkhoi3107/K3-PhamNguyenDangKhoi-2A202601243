"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.models import LlmResponse
from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        # See InputGuardrailPlugin in guardrails/input_guardrails.py:
        # on_user_message_callback can't halt the runner in this ADK
        # version — only before_model_callback's return value is checked
        # to skip the model call. This carries the decision across (single
        # request in flight per instance, so no cross-request race).
        self._pending_block: types.Content | None = None

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            self._pending_block = self._block_response(
                f"Rate limit exceeded. Try again in {wait:.0f}s."
            )
            return self._pending_block

        window.append(now)
        self._pending_block = None
        return None

    async def before_model_callback(self, *, callback_context, llm_request):
        """The actual hard gate — see InputGuardrailPlugin.before_model_callback
        for why on_user_message_callback alone can't stop the model call."""
        block, self._pending_block = self._pending_block, None
        if block is None:
            return None
        return LlmResponse(content=block)
