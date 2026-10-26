"""Langfuse tracing wrapper.

One Langfuse *trace* per chat turn (observation type ``agent``), containing
``generation`` observations for each LLM call (with model + token usage) and
``tool`` observations for each dataset tool execution. The session id is
propagated so Langfuse groups turns into conversations.

When Langfuse keys are not configured every method is a no-op, so the rest of
the code never has to check whether tracing is on.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

from app.config import Settings

log = logging.getLogger(__name__)


class Tracer:
    def __init__(self, settings: Settings):
        self._client = None
        if settings.langfuse_enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                log.info("langfuse tracing enabled", extra={"host": settings.langfuse_host})
            except Exception:  # pragma: no cover - defensive
                log.exception("failed to initialise langfuse; tracing disabled")
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextlib.contextmanager
    def turn(self, *, session_id: str, user_input: str, metadata: dict | None = None) -> Iterator[Any]:
        """Root observation for one chat turn. Yields the span (or None)."""
        if not self._client:
            yield None
            return
        from langfuse import propagate_attributes

        with propagate_attributes(session_id=session_id, tags=["unicorn-chat"]):
            with self._client.start_as_current_observation(
                name="chat_turn", as_type="agent", input=user_input, metadata=metadata or {}
            ) as span:
                yield span

    @contextlib.contextmanager
    def generation(self, *, name: str, model: str, input: Any, model_parameters: dict | None = None) -> Iterator[Any]:
        if not self._client:
            yield None
            return
        with self._client.start_as_current_observation(
            name=name, as_type="generation", model=model, input=input, model_parameters=model_parameters
        ) as gen:
            yield gen

    @contextlib.contextmanager
    def tool(self, *, name: str, input: Any) -> Iterator[Any]:
        if not self._client:
            yield None
            return
        with self._client.start_as_current_observation(name=f"tool:{name}", as_type="tool", input=input) as span:
            yield span

    @staticmethod
    def update(span: Any, **kwargs: Any) -> None:
        if span is not None:
            try:
                span.update(**kwargs)
            except Exception:  # pragma: no cover
                log.debug("langfuse span update failed", exc_info=True)

    def score_turn(self, name: str, value: float | str, comment: str | None = None) -> None:
        if self._client:
            try:
                self._client.score_current_trace(name=name, value=value, comment=comment)
            except Exception:  # pragma: no cover
                log.debug("langfuse score failed", exc_info=True)

    def current_trace_id(self) -> str | None:
        if not self._client:
            return None
        try:
            return self._client.get_current_trace_id()
        except Exception:  # pragma: no cover
            return None

    def flush(self) -> None:
        if self._client:
            self._client.flush()

    def shutdown(self) -> None:
        if self._client:
            self._client.shutdown()
