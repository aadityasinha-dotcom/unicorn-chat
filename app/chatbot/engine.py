"""Chat orchestration: sanitise -> build prompt -> tool loop -> answer / clarification.

One ``ChatEngine.chat()`` call == one user turn. The flow:

  1. sanitise the input (block / flag / ok)
  2. assemble messages = condensed history + current user message
  3. run the model; while it asks for tools, execute them and feed results back
     (max ``max_tool_rounds`` rounds); if it calls ``ask_clarifying_question``,
     stop and return that question
  4. record the turn in the session, log it, update metrics, close the trace
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from app.chatbot.llm import ChatModel, ModelResponse
from app.chatbot.prompts import FLAGGED_NOTE, build_system_prompt
from app.chatbot.session import Session, SessionStore, Turn
from app.chatbot.tools import CLARIFY_TOOL, TOOL_DEFINITIONS, ToolExecutor
from app.config import Settings
from app.data.repository import CompanyRepository
from app.logging_setup import CONVERSATION_LOGGER
from app.observability.metrics import Metrics
from app.observability.tracing import Tracer
from app.sanitize import Verdict, blocked_reply, sanitize

log = logging.getLogger(__name__)
conv_log = logging.getLogger(CONVERSATION_LOGGER)

ERROR_REPLY = "Sorry - I ran into a problem answering that. Please try again in a moment."
REFUSAL_REPLY = "I can't help with that request. I can answer questions about the Indian unicorn startups in my dataset."
TOOL_LIMIT_REPLY = "I couldn't finish gathering the data for that question. Could you narrow it down a little?"


@dataclass
class ChatResult:
    session_id: str
    reply: str
    kind: str  # answer | clarification | rejected | error
    latency_ms: float
    tool_calls: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    trace_id: str | None = None
    sanitization: dict | None = None


class ChatEngine:
    def __init__(
        self,
        repo: CompanyRepository,
        model: ChatModel,
        settings: Settings,
        sessions: SessionStore | None = None,
        metrics: Metrics | None = None,
        tracer: Tracer | None = None,
    ):
        self.repo = repo
        self.model = model
        self.settings = settings
        self.sessions = sessions or SessionStore()
        self.metrics = metrics or Metrics()
        self.tracer = tracer or Tracer(settings)
        self.tools = ToolExecutor(repo)
        self.system_prompt = build_system_prompt(repo)

    # ------------------------------------------------------------------ public
    def chat(self, message: str, session_id: str | None = None) -> ChatResult:
        started = time.perf_counter()
        session = self.sessions.get_or_create(session_id)
        self.metrics.inc("queries_total")

        clean = sanitize(message, self.settings.max_input_chars)
        san_info = {"verdict": clean.verdict.value, "reasons": clean.reasons, "original_length": clean.original_length}
        conv_log.info("user_query", extra={"event": "user_query", "session_id": session.id, "query": clean.text or message[:200], "sanitization": san_info})

        if clean.verdict is Verdict.BLOCKED:
            reply = blocked_reply(clean, self.settings.max_input_chars)
            self.metrics.inc("rejected_inputs_total", reason=clean.reasons[0].split(":")[0] if clean.reasons else "unknown")
            return self._finish(session, clean.text or message[:200], reply, "rejected", started, sanitization=san_info)

        with self.tracer.turn(session_id=session.id, user_input=clean.text, metadata={"sanitization": san_info}) as root:
            try:
                result = self._run_turn(session, clean.text, clean.verdict is Verdict.FLAGGED)
            except anthropic.APIStatusError as exc:
                log.error("anthropic api error", extra={"session_id": session.id, "status": exc.status_code, "message": exc.message})
                result = self._error_result(session, started, f"api_error_{exc.status_code}")
            except anthropic.APIConnectionError:
                log.exception("anthropic connection error", extra={"session_id": session.id})
                result = self._error_result(session, started, "connection_error")
            except anthropic.AnthropicError:
                # e.g. no credentials could be resolved - a deployment problem, not a user problem
                log.exception("anthropic sdk error", extra={"session_id": session.id})
                result = self._error_result(session, started, "sdk_error")
            except TypeError as exc:
                # The SDK raises TypeError when no credential (API key / auth token / profile) can be resolved.
                reason = "auth_not_configured" if "authentication" in str(exc).lower() else "unexpected"
                log.exception("chat turn failed", extra={"session_id": session.id, "reason": reason})
                result = self._error_result(session, started, reason)
            except Exception:
                log.exception("unexpected error in chat turn", extra={"session_id": session.id})
                result = self._error_result(session, started, "unexpected")
            reply, kind, tool_calls, usage = result
            self.tracer.update(root, output=reply, metadata={"kind": kind, "tool_calls": tool_calls, "usage": usage})
            self.tracer.score_turn("clarification_asked", 1.0 if kind == "clarification" else 0.0)
            if kind == "error":
                self.tracer.score_turn("error", 1.0)
            trace_id = self.tracer.current_trace_id()

        return self._finish(session, clean.text, reply, kind, started, tool_calls=tool_calls, usage=usage, trace_id=trace_id, sanitization=san_info)

    # ----------------------------------------------------------------- private
    def _run_turn(self, session: Session, user_text: str, flagged: bool) -> tuple[str, str, list[str], dict]:
        messages = session.history_messages(self.settings.max_history_turns)
        content = f"{FLAGGED_NOTE}\n\n<user_message>\n{user_text}\n</user_message>" if flagged else user_text
        messages.append({"role": "user", "content": content})
        if flagged:
            self.metrics.inc("flagged_inputs_total")

        tool_calls: list[str] = []
        usage: dict[str, int] = {"input": 0, "output": 0, "cache_read_input_tokens": 0}

        for _round in range(self.settings.max_tool_rounds + 1):
            response = self._call_model(messages)
            for k, v in response.usage.as_dict().items():
                usage[k] += v

            if response.stop_reason == "refusal":
                self.metrics.inc("refusals_total", category=str(response.refusal_category))
                return REFUSAL_REPLY, "answer", tool_calls, usage

            if response.stop_reason == "max_tokens":
                self.metrics.inc("truncated_responses_total")
                return (response.text or ERROR_REPLY), "answer", tool_calls, usage

            tool_uses = response.tool_uses
            if not tool_uses:
                return (response.text or ERROR_REPLY), "answer", tool_calls, usage

            # The model wants tools. First check for a clarification request - that ends the turn.
            clarify = next((t for t in tool_uses if t.name == CLARIFY_TOOL), None)
            if clarify is not None:
                question = str(clarify.input.get("question", "")).strip() or "Could you tell me a bit more about what you're looking for?"
                self.metrics.inc("clarifications_total")
                tool_calls.append(CLARIFY_TOOL)
                log.info("clarification requested", extra={"session_id": session.id, "reason": clarify.input.get("reason")})
                return question, "clarification", tool_calls, usage

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                tool_calls.append(tu.name)
                self.metrics.inc("tool_calls_total", tool=tu.name)
                with self.tracer.tool(name=tu.name, input=tu.input) as span:
                    output, is_error = self.tools.execute(tu.name, dict(tu.input))
                    self.tracer.update(span, output=output[:2000], level="ERROR" if is_error else "DEFAULT")
                if is_error:
                    self.metrics.inc("tool_errors_total", tool=tu.name)
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": output, "is_error": is_error})
            messages.append({"role": "user", "content": results})

        self.metrics.inc("tool_round_limit_total")
        return TOOL_LIMIT_REPLY, "answer", tool_calls, usage

    def _call_model(self, messages: list[dict]) -> ModelResponse:
        self.metrics.inc("llm_calls_total")
        params = getattr(self.model, "model_parameters", None)
        t0 = time.perf_counter()
        with self.tracer.generation(name="llm_call", model=self.model.model_name, input=_trace_safe(messages[-1]), model_parameters=params) as gen:
            response = self.model.create(system=self.system_prompt, messages=messages, tools=TOOL_DEFINITIONS)
            self.tracer.update(
                gen,
                output=_trace_safe({"stop_reason": response.stop_reason, "content": response.content}),
                usage_details=response.usage.as_dict(),
                metadata={"request_id": response.request_id, "served_by": response.model},
            )
        self.metrics.inc("llm_input_tokens_total", response.usage.input_tokens)
        self.metrics.inc("llm_output_tokens_total", response.usage.output_tokens)
        self.metrics.inc("llm_cache_read_tokens_total", response.usage.cache_read_input_tokens)
        log.debug("llm call", extra={"latency_ms": round((time.perf_counter() - t0) * 1000, 1), "stop_reason": response.stop_reason, "usage": response.usage.as_dict()})
        return response

    def _error_result(self, session: Session, started: float, reason: str) -> tuple[str, str, list[str], dict]:
        self.metrics.inc("errors_total", reason=reason)
        return ERROR_REPLY, "error", [], {}

    def _finish(self, session: Session, user_text: str, reply: str, kind: str, started: float, *, tool_calls=None, usage=None, trace_id=None, sanitization=None) -> ChatResult:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        self.metrics.observe_latency(latency_ms)
        self.metrics.inc("responses_total", kind=kind)

        session.turns.append(Turn("user", user_text))
        session.turns.append(Turn("assistant", reply, kind=kind))
        session.pending_clarification = reply if kind == "clarification" else None

        conv_log.info(
            "bot_response",
            extra={"event": "bot_response", "session_id": session.id, "kind": kind, "response": reply, "latency_ms": latency_ms, "tool_calls": tool_calls or [], "usage": usage or {}, "trace_id": trace_id},
        )
        return ChatResult(session.id, reply, kind, latency_ms, tool_calls or [], usage or {}, trace_id, sanitization)


def _trace_safe(obj: Any) -> Any:
    """Convert SDK content blocks to plain data for Langfuse."""
    if isinstance(obj, dict):
        return {k: _trace_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_trace_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float)):
        return {k: _trace_safe(v) for k, v in vars(obj).items()}
    return obj
