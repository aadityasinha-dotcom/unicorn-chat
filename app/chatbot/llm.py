"""Thin adapter around the Anthropic SDK.

``ChatModel`` is the interface the engine depends on; ``AnthropicChatModel`` is
the production implementation and tests inject a scripted fake. Keeping this
boundary small is what lets the orchestration be unit-tested without network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"input": self.input_tokens, "output": self.output_tokens, "cache_read_input_tokens": self.cache_read_input_tokens}


@dataclass
class ModelResponse:
    stop_reason: str
    content: list[Any]  # SDK content blocks (type: text | tool_use | thinking | ...)
    usage: Usage
    model: str
    request_id: str | None = None
    refusal_category: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if getattr(b, "type", None) == "text").strip()

    @property
    def tool_uses(self) -> list[Any]:
        return [b for b in self.content if getattr(b, "type", None) == "tool_use"]


class ChatModel(Protocol):
    model_name: str

    def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse: ...


class AnthropicChatModel:
    """Calls the Messages API with prompt caching, adaptive thinking and
    (optionally) server-side refusal fallbacks."""

    def __init__(self, settings: Settings):
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model_name = settings.anthropic_model
        self._max_tokens = settings.anthropic_max_tokens
        self._effort = settings.anthropic_effort
        self._fallbacks = settings.anthropic_enable_fallbacks

    @property
    def model_parameters(self) -> dict:
        return {"effort": self._effort, "max_tokens": self._max_tokens, "fallbacks": "default" if self._fallbacks else "off"}

    def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model_name,
            max_tokens=self._max_tokens,
            # Cache the static prefix (instructions + catalogue); the breakpoint sits after the system block.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
        )
        if self._fallbacks:
            # Server-side fallback: on a safety-classifier refusal the API re-runs the request on a fallback model.
            resp = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            resp = self._client.messages.create(**kwargs)

        usage = Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_input_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        )
        refusal_category = None
        if resp.stop_reason == "refusal" and getattr(resp, "stop_details", None):
            refusal_category = getattr(resp.stop_details, "category", None)
        return ModelResponse(
            stop_reason=resp.stop_reason or "end_turn",
            content=list(resp.content),
            usage=usage,
            model=resp.model,
            request_id=getattr(resp, "_request_id", None),
            refusal_category=refusal_category,
        )
