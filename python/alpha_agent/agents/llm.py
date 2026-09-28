"""LLM transport for the runtime research layer (Phase 16).

The research agent is wired to Claude only now that every deterministic tool it
depends on exists. This module is the *only* place an external model client is
touched, and it is deliberately thin:

* :class:`LLMClient` is a structural protocol -- one ``complete`` call in, one
  :class:`LLMResponse` out. Nothing here knows what a ``HypothesisSpec`` is.
* :class:`AnthropicClient` is a lazy adapter. ``import anthropic`` happens inside
  the call, never at module import, so the core test suite runs with no API
  client installed (CLAUDE.md: "keep optional vendor imports lazy").
* :class:`ScriptedLLMClient` is the deterministic test double: canned responses
  in, recorded calls out. Every Phase 16 test uses it -- no network, ever.

An API key is never read from source. :class:`AnthropicClient` either takes a
key the caller already pulled from the environment, or lets the SDK read
``ANTHROPIC_API_KEY`` itself.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

#: Default research model. The exact id, not an alias.
DEFAULT_MODEL = "claude-sonnet-5"


class LLMClientUnavailable(RuntimeError):
    """The requested LLM client cannot be constructed (SDK not installed, or no
    credentials). Raised lazily, never at import time."""


@dataclass(frozen=True)
class LLMResponse:
    """One model completion, normalised across providers."""

    text: str
    model: str
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LLMClient(Protocol):
    """Minimal completion surface the research agent depends on."""

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse: ...


class AnthropicClient:
    """Lazy Anthropic adapter. Constructs nothing and imports nothing until the
    first :meth:`complete` call."""

    def __init__(self, *, api_key: str | None = None, client: object | None = None):
        # ``client`` is an already-built SDK client (used by advanced callers and
        # never by tests -- tests use ScriptedLLMClient). ``api_key`` is passed
        # through from the caller's environment read; it is not sourced here.
        self._api_key = api_key
        self._client = client

    def _ensure_client(self) -> object:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise LLMClientUnavailable(
                "the 'anthropic' package is not installed; install the 'ai' extra "
                "or inject an LLMClient explicitly"
            ) from exc
        self._client = (
            anthropic.Anthropic(api_key=self._api_key)
            if self._api_key
            else anthropic.Anthropic()
        )
        return self._client

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        client = self._ensure_client()
        # `temperature` is accepted on this Protocol (and recorded by
        # ScriptedLLMClient) for interface stability, but the installed
        # Anthropic SDK (1.x) no longer exposes a sampling parameter on
        # `messages.create()` for current-generation models -- passing it
        # raises a client-side TypeError before any request is sent. Current
        # models default to adaptive reasoning instead; `temperature` is
        # deliberately not forwarded.
        del temperature
        resp = client.messages.create(  # type: ignore[attr-defined]
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(resp, "content", [])
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            model=getattr(resp, "model", model),
            stop_reason=getattr(resp, "stop_reason", None),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


@dataclass
class ScriptedLLMClient:
    """Deterministic test double. Pops one canned response per ``complete`` call
    and records exactly what it was asked."""

    responses: Sequence[str | LLMResponse]
    model: str = "scripted-model"
    calls: list[dict] = field(default_factory=list)
    _cursor: int = 0

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "messages": [dict(m) for m in messages],
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self._cursor >= len(self.responses):
            raise AssertionError(
                f"ScriptedLLMClient exhausted after {self._cursor} response(s); "
                "the agent made more calls than the test scripted"
            )
        item = self.responses[self._cursor]
        self._cursor += 1
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(
            text=item,
            model=model,
            stop_reason="end_turn",
            output_tokens=max(1, len(item) // 4),
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)
