"""Bounded, read-only HTTP GET -- the ONLY network primitive every
market-news/event connector in this package uses. Deliberately self-
contained (not shared with `alpha_agent.knowledge.connector_support`, which
serves the separate ResearchSource/knowledge-base plane -- Section 15/31:
these two planes never share a store, and keeping their network glue
separate too means neither can accidentally couple to the other's retry/
error semantics). One bounded timeout, one retry on a transient 5xx/timeout,
identical shape to the sanctioned pattern already used elsewhere in this
codebase.
"""
from __future__ import annotations

from typing import Any


class MarketIntelHttpError(RuntimeError):
    """A real HTTP call failed -- callers classify this into a typed
    connector health state, never crash the page."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


#: A real, honest identifying User-Agent -- some official government feeds
#: (verified live: BLS) reject default HTTP-client user agents outright as
#: bot traffic; this is never spoofed to impersonate a browser.
_USER_AGENT = "AlphaAgentResearch/1.0 (market-intelligence observation; contact: research-bot@example.invalid)"


def http_get_text(url: str, *, timeout: float = 10.0, max_bytes: int = 2_000_000) -> str:
    """One bounded GET, text body (RSS/Atom/HTML) -- never executed as
    anything, always treated as data. Raises `MarketIntelHttpError` on any
    non-2xx or transport failure after one retry on a transient condition."""
    import httpx

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = httpx.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT}, follow_redirects=True)
        except httpx.TimeoutException as exc:
            last_exc = exc
            continue
        except httpx.HTTPError as exc:
            raise MarketIntelHttpError(f"transport error: {exc}") from exc
        if resp.status_code >= 500 and attempt == 0:
            last_exc = MarketIntelHttpError(f"HTTP {resp.status_code}", status_code=resp.status_code)
            continue
        if resp.status_code >= 400:
            raise MarketIntelHttpError(f"HTTP {resp.status_code} from {url.split('?')[0]}", status_code=resp.status_code)
        return resp.content[:max_bytes].decode("utf-8", errors="replace")
    raise MarketIntelHttpError(f"request to {url.split('?')[0]} failed after retry: {last_exc}")


def http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    """One bounded GET, JSON body expected -- used by the OPTIONAL,
    key-gated USDA MyMarketNews connector."""
    import httpx

    merged = {"User-Agent": _USER_AGENT, **(headers or {})}
    try:
        resp = httpx.get(url, timeout=timeout, headers=merged, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise MarketIntelHttpError(f"transport error: {exc}") from exc
    if resp.status_code >= 400:
        raise MarketIntelHttpError(f"HTTP {resp.status_code} from {url.split('?')[0]}", status_code=resp.status_code)
    try:
        return resp.json()
    except ValueError as exc:
        raise MarketIntelHttpError(f"non-JSON response from {url.split('?')[0]}") from exc


def http_head(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """A bounded HEAD request -- used only to read a real server-reported
    ``Last-Modified``/status, never to download a body (Section 18's USDA
    "public official publication source" mode)."""
    import httpx

    try:
        resp = httpx.head(url, timeout=timeout, headers={"User-Agent": _USER_AGENT}, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise MarketIntelHttpError(f"transport error: {exc}") from exc
    if resp.status_code >= 400:
        raise MarketIntelHttpError(f"HTTP {resp.status_code} from {url.split('?')[0]}", status_code=resp.status_code)
    return dict(resp.headers)


__all__ = ["MarketIntelHttpError", "http_get_json", "http_get_text", "http_head"]
