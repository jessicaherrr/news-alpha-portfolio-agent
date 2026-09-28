"""Alpha Discovery live-research campaign, Checkpoints 5-7 -- shared support
for the REAL, read-only external-source connectors (GitHub / academic /
community / practitioner).

Everything here is deliberately boring and mechanical, on purpose (task spec
Part 4 section 43: "mechanism identity" -- and by extension every other
extraction step -- must be deterministic and auditable, never LLM judgment):

* :func:`classify_mechanism` is a fixed keyword-vote classifier over the
  SAME closed :class:`~alpha_agent.knowledge.models.EconomicMechanism`
  catalog every other source already uses. It returns ``None`` (never a
  guess) when no category has a real signal -- callers must skip such a
  source rather than fabricate a mechanism label.
* :func:`http_get_json` / :func:`http_get_text` are the ONLY network
  primitives every connector uses -- one bounded timeout, one retry on a
  transient 5xx/timeout, and NEVER a repository's own code executed,
  imported, or shelled out to (see ``alpha_agent.knowledge.external_sources.
  FORBIDDEN_OPERATIONS``, which every connector module is grepped against by
  the same static safety test).
* :func:`load_dotenv_if_available` mirrors ``scripts/databento_probe.py``'s
  own helper -- a local ``.env`` may carry ``GITHUB_TOKEN`` (never committed,
  never printed, never logged); this module never reads the raw file itself,
  it only asks ``python-dotenv`` to populate ``os.environ``.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from alpha_agent.knowledge.models import EconomicMechanism

__all__ = [
    "CROSSREF_API",
    "MECHANISM_KEYWORDS",
    "OPENALEX_API",
    "POLITE_CONTACT",
    "ConnectorBudget",
    "ConnectorHttpError",
    "classify_mechanism",
    "dedup_key",
    "http_get_json",
    "http_get_text",
    "load_dotenv_if_available",
    "normalize_title",
    "now_iso",
    "provenance_hash",
    "truncate",
]

#: The two free, keyless bibliographic providers `academic_connector.py` AND
#: `practitioner_connector.py` both use -- declared once here so the two
#: modules never drift on the endpoint URL.
OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"
#: OpenAlex asks polite-pool users to identify themselves via `mailto` -- a
#: generic, non-personal contact is fine and never a secret.
POLITE_CONTACT = "research-bot@example.invalid"

_NORMALIZE_TITLE_RE = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    return _NORMALIZE_TITLE_RE.sub(" ", title.lower()).strip()


def dedup_key(*, doi: str | None, title: str) -> str:
    """Task spec "ACADEMIC DEDUPLICATION" -- the same paper found by two
    providers collapses to one key, by DOI when available else normalized
    title."""
    if doi:
        return f"doi:{doi.lower().strip()}"
    return f"title:{normalize_title(title)}"


def load_dotenv_if_available() -> None:
    """Best-effort local `.env` load -- never raises if `python-dotenv` is
    absent, never reads or logs the file's contents itself. Safe to call
    unconditionally at connector construction time (idempotent)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def provenance_hash(prefix: str, payload: dict[str, Any]) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:32]


def truncate(text: str, limit: int = 4000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


@dataclass(frozen=True)
class ConnectorBudget:
    """Task spec "NETWORK BUDGET" -- every live connector is bounded, never an
    uncontrolled crawl. Defaults match the task spec's own recommendations."""

    max_search_results: int = 10
    max_queries: int = 6
    max_files_per_repo: int = 3
    max_documents_total: int = 20
    timeout_seconds: float = 10.0


class ConnectorHttpError(RuntimeError):
    """A real HTTP call failed. Carries enough for the caller to classify the
    outcome (ERROR vs RATE_LIMITED) -- never a raw traceback to the UI."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def http_get_json(
    url: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[dict, dict[str, str]]:
    """One bounded GET, JSON body expected. Returns ``(body, response_headers)``
    -- the caller reads rate-limit headers off the second element rather than
    a second network call. Raises :class:`ConnectorHttpError` on any non-2xx
    or transport failure; never retries silently more than once on a 5xx."""
    import httpx

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = httpx.get(url, headers=headers, params=params, timeout=timeout)
        except httpx.TimeoutException as exc:
            last_exc = exc
            continue
        except httpx.HTTPError as exc:
            raise ConnectorHttpError(f"transport error: {exc}") from exc
        if resp.status_code >= 500 and attempt == 0:
            last_exc = ConnectorHttpError(f"HTTP {resp.status_code}", status_code=resp.status_code)
            continue
        if resp.status_code >= 400:
            raise ConnectorHttpError(
                f"HTTP {resp.status_code} from {url.split('?')[0]}", status_code=resp.status_code
            )
        try:
            return resp.json(), dict(resp.headers)
        except ValueError as exc:
            raise ConnectorHttpError(f"non-JSON response from {url.split('?')[0]}") from exc
    raise ConnectorHttpError(f"request to {url.split('?')[0]} failed after retry: {last_exc}")


def http_get_text(
    url: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None,
    timeout: float = 10.0, max_bytes: int = 500_000,
) -> str:
    """One bounded GET, TEXT body (never parsed as anything executable --
    this is DATA, per the module docstring). Truncates to `max_bytes`."""
    import httpx

    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ConnectorHttpError(f"transport error: {exc}") from exc
    if resp.status_code >= 400:
        raise ConnectorHttpError(
            f"HTTP {resp.status_code} from {url.split('?')[0]}", status_code=resp.status_code
        )
    content = resp.content[:max_bytes]
    return content.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# deterministic mechanism classification -- the SAME closed EconomicMechanism
# catalog, keyword-vote only, never an LLM judgment call.
# ---------------------------------------------------------------------------

#: Ordered so a more SPECIFIC category is checked before a broader relative --
#: e.g. VOLATILITY_BREAKOUT before BREAKOUT, REGIME_CONDITIONED_* before the
#: plain TREND/MEAN_REVERSION they specialize. Order is the tie-break when two
#: categories score an equal number of keyword hits.
MECHANISM_KEYWORDS: tuple[tuple[EconomicMechanism, tuple[str, ...]], ...] = (
    (EconomicMechanism.REGIME_CONDITIONED_TREND, (
        "regime conditioned trend", "regime-conditioned trend", "trend regime", "volatility conditioned trend",
        "volatility-conditioned trend",
    )),
    (EconomicMechanism.REGIME_CONDITIONED_MEAN_REVERSION, (
        "regime conditioned mean reversion", "regime-conditioned mean reversion", "mean reversion regime",
    )),
    (EconomicMechanism.HYBRID_TREND_REVERSAL, (
        "trend reversal", "hybrid strategy", "trend and reversal", "trend/reversal",
    )),
    (EconomicMechanism.ML_META_LABELING, (
        "meta-labeling", "meta labeling", "meta-labelling", "triple barrier", "machine learning strategy",
        "ml model", "random forest strategy", "gradient boosting strategy",
    )),
    (EconomicMechanism.MULTI_SIGNAL_ENSEMBLE, (
        "ensemble strategy", "multi-factor", "multi factor", "combine signals", "signal ensemble",
    )),
    (EconomicMechanism.VOLATILITY_BREAKOUT, (
        "volatility breakout", "vol breakout", "atr breakout", "range expansion breakout",
    )),
    (EconomicMechanism.VOLATILITY_TRANSITION, (
        "volatility regime", "vol regime", "volatility transition", "volatility targeting", "vol targeting",
        "volatility percentile",
    )),
    (EconomicMechanism.OPENING_RANGE, (
        "opening range", "orb strategy", "opening range breakout", "first hour range",
    )),
    (EconomicMechanism.SESSION_EFFECTS, (
        "session effect", "time of day effect", "intraday seasonality", "session-based",
    )),
    (EconomicMechanism.OVERNIGHT_GAP, (
        "overnight gap", "overnight return", "gap trading", "close to open", "close-to-open",
    )),
    (EconomicMechanism.FAILED_BREAKOUT, (
        "failed breakout", "false breakout", "fakeout", "fake breakout",
    )),
    (EconomicMechanism.BREAKOUT, (
        "breakout strategy", "donchian", "channel breakout", "range breakout", "price breakout",
    )),
    (EconomicMechanism.TERM_STRUCTURE, (
        "term structure", "futures curve", "contango", "backwardation", "roll yield",
    )),
    (EconomicMechanism.CARRY, (
        "carry strategy", "carry trade", "carry factor", "futures carry",
    )),
    (EconomicMechanism.CROSS_MARKET_LEAD_LAG, (
        "lead-lag", "lead lag", "cross-asset momentum", "cross asset momentum", "cross-market",
    )),
    (EconomicMechanism.RELATIVE_VALUE, (
        "relative value", "pairs trading", "spread trading", "statistical arbitrage", "stat arb",
    )),
    (EconomicMechanism.CORRELATION_SPREAD, (
        "correlation strategy", "cointegration", "correlation spread",
    )),
    (EconomicMechanism.VOLUME_LIQUIDITY, (
        "volume strategy", "liquidity strategy", "amihud", "volume spike", "order flow",
    )),
    (EconomicMechanism.MEAN_REVERSION, (
        "mean reversion", "mean-reversion", "z-score", "zscore", "bollinger band", "oversold", "overbought",
        "reversion to the mean",
    )),
    (EconomicMechanism.MOMENTUM, (
        "time series momentum", "time-series momentum", "tsmom", "price momentum", "momentum strategy",
        "momentum factor",
    )),
    (EconomicMechanism.TREND, (
        "trend following", "moving average crossover", "ma crossover", "dual moving average",
        "golden cross", "trend-following",
    )),
)

_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def _pattern_for(keyword: str) -> re.Pattern[str]:
    pat = _WORD_BOUNDARY_CACHE.get(keyword)
    if pat is None:
        pat = re.compile(re.escape(keyword))
        _WORD_BOUNDARY_CACHE[keyword] = pat
    return pat


def classify_mechanism(text: str) -> EconomicMechanism | None:
    """Deterministic keyword-vote classification into the closed
    `EconomicMechanism` catalog. `None` when nothing matches -- the caller
    must skip the source rather than invent a mechanism (task spec: "Do not
    hallucinate fields")."""
    if not text:
        return None
    lowered = text.lower()
    best: EconomicMechanism | None = None
    best_hits = 0
    for mechanism, keywords in MECHANISM_KEYWORDS:
        hits = sum(1 for kw in keywords if _pattern_for(kw).search(lowered))
        if hits > best_hits:
            best_hits = hits
            best = mechanism
    return best if best_hits > 0 else None
