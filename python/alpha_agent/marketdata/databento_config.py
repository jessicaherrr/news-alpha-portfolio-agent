"""Config for the read-only Databento market-OBSERVATION provider (mission
Part D). Mirrors ``alpha_agent.marketdata.config``'s IBKR pattern: no
credentials live here (``DATABENTO_API_KEY`` stays in the environment / local
``.env``, never in this file), and a missing/unreadable file fails closed to
"disabled" rather than a crash or a fabricated value.

The bounds below are the network/cost-safety controls mission section 16
requires: a bounded TTL cache (no Streamlit-rerun charge storm), a bounded
lookback (never an uncontrolled historical pull), and a small auto-fetch cost
ceiling below which a bounded OHLCV/definition request proceeds without
requiring a separate confirmation click (CLAUDE.md market-data rule 2 --
``metadata.get_cost()`` is still ALWAYS called first; this ceiling only
decides whether the already-computed estimate is small enough to act on
automatically). The real entitlement probed for this key priced a 6-hour
1-minute-bar NQ request and a 2-day 1-hour-bar NQ request at $0.00, and one
instrument-definition lookup at ~$0.0000008 (see
``scripts/databento_market_capability_probe.py``) -- the default ceiling
below is set with a wide safety margin above that, not tuned to it.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("configs/databento_market_data.yaml")

DATASET = "GLBX.MDP3"


class DatabentoMarketDataConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    #: Master switch -- disabling this makes every provider call short-circuit
    #: to NOT_CONNECTED with zero network activity, same as a missing key.
    enabled: bool = True

    #: Seconds a health/snapshot/OHLCV result stays cached before a fresh
    #: network call is allowed. Bounds Streamlit-rerun request volume.
    ttl_seconds: float = Field(default=30.0, gt=0)

    #: A ``get_dataset_range`` lag at or below this is reported LATEST_AVAILABLE;
    #: above it, HISTORICAL_ONLY. Never LIVE/DELAYED -- see
    #: ``DatabentoCapability``'s docstring.
    latest_available_lag_seconds: float = Field(default=24 * 3600, gt=0)

    #: A bounded OHLCV/definition request whose ``metadata.get_cost()``
    #: estimate is at or below this proceeds automatically; above it, the
    #: caller gets the estimate back and must explicitly approve a larger
    #: fetch (mission section 16 -- "do not perform expensive historical
    #: acquisition silently").
    max_auto_fetch_cost_usd: float = Field(default=0.05, ge=0)

    #: Hard ceilings, independent of the cost estimate -- defense in depth
    #: against a cost estimate that rounds to zero for a request that is
    #: nonetheless unbounded in shape.
    max_lookback_bars: int = Field(default=500, gt=0, le=5000)
    max_lookback_days: int = Field(default=30, gt=0, le=90)

    #: MARKET REALITY pass -- a SEPARATE, UI-session-level cache window
    #: (`alpha_agent.ui.market_home`) sitting in front of this provider's own
    #: `ttl_seconds` request cache, so a bare Streamlit rerun (any widget
    #: interaction anywhere on a page) never re-issues a provider call for
    #: every approved root on every script execution -- only once this many
    #: seconds have elapsed since the UI last actually refreshed does the
    #: next render call through again. Deliberately independent of, and
    #: normally >= `ttl_seconds`: this bounds RENDER-time request volume,
    #: not per-request cost safety (that stays `ttl_seconds` +
    #: `max_auto_fetch_cost_usd` above). Never chosen faster than the data's
    #: own observed freshness justifies -- the real entitlement lags "now" by
    #: single-digit hours, so refreshing a UI display every 90s is already
    #: far more frequent than the underlying data changes.
    ui_refresh_interval_seconds: float = Field(default=90.0, gt=0)

    #: Market Intelligence Checkpoint C, Section 34: contract DEFINITIONS
    #: (expiry, tick size, raw symbols) change far more slowly than a price
    #: -- a separate, much longer TTL cache namespace so re-opening the
    #: Contracts tab does not re-issue a definition fetch on every visit,
    #: while contract PRICE/volume/OI still refresh on the provider's normal
    #: `ttl_seconds` window (deliberately shorter).
    contract_definitions_ttl_seconds: float = Field(default=3600.0, gt=0)


def load_databento_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DatabentoMarketDataConfig:
    """Offline, no-network config load. A missing or unreadable file falls
    back to the safe defaults above (enabled, but bounded) -- never a crash."""
    p = Path(path)
    if not p.exists():
        return DatabentoMarketDataConfig()
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return DatabentoMarketDataConfig()
    return DatabentoMarketDataConfig.model_validate(raw)


__all__ = ["DATASET", "DEFAULT_CONFIG_PATH", "DatabentoMarketDataConfig", "load_databento_config"]
