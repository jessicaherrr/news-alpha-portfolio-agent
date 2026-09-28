"""Connection settings for the optional delayed market-data feed.

Deliberately carries NO credentials -- IBKR TWS API authentication happens
entirely inside a locally-running Trader Workstation / IB Gateway; this
config only tells the adapter where to find it. Never put a username,
password, or API token in `configs/market_data.yaml` or anywhere in this
repo (CLAUDE.md: "Never store API keys in source files, prompts, logs, or
test fixtures" -- the same rule, applied here to a local-only connection
that has no key to store in the first place).
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG_PATH = Path("configs/market_data.yaml")


class IBKRConnectionConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 42
    connect_timeout_seconds: float = 1.5
    request_timeout_seconds: float = 3.0


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> IBKRConnectionConfig:
    """Offline, no-network config load. A missing or unreadable file is
    treated as "disabled" (fails closed to no market context, never to a
    fabricated quote or a crash)."""
    p = Path(path)
    if not p.exists():
        return IBKRConnectionConfig(enabled=False)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return IBKRConnectionConfig(enabled=False)
    return IBKRConnectionConfig.model_validate(raw)
