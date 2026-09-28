"""Alpha Discovery campaign, Part I -- experiment-bound execution artifacts
(task spec sections 48-50).

For NEW research runs, persist real, experiment-bound artifacts (fills,
trades, daily equity / drawdown path) so Part J's investor visualizations
have something authentic to render. This module NEVER reconstructs a fill,
a trade, or an equity point -- it only copies the already-written, C++-
produced CSV exports (`alpha_agent.validation.runner.CliBacktestRunner`'s
new optional `trades_out_path` / `fills_out_path`, Phase 15A's additive CLI
flags) and the official `daily_equity` trace already aggregated into a
`DailyReturnSeries` (`alpha_agent.validation.returns`) into a permanent,
hash-verified location, and writes one small manifest recording what it did.

An OLD experiment that was never run with an artifact directory bound stays
"PATH DATA NOT AVAILABLE" (task spec section 50) -- this module has no
retroactive path: it is only ever called at the moment a NEW trial executes,
never after the fact, and it never fabricates a plausible-looking equity
curve from a summary Net PnL / Sharpe / trade count.

Artifacts live under `data/discovery/artifacts/` (gitignored, like
`data/paper_trading/` and this campaign's own `data/discovery/freezes/`) --
operational output, never source or registry truth. `ArtifactBundle`'s own
`bundle_sha256` is what `TrialEvidence.source_artifact_sha256` (an ALREADY
EXISTING, previously always-`None`, frozen `TrialEvidence` field -- no
schema change) is set to; `TrialEvidence.source_artifact` is the manifest
file's path. Both fields are populated only when real data was actually
written -- never a synthetic placeholder.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from alpha_agent.validation.returns import DailyReturnSeries

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "data" / "discovery" / "artifacts"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactFile(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: str  # "fills" | "trades" | "daily_equity"
    path: str  # absolute path, this checkout
    sha256: str
    n_rows: int


class ArtifactBundle(BaseModel):
    """Everything one execution attempt actually produced. Any of
    `fills`/`trades` may be `None` -- the C++ CLI's export flags are optional
    and a caller that did not request them leaves that artifact honestly
    absent, never fabricated. `daily_equity` is present whenever the run
    itself produced ANY daily trace (`DailyReturnSeries.n_days > 0`)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "experiment-artifact-bundle/1"
    experiment_identity: str
    attempt_ordinal: int
    strategy_fingerprint: str
    fills: ArtifactFile | None = None
    trades: ArtifactFile | None = None
    daily_equity: ArtifactFile | None = None
    #: Alpha Discovery live-research campaign, Checkpoint 11 -- the REAL
    #: executed price bars (raw contract OHLCV, exactly the frame the C++
    #: engine ran against) covering the headline OOS window, so a future
    #: Price + Position chart can align price to the real `fills`/`trades`
    #: timestamps without ever reconstructing a synthetic bar. `None` for
    #: every caller that predates this field or did not request it -- an
    #: additive field, never required.
    price_bars: ArtifactFile | None = None
    bundle_sha256: str = ""

    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("bundle_sha256", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _write_daily_equity_csv(daily: DailyReturnSeries, path: Path) -> int:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "session_day_index", "trading_day", "ts_ns", "equity_usd",
            "daily_pnl_usd", "return", "fills_cumulative",
        ])
        for i in range(daily.n_days):
            writer.writerow([
                daily.session_day_index[i], daily.trading_day[i], daily.ts_ns[i],
                daily.equity_usd[i], daily.daily_pnl_usd[i], daily.returns[i],
                daily.fills_cumulative[i],
            ])
    return daily.n_days


def _write_price_bars_csv(bars, path: Path) -> int:
    """`bars` is the exact executed-price DataFrame the engine ran against
    (`ts_event_ns, instrument_id, open, high, low, close, volume`) -- copied
    verbatim, never resampled, reordered, or reconstructed."""
    cols = [c for c in ("ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume") if c in bars.columns]
    bars[cols].to_csv(path, index=False)
    return len(bars)


def _count_csv_rows(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        return max(0, sum(1 for _ in fh) - 1)  # minus header


def persist_run_artifacts(
    *,
    experiment_identity: str,
    attempt_ordinal: int,
    strategy_fingerprint: str,
    daily: DailyReturnSeries | None = None,
    fills_csv_path: str | Path | None = None,
    trades_csv_path: str | Path | None = None,
    price_bars: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> ArtifactBundle:
    """Copy whichever of `fills_csv_path` / `trades_csv_path` actually exist
    (the CLI only writes them when asked, and only when it had at least one
    fill/trade to report) into the permanent artifact directory, write the
    real `daily` equity trace alongside them, and return the typed bundle.

    `price_bars` (Checkpoint 11, additive, optional): the exact executed-
    price DataFrame the engine ran against for the headline window -- copied
    verbatim (never resampled or reconstructed) so a future Price + Position
    chart can align real price to the real `fills`/`trades` timestamps.
    """
    out_dir = out_dir or DEFAULT_ARTIFACT_DIR
    safe_id = experiment_identity.replace(":", "_").replace("/", "_")
    dest_dir = out_dir / safe_id / f"attempt_{attempt_ordinal}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    fills_file = trades_file = equity_file = price_bars_file = None
    if fills_csv_path is not None and Path(fills_csv_path).exists():
        dest = dest_dir / "fills.csv"
        shutil.copyfile(fills_csv_path, dest)
        fills_file = ArtifactFile(
            kind="fills", path=str(dest), sha256=_sha256_file(dest), n_rows=_count_csv_rows(dest),
        )
    if trades_csv_path is not None and Path(trades_csv_path).exists():
        dest = dest_dir / "trades.csv"
        shutil.copyfile(trades_csv_path, dest)
        trades_file = ArtifactFile(
            kind="trades", path=str(dest), sha256=_sha256_file(dest), n_rows=_count_csv_rows(dest),
        )
    if daily is not None and daily.n_days > 0:
        dest = dest_dir / "daily_equity.csv"
        n = _write_daily_equity_csv(daily, dest)
        equity_file = ArtifactFile(kind="daily_equity", path=str(dest), sha256=_sha256_file(dest), n_rows=n)
    if price_bars is not None and len(price_bars) > 0:
        dest = dest_dir / "price_bars.csv"
        n = _write_price_bars_csv(price_bars, dest)
        price_bars_file = ArtifactFile(kind="price_bars", path=str(dest), sha256=_sha256_file(dest), n_rows=n)

    bundle = ArtifactBundle(
        experiment_identity=experiment_identity, attempt_ordinal=attempt_ordinal,
        strategy_fingerprint=strategy_fingerprint, fills=fills_file, trades=trades_file,
        daily_equity=equity_file, price_bars=price_bars_file,
    )
    bundle_sha = hashlib.sha256(bundle.canonical_json().encode("utf-8")).hexdigest()
    bundle = bundle.model_copy(update={"bundle_sha256": bundle_sha})

    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    return bundle


def manifest_path_for(bundle: ArtifactBundle, *, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or DEFAULT_ARTIFACT_DIR
    safe_id = bundle.experiment_identity.replace(":", "_").replace("/", "_")
    return out_dir / safe_id / f"attempt_{bundle.attempt_ordinal}" / "manifest.json"


def read_artifact_bundle(manifest_path: str | Path) -> ArtifactBundle:
    return ArtifactBundle.model_validate(json.loads(Path(manifest_path).read_text(encoding="utf-8")))
