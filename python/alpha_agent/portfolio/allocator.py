"""News Alpha Phase G -- the bridge to the C++ allocator (`quant::construct_portfolio`).

Python prepares the inputs (selected signals, risk statistics, constraints)
and writes them as four machine CSVs; ``quant_portfolio_construct_csv`` sizes
the book and prints one JSON line; `AllocationOutput` mirrors that JSON field
for field (``extra="forbid"``, so C++/Python schema drift fails loudly).

There is NO Python fallback. If the C++ binary is missing the plan says so
(`AllocatorUnavailable`) -- a second, Python allocator could silently disagree
with the engine that executes the plan. Numbers cross both ways as
round-trip-exact text (``repr`` in, ``%.17g`` out); Python never recomputes
a notional, a unit or a risk contribution.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from alpha_agent.portfolio.policy import PortfolioConstraints
from alpha_agent.portfolio.risk_model import RiskModel
from alpha_agent.portfolio.selection import SelectedSignal

__all__ = [
    "AllocationOutput",
    "AllocatorError",
    "AllocatorInputs",
    "AllocatorUnavailable",
    "ClusterAllocation",
    "ConstraintOutcome",
    "InstrumentAllocation",
    "PortfolioStats",
    "SignalAllocation",
    "build_allocator_inputs",
    "find_allocator_cli",
    "run_allocator",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_NAME = "quant_portfolio_construct_csv"
CLI_ENV = "QUANT_PORTFOLIO_CONSTRUCT_CLI"
_BUILD_DIRS = ("build/cpp/cpp", "build/cpp", "build")


class AllocatorUnavailable(RuntimeError):
    """The C++ allocator binary is not built -- no plan can be sized."""


class AllocatorError(RuntimeError):
    """The C++ allocator refused its input (a malformed input, never a property of the portfolio)."""


def find_allocator_cli(name: str = CLI_NAME, env: str = CLI_ENV) -> Path | None:
    override = os.environ.get(env)
    if override:
        return Path(override) if Path(override).exists() else None
    for d in _BUILD_DIRS:
        p = REPO_ROOT / d / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


class AllocatorInputs(BaseModel):
    """The exact CSV text the allocator reads -- kept so a plan can be
    re-run byte for byte."""

    model_config = {"frozen": True, "extra": "forbid"}

    instruments_csv: str
    signals_csv: str
    correlations_csv: str
    limits_csv: str

    def sha256(self) -> dict[str, str]:
        return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in self.files().items()}

    def files(self) -> dict[str, str]:
        return {"instruments.csv": self.instruments_csv, "signals.csv": self.signals_csv,
                "correlations.csv": self.correlations_csv, "limits.csv": self.limits_csv}

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in self.files().items():
            (directory / name).write_text(text, encoding="utf-8")


def _num(x: float) -> str:
    return repr(float(x))


def _field(text: str) -> str:
    if "," in text or "\n" in text:
        raise ValueError(f"field {text!r} cannot cross the CSV boundary")
    return text


def build_allocator_inputs(
    signals: Sequence[SelectedSignal], risk: RiskModel, constraints: PortfolioConstraints,
) -> AllocatorInputs:
    used = sorted({s.instrument_key for s in signals})
    instruments = [("instrument_key,domain,root_symbol,asset_class,sector,price,multiplier,annual_vol,adv_usd,"
                    "max_units,decision_ts_ns")]
    for key in used:
        i = risk.instrument(key)
        max_units = constraints.max_futures_contracts if i.unit == "contract" and constraints.max_futures_contracts else 0
        instruments.append(",".join([
            _field(i.key), i.domain.value, _field(i.execution_root), i.asset_class.value, _field(i.sector),
            _num(i.price), _num(i.multiplier), _num(i.annual_vol), _num(i.adv_usd if i.adv_usd is not None else -1.0),
            str(max_units), str(i.decision_ts_ns),
        ]))
    rows = ["signal_id,instrument_key,cluster_id,direction,priority"]
    for s in sorted(signals, key=lambda x: (x.rank, x.candidate_signal_id)):
        if s.direction not in (1, -1):
            raise ValueError(f"{s.candidate_signal_id}: direction is resolved before allocation")
        rows.append(",".join([_field(s.candidate_signal_id), _field(s.instrument_key), _field(s.exposure_group_id),
                              str(s.direction), str(s.rank)]))
    corr = ["instrument_a,instrument_b,correlation"]
    for i, a in enumerate(used):
        for b in used[i + 1:]:
            corr.append(f"{a},{b},{_num(risk.correlation(a, b))}")
    c = constraints
    limits = [
        ("capital_usd,target_annual_vol,max_gross_leverage,max_net_exposure,max_instrument_gross_share,"
         "max_sector_gross_share,max_asset_class_gross_share,max_cluster_risk_share,max_adv_participation,"
         "min_adv_usd,max_turnover,shorting_allowed,allowed_domains"),
        ",".join([
            _num(c.capital_usd), _num(c.target_annual_vol), _num(c.max_gross_leverage),
            _num(c.max_net_exposure or 0.0), _num(c.max_instrument_gross_share), _num(c.max_sector_gross_share),
            _num(c.max_asset_class_gross_share), _num(c.max_exposure_risk_share), _num(c.max_adv_participation),
            _num(c.min_adv_usd), _num(c.max_turnover or 0.0), "true" if c.shorting_allowed else "false",
            "|".join(d.value for d in c.allowed_domains) or "NONE",
        ]),
    ]
    return AllocatorInputs(instruments_csv="\n".join(instruments) + "\n", signals_csv="\n".join(rows) + "\n",
                           correlations_csv="\n".join(corr) + "\n", limits_csv="\n".join(limits) + "\n")


# ---------------------------------------------------------------------------
# output -- a verbatim mirror of quant::ConstructionResult's JSON
# ---------------------------------------------------------------------------


class InstrumentAllocation(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    instrument_key: str
    domain: str
    root_symbol: str
    asset_class: str
    sector: str
    price: float
    multiplier: float
    unit_notional_usd: float
    annual_vol: float
    adv_usd: float
    max_units: int
    decision_ts_ns: int
    admitted: bool
    reason: str
    target_weight: float
    constrained_weight: float
    previous_units: int
    units: int
    executable_weight: float
    target_notional_usd: float
    executable_notional_usd: float
    rounding_residual_usd: float
    adv_participation: float
    risk_contribution: float
    unit_risk_fraction: float
    below_one_unit: bool


class SignalAllocation(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    signal_id: str
    instrument_key: str
    cluster_id: str
    direction: int
    priority: int
    allocated: bool
    reason: str
    target_weight: float
    executable_weight: float
    risk_contribution: float


class ClusterAllocation(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    cluster_id: str
    signal_ids: tuple[str, ...]
    allocated: bool
    reason: str
    composite_vol: float
    erc_weight: float
    target_risk_contribution: float
    executable_risk_contribution: float


class ConstraintOutcome(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    name: str
    scope: str
    stage: str
    status: str
    limit: float
    before: float
    after: float
    affected: tuple[str, ...]


class PortfolioStats(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    annual_vol: float
    positions: int
    effective_exposures: float
    max_cluster_risk_share: float


class AllocationOutput(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    method: str
    status: str
    capital_usd: float
    target_annual_vol: float
    instruments: tuple[InstrumentAllocation, ...]
    signals: tuple[SignalAllocation, ...]
    clusters: tuple[ClusterAllocation, ...]
    constraints: tuple[ConstraintOutcome, ...]
    target: PortfolioStats
    constrained: PortfolioStats
    executable: PortfolioStats
    erc_sweeps: int
    erc_max_deviation: float
    turnover_target: float
    turnover_executable: float
    repair_units_removed: int

    def instrument(self, key: str) -> InstrumentAllocation:
        return next(i for i in self.instruments if i.instrument_key == key)

    def signal(self, signal_id: str) -> SignalAllocation:
        return next(s for s in self.signals if s.signal_id == signal_id)


def run_allocator(inputs: AllocatorInputs, *, cli: Path | None = None, timeout_s: float = 60.0) -> AllocationOutput:
    binary = cli or find_allocator_cli()
    if binary is None or not Path(binary).exists():
        raise AllocatorUnavailable(
            f"{CLI_NAME} is not built -- build the C++ core (cmake -S . -B build/cpp && cmake --build build/cpp)")
    with tempfile.TemporaryDirectory(prefix="portfolio-construct-") as tmp:
        inputs.write(Path(tmp))
        proc = subprocess.run([str(binary), tmp], capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise AllocatorError(proc.stderr.strip() or f"{CLI_NAME} exited {proc.returncode}")
    return AllocationOutput.model_validate_json(proc.stdout)
