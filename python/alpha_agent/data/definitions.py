"""Contract-definition parsing and the Python contract registry (layer 5).

Mirrors the C++ ``quant::ContractRegistry``. The registry is the *authoritative*
answer to "is this a real tradable contract" -- symbol syntax alone is never
enough (prompt 03 point 4).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_agent.data.contract_economics import ContractEconomics, derive_contract_economics
from alpha_agent.schemas.market_data import (
    CONTRACT_COLUMNS,
    VENDOR_DEFINITION_COLUMNS,
    ContractSpecModel,
    is_tradable_contract_symbol,
)


def _opt_ns(value) -> int | None:
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return None
    iv = int(value)
    return iv if iv > 0 else None


def _opt(value):
    if value == "" or value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def _econ_for_row(row: dict, *, allow_fractional: bool):
    return derive_contract_economics(
        min_price_increment=row["min_price_increment"],
        min_price_increment_amount=row["min_price_increment_amount"],
        display_factor=row["display_factor"],
        unit_of_measure_qty=row["unit_of_measure_qty"],
        unit_of_measure=row["unit_of_measure"],
        main_fraction=_opt(row.get("main_fraction")),
        contract_multiplier=_opt(row.get("contract_multiplier")),
        allow_fractional=allow_fractional,
    )


def contract_economics_for(
    frame: pd.DataFrame, *, allow_fractional: bool = False
) -> dict[int, ContractEconomics]:
    """Derive :class:`ContractEconomics` per instrument_id from a vendor frame."""
    return {
        int(row["instrument_id"]): _econ_for_row(row, allow_fractional=allow_fractional)
        for row in frame.to_dict("records")
    }


def parse_definition_frame(
    frame: pd.DataFrame,
    *,
    root_map: dict[str, str] | None = None,
    allow_fractional: bool = False,
) -> list[ContractSpecModel]:
    """Turn vendor instrument-definition records into validated ``ContractSpecModel``.

    ``frame`` must contain ``VENDOR_DEFINITION_COLUMNS`` (extra columns ignored).
    ``tick_size`` and ``multiplier`` (= USD point value) are DERIVED via
    ``contract_economics`` -- an ``EconomicsError`` there aborts the parse rather
    than guessing. ``allow_fractional`` enables the CBOT-Treasury path.
    ``root_map`` (raw_symbol -> root) is consulted only when the vendor ``asset``
    field is blank; a row whose root cannot be determined is rejected.
    """
    missing = [c for c in VENDOR_DEFINITION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"definition frame missing columns: {missing}")

    root_map = root_map or {}
    specs: list[ContractSpecModel] = []
    for row in frame.to_dict("records"):
        raw_symbol = str(row["raw_symbol"]).strip()
        asset = str(row["asset"]).strip() if row["asset"] not in (None, "") else ""
        root = asset or root_map.get(raw_symbol, "")
        if not root:
            raise ValueError(
                f"cannot determine root for contract {raw_symbol!r}: vendor 'asset' is blank "
                f"and no root_map entry was supplied"
            )
        econ = _econ_for_row(row, allow_fractional=allow_fractional)
        spec = ContractSpecModel(
            instrument_id=int(row["instrument_id"]),
            raw_symbol=raw_symbol,
            root_symbol=root,
            exchange=str(row["exchange"]).strip(),
            tick_size=econ.quote_tick_size,
            multiplier=econ.point_value_usd,   # USD PnL per 1.0 quoted-price move
            activation_ns=int(row["activation"]),
            expiration_ns=int(row["expiration"]),
            first_notice_ns=_opt_ns(row["first_notice"]),
            last_trade_ns=_opt_ns(row["last_trade"]),
        )
        specs.append(spec)
    return specs


class DefinitionRegistry:
    """Lookup over validated ``ContractSpecModel`` -- the Python mirror of
    ``quant::ContractRegistry``."""

    def __init__(self, specs: list[ContractSpecModel]):
        self._by_id: dict[int, ContractSpecModel] = {}
        self._by_symbol: dict[str, int] = {}
        for spec in specs:
            self.add(spec)

    def add(self, spec: ContractSpecModel) -> None:
        if not is_tradable_contract_symbol(spec.raw_symbol):
            raise ValueError(f"not a tradable contract symbol: {spec.raw_symbol!r}")
        if spec.instrument_id in self._by_id:
            raise ValueError(f"duplicate instrument_id {spec.instrument_id}")
        if spec.raw_symbol in self._by_symbol:
            raise ValueError(f"duplicate raw_symbol {spec.raw_symbol!r}")
        self._by_id[spec.instrument_id] = spec
        self._by_symbol[spec.raw_symbol] = spec.instrument_id

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, instrument_id: int) -> bool:
        return instrument_id in self._by_id

    def get(self, instrument_id: int) -> ContractSpecModel | None:
        return self._by_id.get(instrument_id)

    def require(self, instrument_id: int) -> ContractSpecModel:
        spec = self._by_id.get(instrument_id)
        if spec is None:
            raise KeyError(f"unknown instrument_id {instrument_id}")
        return spec

    def by_raw_symbol(self, raw_symbol: str) -> ContractSpecModel | None:
        iid = self._by_symbol.get(raw_symbol)
        return None if iid is None else self._by_id[iid]

    def is_tradable(self, instrument_id: int) -> bool:
        return instrument_id in self._by_id

    def specs(self) -> list[ContractSpecModel]:
        return [self._by_id[i] for i in sorted(self._by_id)]

    def roots(self) -> list[str]:
        return sorted({s.root_symbol for s in self._by_id.values()})


def contracts_frame(specs: list[ContractSpecModel]) -> pd.DataFrame:
    """A DataFrame in the frozen ``CONTRACT_COLUMNS`` order (empty optionals as "")."""
    rows = []
    for s in specs:
        rows.append(
            {
                "instrument_id": s.instrument_id,
                "raw_symbol": s.raw_symbol,
                "root_symbol": s.root_symbol,
                "exchange": s.exchange,
                "tick_size": s.tick_size,
                "multiplier": s.multiplier,
                "activation_ns": s.activation_ns,
                "expiration_ns": s.expiration_ns,
                "first_notice_ns": "" if s.first_notice_ns is None else s.first_notice_ns,
                "last_trade_ns": "" if s.last_trade_ns is None else s.last_trade_ns,
            }
        )
    return pd.DataFrame(rows, columns=list(CONTRACT_COLUMNS))


def write_contracts_parquet(specs: list[ContractSpecModel], out_dir: str | Path) -> list[Path]:
    """Write ``data/processed/contracts/<root>.parquet`` per root. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    frame = contracts_frame(specs)
    for root, group in frame.groupby("root_symbol"):
        path = out_dir / f"{root}.parquet"
        group.reset_index(drop=True).to_parquet(path, index=False)
        written.append(path)
    return written
