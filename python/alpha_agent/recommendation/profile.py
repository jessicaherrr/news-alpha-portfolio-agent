"""Phase B1 -- the investor / research preference profile.

CLAUDE.md boundary (Phase B1 task spec, sections 4/5/12): a user preference is
presentation / research-prioritization state, never scientific evidence. It is
kept OUT of `ExperimentRegistry`, `experiment_identity`, and every validation
artifact, and it never changes `RegistryVerdict`, `ReliabilityPolicy`, or any
BH/FDR/DSR computation -- see `alpha_agent.recommendation.promise` /
`alpha_agent.recommendation.fit`, both of which take an `InvestorProfile` only
to score PRESENTATION dimensions (User Fit) that are architecturally
independent of the scientific score (Research Promise) and of the verdict.

`InvestorProfile` is a small, frozen (immutable), validated Pydantic model
with explicit enums -- every field is a closed choice, never free text, so a
profile is always well-formed and comparable. `ProfileStore` is the smallest
safe persistence mechanism (preference order #1 in the task spec): a single
local JSON file under `data/user_prefs/`, gitignored (personal preference
data, not registry truth, never committed to the repository). The Streamlit
layer additionally caches the loaded profile in `st.session_state` for the
lifetime of one session -- see `alpha_agent.ui.panels.render_profile_panel`.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

#: python/alpha_agent/recommendation/profile.py -> repo root, independent of
#: the process's current working directory (same derivation as
#: `alpha_agent.ui.services.REPO_ROOT`).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_PATH = REPO_ROOT / "data" / "user_prefs" / "investor_profile.json"


class HoldingPeriod(str, Enum):
    INTRADAY = "Intraday"
    ONE_TO_THREE_DAYS = "1-3 Days"
    SEVERAL_DAYS = "Several Days"
    WEEKS = "Weeks"
    FLEXIBLE = "Flexible"


class RiskStyle(str, Enum):
    CONSERVATIVE = "Conservative"
    BALANCED = "Balanced"
    AGGRESSIVE = "Aggressive"


class MaxDrawdown(str, Enum):
    PCT_5 = "5%"
    PCT_10 = "10%"
    PCT_15 = "15%"
    PCT_20_PLUS = "20%+"


class TradingFrequency(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    NO_PREFERENCE = "No Preference"


class OvernightPreference(str, Enum):
    AVOID = "Avoid"
    ALLOWED = "Allowed"
    NO_PREFERENCE = "No Preference"


class StrategyPreference(str, Enum):
    TREND = "Trend"
    MEAN_REVERSION = "Mean Reversion"
    BREAKOUT = "Breakout"
    RELATIVE_VALUE = "Relative Value"
    MIXED = "Mixed / No Preference"


class TurnoverSensitivity(str, Enum):
    """Optional field (task spec section 4). Distinct enum from
    `TradingFrequency` even though the buckets read the same, because it
    measures a different (currently unscored -- see `alpha_agent.recommendation.
    fit`) preference axis."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    NO_PREFERENCE = "No Preference"


class InvestorProfile(BaseModel):
    """A closed, immutable, validated research-preference profile. Every
    field is a fixed enum choice -- there is no free-text field an LLM (or a
    hand-edited file) could use to smuggle a scientific claim into this
    object. Frozen: a "changed profile" is always a NEW `InvestorProfile`
    instance, never a mutation of one already in use elsewhere."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "investor-profile/1"

    holding_period: HoldingPeriod = HoldingPeriod.FLEXIBLE
    risk_style: RiskStyle = RiskStyle.BALANCED
    max_drawdown: MaxDrawdown = MaxDrawdown.PCT_15
    trading_frequency: TradingFrequency = TradingFrequency.NO_PREFERENCE
    overnight: OvernightPreference = OvernightPreference.NO_PREFERENCE
    strategy_preference: StrategyPreference = StrategyPreference.MIXED

    #: Optional fields (task spec section 4: "only if easy and safe"). Never
    #: used to gate anything; `approximate_capital_usd` / `max_contracts` are
    #: not yet consumed by `alpha_agent.recommendation.fit` (no committed
    #: registry evidence exists to compare them against -- see that module's
    #: docstring) and are carried here only so the profile form need not be
    #: revisited when such evidence lands.
    approximate_capital_usd: float | None = Field(default=None, ge=0)
    max_contracts: int | None = Field(default=None, ge=1)
    turnover_sensitivity: TurnoverSensitivity | None = None


#: The sensible, clearly-non-personalized default (task spec section 20).
#: Never presented as personalized advice -- see
#: `alpha_agent.ui.panels.render_profile_panel`'s "using defaults" caption.
DEFAULT_PROFILE = InvestorProfile()


class ProfileStore:
    """Smallest-safe local persistence for one `InvestorProfile`: a single
    JSON file (task spec section 5, preference #1: "typed local application
    preference model"). No database, no registry table, no network call.
    Corrupt or missing state always falls back to `DEFAULT_PROFILE` -- this
    is presentation preference, never data whose loss should break the app.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PROFILE_PATH

    def load(self) -> InvestorProfile:
        if not self.path.exists():
            return DEFAULT_PROFILE
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return InvestorProfile.model_validate(data)
        except Exception:  # noqa: BLE001 -- a corrupt local preference file is not fatal
            return DEFAULT_PROFILE

    def save(self, profile: InvestorProfile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(profile.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
        )
