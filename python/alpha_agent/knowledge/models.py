"""Alpha Discovery campaign, Part C -- the typed Strategy Knowledge Base model.

``StrategyKnowledgeItem`` is IDEA MEMORY, never execution authority (CLAUDE.md
/ task spec section 13): nothing here computes PnL, assigns a verdict, or
enters the BH/FDR family. A reported historical result from an external
source (``reported_results``) is a free-text field the platform NEVER treats
as our evidence -- it exists only so a human/agent can see what a source
*claimed*, side by side with what our own C++ engine actually measured.

Source-quality tiers (task spec section 15) influence research PRIORITY --
which mechanism a generation pass reaches for first -- and nothing else. They
never touch ``ReliabilityPolicy``, BH/FDR, DSR, or any validation threshold;
every strategy, regardless of its inspirations' tier, is independently tested
through the same closed DSL and the same C++ engine.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class SourceType(str, Enum):
    """Task spec section 14."""

    INTERNAL = "INTERNAL"
    CLASSIC = "CLASSIC"
    ACADEMIC = "ACADEMIC"
    GITHUB = "GITHUB"
    COMMUNITY = "COMMUNITY"
    PRACTITIONER = "PRACTITIONER"


class SourceQualityTier(str, Enum):
    """Task spec section 15. Research-priority metadata ONLY -- see module
    docstring. `UNKNOWN` is the safe default for a source whose provenance or
    rigor cannot be assessed; it is never silently upgraded."""

    TIER_A = "TIER_A"  # peer-reviewed academic / well-documented institutional research
    TIER_B = "TIER_B"  # established open-source quant libraries / reputable platforms
    TIER_C = "TIER_C"  # community posts, forum research, individual blogs
    TIER_D = "TIER_D"  # unverified strategy claims
    UNKNOWN = "UNKNOWN"


class EconomicMechanism(str, Enum):
    """A closed, fixed catalog of economically distinct mechanism categories
    (task spec section 3 / section 16). This is what stays constant across
    many *sources* describing the "same idea" -- see
    ``alpha_agent.knowledge.dedup``. Adding a mechanism here is a documented,
    reviewed change, never something free-text can smuggle in."""

    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    FAILED_BREAKOUT = "FAILED_BREAKOUT"
    VOLATILITY_BREAKOUT = "VOLATILITY_BREAKOUT"
    VOLATILITY_TRANSITION = "VOLATILITY_TRANSITION"
    OPENING_RANGE = "OPENING_RANGE"
    SESSION_EFFECTS = "SESSION_EFFECTS"
    OVERNIGHT_GAP = "OVERNIGHT_GAP"
    VOLUME_LIQUIDITY = "VOLUME_LIQUIDITY"
    CARRY = "CARRY"
    TERM_STRUCTURE = "TERM_STRUCTURE"
    CROSS_MARKET_LEAD_LAG = "CROSS_MARKET_LEAD_LAG"
    RELATIVE_VALUE = "RELATIVE_VALUE"
    CORRELATION_SPREAD = "CORRELATION_SPREAD"
    REGIME_CONDITIONED_TREND = "REGIME_CONDITIONED_TREND"
    REGIME_CONDITIONED_MEAN_REVERSION = "REGIME_CONDITIONED_MEAN_REVERSION"
    HYBRID_TREND_REVERSAL = "HYBRID_TREND_REVERSAL"
    MULTI_SIGNAL_ENSEMBLE = "MULTI_SIGNAL_ENSEMBLE"
    ML_META_LABELING = "ML_META_LABELING"


class IngestionStatus(str, Enum):
    """Task spec section 23/30. A live external adapter reports its ACTUAL
    outcome honestly rather than being faked -- never silently downgraded to
    NOT_CONNECTED on a real, distinguishable failure mode, and never silently
    upgraded to CONNECTED on a partial/degraded fetch.

    DISABLED (Release UX bugfix pass) is DISTINCT from NOT_CONNECTED: a
    caller uses DISABLED when this source category was never even ATTEMPTED
    because the user's own research-sources selection excluded it for this
    campaign (e.g. "Internal + Classic only") -- see
    `alpha_agent.knowledge.external_sources.disabled_adapters`. NOT_CONNECTED
    stays reserved for a source that WAS attempted (live research was
    requested) and the connector itself could not connect (missing
    credential, unreachable). Showing NOT_CONNECTED for a source that was
    never requested falsely implies a real attempt was made and failed."""

    SEEDED = "SEEDED"              # deterministic, committed fixture / curated item
    DISABLED = "DISABLED"          # not requested for this campaign -- never attempted
    NOT_CONNECTED = "NOT_CONNECTED"  # live research was requested; this connector could not connect
    CONNECTED = "CONNECTED"        # a real live fetch actually happened
    DEGRADED = "DEGRADED"          # a real live fetch partially succeeded (some items missing)
    RATE_LIMITED = "RATE_LIMITED"  # a real live fetch was refused by the provider's rate limit
    ERROR = "ERROR"                # a real live fetch was attempted and failed (auth, network, ...)


class StrategyKnowledgeItem(BaseModel):
    """One idea reference (task spec section 14). Every field beyond the
    identity/source ones is optional -- "do not require every external source
    to populate every field" (section 14); a missing field stays missing,
    never guessed."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "strategy-knowledge-item/1"
    knowledge_id: str

    source_type: SourceType
    source_quality: SourceQualityTier = SourceQualityTier.UNKNOWN
    ingestion_status: IngestionStatus = IngestionStatus.SEEDED

    title: str
    authors: tuple[str, ...] = ()
    source_url: str | None = None
    repository: str | None = None
    commit_sha: str | None = None
    license: str | None = None
    accessed_at: str | None = None
    publication_date: str | None = None

    markets: tuple[str, ...] = ()
    asset_classes: tuple[str, ...] = ()
    time_horizon: str | None = None

    economic_mechanism: EconomicMechanism
    required_features: tuple[str, ...] = ()
    entry_logic_summary: str = ""
    exit_logic_summary: str = ""
    risk_logic_summary: str = ""
    parameter_summary: str = ""

    #: Free text. NEVER our evidence -- see module docstring.
    reported_results: str | None = None

    implementation_notes: str = ""
    limitations: str = ""
    internal_supported_capabilities: str = ""
    #: A human-readable pointer to a Phase-11/candidate-DSL template this idea
    #: most resembles, if any (e.g. "tsmom", "breakout") -- documentation only,
    #: never a compiled StrategySpec and never itself executed.
    candidate_dsl_template: str | None = None

    provenance_hash: str = ""

    def render_snippet(self) -> str:
        """One deterministic, compact line for `ResearchContext.knowledge_base`
        / `OrchestratorConfig.knowledge_base` (both already holdout-guarded
        downstream). Plain text only -- never executable, never a claim of
        our own performance."""
        bits = [f"[{self.source_type.value}/{self.source_quality.value}] {self.title}"]
        bits.append(f"mechanism={self.economic_mechanism.value}")
        if self.markets:
            bits.append(f"markets={','.join(self.markets)}")
        if self.entry_logic_summary:
            bits.append(f"entry: {self.entry_logic_summary}")
        if self.limitations:
            bits.append(f"limitations: {self.limitations}")
        return " | ".join(bits)


class MechanismCluster(BaseModel):
    """Task spec section 24: several sources describing ONE underlying
    mechanism collapse to one cluster -- "5 GitHub repos implementing a 20/50
    MA crossover" is one economic hypothesis with five references, not five."""

    model_config = {"frozen": True, "extra": "forbid"}

    economic_mechanism: EconomicMechanism
    items: tuple[StrategyKnowledgeItem, ...]

    @property
    def knowledge_ids(self) -> tuple[str, ...]:
        return tuple(i.knowledge_id for i in self.items)

    @property
    def best_source_quality(self) -> SourceQualityTier:
        order = [
            SourceQualityTier.TIER_A, SourceQualityTier.TIER_B,
            SourceQualityTier.TIER_C, SourceQualityTier.TIER_D, SourceQualityTier.UNKNOWN,
        ]
        return min(
            (i.source_quality for i in self.items),
            key=lambda t: order.index(t),
            default=SourceQualityTier.UNKNOWN,
        )


class IngestionResult(BaseModel):
    """What one source adapter produced (or honestly did not) -- task spec
    section 23/68."""

    model_config = {"frozen": True, "extra": "forbid"}

    source_type: SourceType
    status: IngestionStatus
    items: tuple[StrategyKnowledgeItem, ...] = ()
    detail: str = ""


TIER_ORDER: tuple[SourceQualityTier, ...] = (
    SourceQualityTier.TIER_A,
    SourceQualityTier.TIER_B,
    SourceQualityTier.TIER_C,
    SourceQualityTier.TIER_D,
    SourceQualityTier.UNKNOWN,
)
