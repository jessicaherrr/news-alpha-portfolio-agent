"""Phase 4 -- the Concepts library: one typed, Three-Lens (Intuition / Quant /
Implementation) explanation per validation gate and market-structure idea a
beginner meets on a professional surface.

Every ``pointers`` entry is a REAL, checked reference into this repository --
either ``"module.path:attribute"`` (resolved by ``importlib`` + ``getattr`` in
``test_phase4_learn_concepts.py``) or a bare relative file path (checked to
exist on disk). Nothing here is prose about a hypothetical implementation.

Concept content is fixed, hand-written, and versioned by this file's own git
history -- never generated at runtime, never LLM-authored (CLAUDE.md: the LLM
may narrate a *specific result*, per :mod:`alpha_agent.learn.narration_agent`,
but a Concept's own definition is exactly as fixed as a gate's own reason
code).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "CONCEPT_CATEGORIES",
    "CONCEPT_LIBRARY",
    "GATE_FAIL_CODE_TO_CONCEPT",
    "Concept",
    "get_concept",
    "list_concepts",
]


class Concept(BaseModel):
    """One Three-Lens explanation. ``gate_fail_code`` is the exact
    ``ReasonCode.value`` this concept explains when it is a validation gate
    (see ``alpha_agent.ui.services.GATE_DEFINITIONS``); ``None`` for a
    market-structure or execution-integrity concept that is not itself a
    gate."""

    model_config = {"frozen": True, "extra": "forbid"}

    concept_id: str
    title: str
    category: str
    gate_fail_code: str | None = None
    one_line: str
    intuition: str
    quant: str
    implementation: str
    pointers: tuple[str, ...] = Field(default_factory=tuple)


CONCEPT_CATEGORIES: tuple[str, ...] = (
    "validation_gate",
    "market_structure",
    "execution_integrity",
)

_CONCEPTS: tuple[Concept, ...] = (
    Concept(
        concept_id="bootstrap_null",
        title="Null Hypothesis (Bootstrap)",
        category="validation_gate",
        gate_fail_code="null_hypothesis_not_rejected",
        one_line="Could a strategy that knows nothing have produced this by chance?",
        intuition=(
            "Before trusting a strategy's Sharpe ratio, ask whether a coin-flip strategy could have "
            "produced something similar just by luck. This gate reshuffles the strategy's own daily "
            "returns (in overlapping blocks, so it doesn't destroy real autocorrelation) many times and "
            "checks where the REAL result falls in that random distribution. If the real result isn't "
            "clearly better than the shuffled versions, it's not distinguishable from noise."
        ),
        quant=(
            "A centered block bootstrap resamples the daily PnL series in contiguous blocks (preserving "
            "short-horizon autocorrelation structure that a naive i.i.d. resample would destroy), "
            "recomputes annualized Sharpe on each resample, and reports the gating null p-value: the "
            "fraction of resamples whose statistic is at least as extreme as the observed one. A small "
            "p-value means the observed Sharpe is unlikely under the null that returns are exchangeable "
            "noise with the same block structure."
        ),
        implementation=(
            "Python computes the block bootstrap over the committed daily-equity trace; the frozen "
            "config (block length, number of resamples) is bound into the experiment's identity so the "
            "null test itself cannot be silently loosened between runs."
        ),
        pointers=(
            "alpha_agent.validation.nulls:centered_block_bootstrap_null_stats",
            "alpha_agent.validation.bootstrap:bootstrap_daily_returns",
        ),
    ),
    Concept(
        concept_id="bh_fdr",
        title="Multiple Testing (BH-FDR)",
        category="validation_gate",
        gate_fail_code="fdr_qvalue_above_threshold",
        one_line="If you test 60 ideas, some will look good by chance alone -- this gate corrects for that.",
        intuition=(
            "Test one hypothesis at a 5% significance level and you have a 5% false-positive risk. Test "
            "60 and, by chance alone, a few will clear that bar even if none are real. The "
            "Benjamini-Hochberg False Discovery Rate procedure adjusts each hypothesis's p-value for how "
            "many were tested together (its predeclared 'family'), producing a q-value: the expected "
            "share of false discoveries among everything you'd call significant at that q."
        ),
        quant=(
            "p-values from the whole predeclared family are sorted ascending; the largest p(i) with "
            "p(i) <= (i/m)*q is the rejection threshold (m = family size, q = the target FDR). Every "
            "hypothesis at or below that rank is BH-rejected (i.e. survives); q-value is the smallest q "
            "at which a given hypothesis would still be rejected. The family size m is fixed BEFORE any "
            "result is seen -- CLAUDE.md requires the predeclared BH family to be constant (Phase 15: "
            "always exactly 60) so a convenient post-hoc subset can never inflate significance."
        ),
        implementation=(
            "Python computes q-values once per frozen family (e.g. the whole Phase 13.5C 107-trial "
            "matrix, or Phase 15B's fixed 60), never per-experiment in isolation -- the family "
            "composition is itself part of `ReliabilityPolicy` and bound into every member's identity."
        ),
        pointers=("alpha_agent.validation.fdr:benjamini_hochberg_decisions", "alpha_agent.validation.fdr:bh_qvalues"),
    ),
    Concept(
        concept_id="dsr",
        title="Deflated Sharpe Ratio",
        category="validation_gate",
        gate_fail_code="deflated_sharpe_below_threshold",
        one_line="The more parameter combinations you tried, the luckier the best one looks -- DSR corrects for that.",
        intuition=(
            "If you try 50 parameter variants of the same idea, the BEST one's Sharpe ratio is inflated "
            "just by selection -- picking the max of many noisy draws. The Deflated Sharpe Ratio asks: "
            "given how many independent-ish trials were effectively run to find this one, and given the "
            "skew/kurtosis of its own return distribution, what is the probability the TRUE Sharpe is "
            "actually positive (or above a stated benchmark)?"
        ),
        quant=(
            "DSR = Probabilistic Sharpe Ratio evaluated against a deflated benchmark Sharpe, where the "
            "benchmark accounts for the variance of Sharpe ratios across the effective number of "
            "independent trials (`n_trials`) and the non-normality of returns (skew, excess kurtosis) via "
            "the PSR's own variance formula. A high `dsr_probability` means the edge is unlikely to be a "
            "product of both selection-among-many-trials AND non-normal-return luck."
        ),
        implementation=(
            "The effective trial count is the real count of parameter variants actually run for that "
            "family (never guessed), and the policy's `dsr_probability_over_<threshold>` field is read "
            "verbatim off the committed `ResultRecord` -- this page applies no threshold of its own."
        ),
        pointers=(
            "alpha_agent.validation.dsr:deflated_sharpe_ratio",
            "alpha_agent.validation.dsr:probabilistic_sharpe_ratio",
            "alpha_agent.validation.dsr:deflated_benchmark_sharpe",
        ),
    ),
    Concept(
        concept_id="positive_oos_pnl",
        title="Positive OOS Net PnL",
        category="validation_gate",
        gate_fail_code="negative_oos_net_pnl",
        one_line="After real commissions and slippage, did the strategy actually make money out-of-sample?",
        intuition=(
            "The most basic bar of all: after subtracting real commissions and slippage from every fill, "
            "did the strategy's out-of-sample equity end above where it started? A strategy can pass "
            "every statistical test and still lose money once costs are real -- this gate checks the "
            "plain arithmetic fact, not a statistic about it."
        ),
        quant=(
            "net_pnl_usd = gross_pnl_usd - costs_usd, summed over every fill in the out-of-sample "
            "window, using the SAME cost assumptions (commission per contract, slippage ticks, spread "
            "ticks) declared for the whole experiment family -- never a friendlier cost model applied "
            "after the fact."
        ),
        implementation=(
            "Computed by the C++ portfolio accountant from real fills, never recomputed or approximated "
            "in Python (CLAUDE.md: C++ owns PnL, fills, and accounting)."
        ),
        pointers=("cpp/include/quant_core/portfolio.hpp", "cpp/include/quant_core/execution_simulator.hpp"),
    ),
    Concept(
        concept_id="walk_forward",
        title="Walk-Forward Consistency",
        category="validation_gate",
        gate_fail_code="fold_consistency_below_threshold",
        one_line="Does the edge hold up across many separate time windows, or did one lucky period carry it?",
        intuition=(
            "A strategy that made all its money in one six-month window and lost steadily everywhere "
            "else is not a robust edge, even if the total is positive. Walk-forward validation splits "
            "the research window into sequential folds and checks how CONSISTENTLY the strategy performed "
            "across them, not just the sum."
        ),
        quant=(
            "Fold consistency is the fraction of walk-forward folds with a positive (or above-threshold) "
            "result. Each fold trains/selects on an earlier window and evaluates strictly on a later, "
            "disjoint window -- never on data the fold's own decision could have seen (CLAUDE.md: no "
            "look-ahead)."
        ),
        implementation=(
            "Fold boundaries are built once, deterministically, from the frozen `WalkForwardConfig` -- "
            "the same folds are reused for every strategy in a family so consistency is comparable "
            "across the whole matrix."
        ),
        pointers=("alpha_agent.validation.walkforward:build_folds", "alpha_agent.validation.walkforward:summarize_walk_forward"),
    ),
    Concept(
        concept_id="cost_stress",
        title="Cost Stress",
        category="validation_gate",
        gate_fail_code="cost_stress_degradation_exceeds_limit",
        one_line="What happens to the edge if real-world trading costs are a bit worse than assumed?",
        intuition=(
            "Commission and slippage assumptions are estimates, not certainties. A strategy whose edge "
            "evaporates the moment costs tick up slightly was never robust -- it was riding a thin, "
            "cost-sensitive margin. This gate re-runs the SAME trades under several higher-cost scenarios "
            "and checks how much net PnL degrades."
        ),
        quant=(
            "Each cost scenario scales commission/slippage/spread assumptions upward by a declared "
            "factor and recomputes net PnL from the SAME fills (trade timing and sizing are unchanged -- "
            "only the cost model is stressed). `max_net_pnl_degradation` is the worst-case percentage "
            "drop across scenarios relative to the base-case net PnL."
        ),
        implementation=(
            "Cost scenarios are declared once per family (never tuned per-strategy after seeing "
            "results) and applied to the same committed fill sequence."
        ),
        pointers=("alpha_agent.validation.cost_stress:summarize_cost_stress", "alpha_agent.validation.cost_stress:CostStressPlan"),
    ),
    Concept(
        concept_id="parameter_stability",
        title="Parameter Stability",
        category="validation_gate",
        gate_fail_code="parameter_neighbourhood_unstable",
        one_line="Is this parameter choice a broad plateau of good results, or an isolated lucky spike?",
        intuition=(
            "If the canonical parameter setting (say, a 21-day fast horizon) performs great but every "
            "nearby setting (20-day, 22-day) performs badly, that's a warning sign: an isolated spike is "
            "more consistent with overfitting to noise than with a genuine, smoothly-varying economic "
            "effect. A genuine edge usually looks like a PLATEAU of decent results across nearby "
            "parameter values, not a single needle."
        ),
        quant=(
            "The canonical trial's Sharpe is compared against its declared neighbourhood (small, "
            "predeclared perturbations of each parameter) via a percentile rank and a fraction with "
            "positive Sharpe; `canonical_is_isolated_spike` flags when the canonical result is far above "
            "its own neighbourhood's dispersion."
        ),
        implementation=(
            "Neighbour parameter sets are predeclared per family (never chosen after seeing which "
            "neighbours would look favorable) and each neighbour is a full, independently re-executed "
            "backtest, not an interpolation."
        ),
        pointers=("alpha_agent.validation.stability:summarize_parameter_stability",),
    ),
    Concept(
        concept_id="regime_robustness",
        title="Regime Robustness",
        category="validation_gate",
        gate_fail_code="performance_concentrated_in_one_regime",
        one_line="Did the edge come from the whole sample, or almost entirely from one volatility regime?",
        intuition=(
            "A trend-following strategy that only made money during one high-volatility crash quarter, "
            "and did nothing (or lost) in every other regime, is not evidence of a persistent edge -- it "
            "is evidence that a single unusual period dominated the backtest. This gate buckets time into "
            "volatility regimes and checks how concentrated the PnL is."
        ),
        quant=(
            "Bars are causally labeled into volatility regimes (using only trailing information, never a "
            "look-ahead label) and PnL is attributed per regime; `max_regime_pnl_share` is the largest "
            "single regime's share of total PnL -- a high share means the result rests on one regime."
        ),
        implementation=(
            "Regime labels are computed once from trailing realized volatility, deterministically, "
            "before any PnL attribution -- never derived from the strategy's own returns."
        ),
        pointers=("alpha_agent.validation.regime:causal_volatility_regime_labels", "alpha_agent.validation.regime:evaluate_regime_stability"),
    ),
    Concept(
        concept_id="cross_market_evidence",
        title="Cross-Market Evidence",
        category="validation_gate",
        gate_fail_code="performance_concentrated_in_one_root",
        one_line="Does a similar mechanism show up on other, related markets, or is this one root special?",
        intuition=(
            "An economic mechanism (like time-series momentum) should, in principle, show up across "
            "several related futures markets, not just one. If the SAME strategy family was tested on "
            "several roots and the result is dominated by a single one, that narrows how much you should "
            "generalize the finding."
        ),
        quant=(
            "PnL (or a comparable statistic) is compared across roots within the same strategy family; "
            "`max_single_root_pnl_share` is the largest single root's share -- evaluated only when more "
            "than one root was actually tested for that family."
        ),
        implementation=(
            "Cross-market evidence status is explicitly 'not_evaluated' whenever fewer than two roots "
            "exist for a family -- never silently defaulted to a passing state."
        ),
        pointers=("alpha_agent.validation.crossmarket:evaluate_cross_market",),
    ),
    Concept(
        concept_id="term_structure_backwardation",
        title="Term Structure: Contango vs. Backwardation",
        category="market_structure",
        gate_fail_code=None,
        one_line="Are further-dated contracts priced above (contango) or below (backwardation) the front month?",
        intuition=(
            "A futures curve in CONTANGO has later-dated contracts priced higher than the front month -- "
            "often read as the market pricing in storage costs or an ample near-term supply. A curve in "
            "BACKWARDATION has later contracts priced LOWER -- often read as near-term scarcity or strong "
            "immediate demand (a classic signal in energy markets around inventory draws). Neither shape "
            "is intrinsically bullish or bearish for a directional trade; it describes the SHAPE of "
            "forward pricing, which is itself a research input."
        ),
        quant=(
            "The curve's shape is classified from the pairwise spread and slope across contract months on "
            "the SAME curve date: a front-to-back spread and slope beyond a stated tolerance is CONTANGO "
            "(rising) or BACKWARDATION (falling); within tolerance is FLAT; a non-monotonic curve is "
            "MIXED; too few live contracts is INSUFFICIENT_DATA -- never guessed from two points alone."
        ),
        implementation=(
            "Classification reads real, same-day contract closes from the Databento contract ladder -- "
            "never interpolated or estimated from a single contract's history."
        ),
        pointers=(
            "alpha_agent.marketdata.databento_provider:classify_curve_shape",
            "alpha_agent.marketdata.databento_schemas:CurveShape",
            "alpha_agent.marketdata.databento_schemas:TermStructureResult",
        ),
    ),
    Concept(
        concept_id="look_ahead_bias",
        title="No Look-Ahead: Next-Bar Execution",
        category="execution_integrity",
        gate_fail_code=None,
        one_line="A signal computed from bar T's close can only trade starting at bar T+1 -- never earlier.",
        intuition=(
            "The single most common way a backtest lies is by letting a decision made using information "
            "from bar T (e.g. its closing price) execute AT bar T's own price, as if the strategy had "
            "perfect foresight of the bar that hadn't finished yet. This inflates results, sometimes "
            "enormously, and never shows up in production. The fix: a decision computed from data known "
            "at the close of bar T may execute no earlier than the NEXT eligible bar (T+1, plus modeled "
            "latency)."
        ),
        quant=(
            "Strategy::decide() is called with only bars up to and including T; the resulting order is "
            "queued and can fill no earlier than the next native bar's open plus a modeled latency -- the "
            "engine enforces this structurally, it is not a convention a strategy author could "
            "accidentally violate."
        ),
        implementation=(
            "Enforced inside the C++ reference engine's own event loop, not in Python and not "
            "per-strategy -- the SAME ordering applies to every strategy family, so a strategy with a "
            "look-ahead bug is structurally impossible, not just discouraged."
        ),
        pointers=("cpp/include/quant_core/engine.hpp", "alpha_agent.ui.execution_provenance:EXECUTION_TIMING_TEXT"),
    ),
    Concept(
        concept_id="contract_economics",
        title="Contract Economics: Point Value & Tick Value",
        category="execution_integrity",
        gate_fail_code=None,
        one_line="How many real dollars does one point of price movement mean for one contract?",
        intuition=(
            "A $1 move in ES futures is worth $50 per contract; a $1 move in ZN (10-Year Treasury Notes, "
            "fractionally quoted as a percent of par) is worth something very different. Getting this "
            "wrong by even a small factor silently corrupts every PnL number, every cost-to-edge ratio, "
            "and every Sharpe ratio downstream -- commissions are a flat USD-per-contract charge and do "
            "NOT rescale to compensate for a wrong point value."
        ),
        quant=(
            "For decimal-quoted products, point value is derived from the published contract "
            "specification's unit-of-measure quantity; for fractionally-quoted products (CBOT Treasuries), "
            "the price is a percent of par and point value uses the face value times 0.01 -- the two "
            "conventions are NOT interchangeable and the raw `contract_multiplier` field is a routinely "
            "wrong INT32 sentinel that must never be used directly."
        ),
        implementation=(
            "Derived once per contract from the CME definition record and audited by a dedicated QA "
            "script against the published specification before any experiment using that root can run."
        ),
        pointers=(
            "alpha_agent.data.contract_economics:derive_contract_economics",
            "scripts/phase_13_5c_contract_economics_qa.py",
        ),
    ),
)

CONCEPT_LIBRARY: dict[str, Concept] = {c.concept_id: c for c in _CONCEPTS}

#: gate `ReasonCode` fail value -> the Concept that explains it. Built from
#: the library itself (never a second hand-maintained mapping) so a concept's
#: `gate_fail_code` and this index can never silently drift apart.
GATE_FAIL_CODE_TO_CONCEPT: dict[str, str] = {
    c.gate_fail_code: c.concept_id for c in _CONCEPTS if c.gate_fail_code is not None
}


def list_concepts() -> tuple[Concept, ...]:
    return _CONCEPTS


def get_concept(concept_id: str) -> Concept | None:
    return CONCEPT_LIBRARY.get(concept_id)
