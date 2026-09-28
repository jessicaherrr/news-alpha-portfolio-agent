# ML Meta-Labeling and Regime Research Protocol (Phase 15)

**Status: Phase 15B.1 — EXECUTION LAYER BUILT, PRE-REAL-RUN CHECKPOINT.** The
complete production orchestration (`alpha_agent.ml.phase_15b_runner` and the
modules it drives, `scripts/phase_15b_research.py`) is written and proven end to
end on deterministic synthetic fixtures. No model has been trained on real data,
no real ML performance has been inspected, the real 2018–2024 event corpus has
not been built, no market data was acquired, no market-data network call was
made, no dependency was added, no production registry row was written, and 2025
has never been touched. Integration record:
[`outputs/phase_15/PRE_REAL_RUN_INTEGRATION_REPORT.json`](../outputs/phase_15/PRE_REAL_RUN_INTEGRATION_REPORT.json);
run counts: [`outputs/phase_15/REAL_RUN_ESTIMATE.json`](../outputs/phase_15/REAL_RUN_ESTIMATE.json);
real-path smoke: [`outputs/phase_15/REAL_PATH_SMOKE.json`](../outputs/phase_15/REAL_PATH_SMOKE.json).

Phase 15B.0 (below) remains the frozen pre-run adjudication plane; 15B.1 adds
only execution code and changes no threshold, search, model, feature, label,
split or adjudication semantic.

> **Phase 15B.1b — the regime transformation semantic is FROZEN
> (`REGIME_SEMANTIC_FROZEN = True`).** `REGIME_BASELINE` combines
> `realized_vol(20)` (≈ [0.02, 0.08], always positive) and
> `trend_strength(50,200)` (≈ [−40, +36], signed) — scales that differ by
> ~2000×, so the 15B.1 provisional raw row-wise mean was ≈ 1.000 correlated with
> `trend_strength` alone. A human froze the transformation **before any real ML
> fit and not from performance**: bucket **each input independently** into its
> empirical tertiles (1/3, 2/3 quantile cut points, linear interpolation) fit on
> the **current training fold only** — `realized_vol` → LOW/MID/HIGH, signed
> `trend_strength` → BEARISH/NEUTRAL/BULLISH — then take the **Cartesian
> product**, axis-0-major (`state = vol_bucket·3 + trend_bucket`), exposing **9
> deterministic one-hot columns** `regime_state_0 … regime_state_8`. No
> standardise-and-average, no PCA, no clustering, no performance tuning. A
> missing / non-finite regime input follows the typed feature-exclusion
> semantics (`FEATURES_MISSING_AT_DECISION`) — never zero-imputed. The whole
> transformation (algorithm/version, ordered input specs, per-axis bucket count,
> quantile rule + scope, Cartesian combination, signed-trend axis semantics,
> output-state ordering) is bound into `RegimeSpec.identity()`. This is **one
> predeclared transformation, not 9 hypotheses**: BH stays 60, DSR stays 3/8 and
> family breadth 380. Only the **40** regime-baseline `experiment_identity`
> values regenerated; the **20** NO_REGIME ablation identities are byte-identical
> (`c879559b…`). Manifest schema bumped `/3` → `/4`
> (`p15manifest1:e065ae99…`), the 15A.2 manifest and the 15B.0 identity map
> preserved as history. Supersession record:
> [`outputs/phase_15/REGIME_SEMANTIC_SUPERSESSION.json`](../outputs/phase_15/REGIME_SEMANTIC_SUPERSESSION.json);
> freeze:
> [`outputs/phase_15/PROTOCOL_FREEZE_REGIME_PLANE.json`](../outputs/phase_15/PROTOCOL_FREEZE_REGIME_PLANE.json),
> [`outputs/phase_15/PRERUN_IDENTITY_MAP_COMPLETE_REGIME_CORRECTED.json`](../outputs/phase_15/PRERUN_IDENTITY_MAP_COMPLETE_REGIME_CORRECTED.json).
> Checks 2 (placebo accounting) and 3 (offline real-path engineering smoke:
> feature columns, causal availability, C++ wiring incl. `ValidationDayPlan` +
> roll continuation + roll-close marks, no 2025, no network) both PASS.

This document is the frozen protocol that governs Phase 15B. Its authoritative,
hashed records are
`data/manifests/phase_15/ml_candidate_manifest_identity_corrected.json` (the
candidate manifest, Phase 15A.2) and
`data/manifests/phase_15/ml_validation_plane_adjudication_corrected.json` (the
validation / adjudication plane and the six identity planes, Phase 15B.0); this
document and `configs/phase_15_ml.yaml` are the human-readable mirrors. Editing
either changes nothing — regenerate with
`python scripts/phase_15a2_identity_correction.py` then
`python scripts/phase_15a3_validation_freeze.py` then
`python scripts/phase_15b_adjudication_freeze.py`.

> **Phase 15B.0 supersedes the Phase 15A.3 validation plane**
> `mlvalidationspec1:bfbd460c…` (commits `f7d421d`, `0289f37`), preserved
> unchanged on disk (`data/manifests/phase_15/ml_validation_plane.json`) and in
> git. Two **Phase-15-specific adjudication rules** are frozen on top of the
> inherited Phase 13 `ReliabilityPolicy`, **which is not modified** (it still
> fingerprints identically to Phase 13.5C's, `valreliabilitypolicy1:b0682e86…`):
>
> 1. **The predeclared BH family is always exactly 60 hypotheses.** A trial that
>    a typed refusal/failure keeps from producing a valid statistical result is
>    never silently dropped from the family: it enters BH accounting at the
>    conservative `p = 1` (never significant) and its actual typed
>    refusal/failure is preserved in the registry. `family_size_is_fixed`,
>    `refused_trial_bh_p_value = 1.0`; helper `phase_15_bh_family_p_values`.
> 2. **A Phase 15 PASS must demonstrate the ML filter adds value** to its
>    primary, not merely that the filtered strategy is profitable standalone. On
>    top of the inherited `ReliabilityPolicy` PASS it also requires
>    `meta daily Sharpe > primary daily Sharpe` on exactly aligned support, and
>    requires the already-predeclared 20-draw TAKE-rate-matched placebo control
>    to satisfy `empirical placebo p <= 0.05` with daily Sharpe as the comparison
>    statistic and `p = (1 + #{placebo >= observed}) / (20 + 1)`. The placebo
>    remains a null/control and never becomes a BH hypothesis.
>    (`alpha_agent.ml.adjudicate_phase_15_trial`, `placebo_empirical_p`.)
>
> Because `MLValidationSpec` changed (schema `ml-validation-spec/2`), its
> fingerprint, the `validation_spec_fingerprint` identity plane and **all 60**
> pre-run `experiment_identity` values moved. Every trial label is unchanged.
> Proof: `outputs/phase_15/PRERUN_IDENTITY_MAP_COMPLETE_ADJUDICATION_CORRECTED.json`
> (60 trials, 60 complete distinct identities) and
> `outputs/phase_15/PRERUN_REGISTRY_PREFLIGHT_ADJUDICATION_CORRECTED.json`
> (all 60 queried against the registry before any fit, 0 exact duplicates).
> **No research count changed**: 60 BH trials, 380 DSR breadth, threshold 0.50,
> event gates 100/20, 1,200 fits, 240 C++ evaluations.

> **Phase 15A.3 closes the last pre-training blocker.** Phase 15A.2 could prove
> the 60 predeclared trials produce 60 distinct *discriminators*, but not 60
> complete `experiment_identity` values: two of the six identity planes — the
> Phase 15 `ValidationSpec` and `ReliabilityPolicy` — did not exist, and
> inventing them would have been fabricated provenance. They exist now, and
> **nothing was chosen from ML performance, because there is none to choose
> from**. The `ReliabilityPolicy` is the frozen Phase 13/13.5C one *unchanged* —
> proved by fingerprint equality with the value committed in all 21 Phase 13.5C
> validation reports (`valreliabilitypolicy1:b0682e86…`), which is what the
> protocol's own "it replaces nothing … not the `ReliabilityPolicy`" means when
> it is made checkable. The `ValidationSpec` is Phase 15's own, family-wide, and
> every value in it is either restated by this protocol or inherited unchanged
> from Phase 13.5C. Proof:
> [`outputs/phase_15/PRERUN_IDENTITY_MAP_COMPLETE.json`](../outputs/phase_15/PRERUN_IDENTITY_MAP_COMPLETE.json)
> — 60 trials, 60 complete distinct pre-run identities — and
> [`outputs/phase_15/PRERUN_REGISTRY_PREFLIGHT.json`](../outputs/phase_15/PRERUN_REGISTRY_PREFLIGHT.json)
> — all 60 queried against the registry before any fit, 0 exact duplicates.
> **No count changed**: 60 BH trials, 3/8 per-trial and 380 family-level DSR
> breadth, TAKE threshold 0.50, event gates 100/20, 1,200 fits, 240 C++
> evaluations.

> **Phase 15A.2 supersedes the Phase 15A.1 manifest** `p15manifest1:82983cf4…`
> (commits `0bb6014`, `c375399`), preserved unchanged on disk and in git.
> Phase 15A/15A.1 built the pre-run ML `experiment_identity` around **one concrete
> `MLModelSpec`** — a single hyperparameter point. Under the frozen nested-CV
> protocol a point is selected *inside each outer training fold*, so it does not
> exist before the run and different folds may legitimately select different
> ones. Keying identity on a point therefore (a) made identity un-computable
> pre-run, defeating the duplicate detection it exists for, (b) named a
> configuration the protocol never uniquely produces, and (c) split each of the 60
> predeclared economic hypotheses into `H` registry-facing identities — **380 rows
> for 60 hypotheses**, silently disagreeing with the BH/FDR denominator of 60.
> The pre-run identity now carries the **whole predeclared model-search
> procedure** (family, full frozen grid in declaration order, selection objective,
> tie-break, seed) and the selected hyperparameters are **post-fit provenance**.
> Proof: [`outputs/phase_15/PRETRAIN_IDENTITY_MAP.json`](../outputs/phase_15/PRETRAIN_IDENTITY_MAP.json)
> — 60 trials, 60 distinct pre-run identities. **No count changed**: 60 BH
> trials, 380 DSR configurations, 1,200 model fits, 240 C++ evaluations,
> threshold 0.50, gates 100/20. Corrected **before any real ML training**.

> **Phase 15A.1 supersedes the Phase 15A manifest**
> `p15manifest1:3c28e643…` (commits `72d714b`, `7234b7a`), which is preserved
> unchanged on disk and in git. The Phase 15A inner loop swept a four-point TAKE
> threshold while scoring candidates with mean inner-fold **log loss** — a
> probability-based objective that is *independent of the decision threshold*. All
> four thresholds therefore scored identically on every fold and the "selection"
> resolved through a deterministic tie-break: the threshold was never actually
> selected by cross-validation, while still multiplying the DSR search-breadth
> term by four. **`TAKE_THRESHOLD` is now frozen at 0.50**, the sweep is removed
> from inner model selection, and the pre-run search accounting is rebuilt in
> [`outputs/phase_15/PRETRAIN_TRIAL_ACCOUNTING.json`](../outputs/phase_15/PRETRAIN_TRIAL_ACCOUNTING.json).
> The correction was made **before any real ML performance was observed**.

---

## 0. What Phase 15 is, and what it is not

```
primary StrategySpec  (deterministic, frozen, Phase 13.5C)
        |
        v
candidate signal  =  a primary trade EPISODE
        |
        +--> causal market / strategy / regime features (typed FeatureSpecs)
        |
        v
ML meta-labeler   ->   TAKE / SKIP
        |
        v
closed TargetSchedule
        |
        v
C++ Quant Core   ->   official Fill-derived economics
        |
        v
Reliability Validation   ->   Experiment Registry
```

The model is a **secondary filter** around a deterministic primary. It replaces
nothing: not the `StrategySpec`, not C++ execution, not Fill accounting, not
risk, not the `ReliabilityPolicy`, not the experiment registry. Python never
becomes the official PnL calculator.

It is **not** generic price prediction, not an LLM trading model, not black-box
end-to-end trading, not a rescue attempt for rejected strategies via
uncontrolled hyperparameter search, and not a search of 2023–2024 until something
profitable appears.

**The primary strategies have no validated alpha.** Phase 13.5C returned
**0 PASS / 19 REJECT / 2 INCONCLUSIVE** across 107 hypotheses. Phase 15 asks a
genuinely different question:

> Can a causal secondary model identify conditions under which a predeclared
> primary signal should be accepted or skipped?

That is a new registered hypothesis family, and its prior is not encouraging.

---

## 1. Data roles — Phase 15's own, truthful

Phase 13's history is unchanged and is not rewritten:
2018–2022 RESEARCH, 2023–2024 Phase 13 VALIDATION, 2025 LOCKED HOLDOUT.

But Phase 15 is a **new research program created after Phase 13's results were
observed**. Calling 2023–2024 an untouched out-of-sample set for Phase 15 would
be false: those years already informed which strategies exist to meta-label. So
Phase 15 declares its own roles:

| Role | Window |
|---|---|
| `PHASE_15_DEVELOPMENT_CORPUS` | `2018-01-01 .. 2024-12-31` |
| `LOCKED_FINAL_HOLDOUT` | `2025`, never accessed |

All Phase 15 development — features, labels, folds, selection, scoring,
economics — happens inside the development corpus, through predeclared
chronological nested walk-forward only.

`2025-01-01` appears in the config **exclusively as a half-open upper bound**.
`assert_exclusive_corpus_bound` enforces that it is *exactly* that boundary and
nothing else, and the allowance is granted by key, so a stray 2025 timestamp
anywhere else still fails loudly.

---

## 2. The primary trade episode

The label had to be aligned to the primary strategy's own closed trade episode
rather than an arbitrary future-return horizon. It can be, with **no change to
primary strategy semantics and no change to frozen C++ economics**, because both
halves already existed.

### 2.1 Episode boundaries — pure target intent

From the Phase 11 `TargetSchedule` alone, so they are computable before any
backtest runs. Let `sign(u) ∈ {-1, 0, +1}` over the strictly time-ordered rows.

* An episode **opens** at the first row whose sign is non-zero and differs from
  the sign in force immediately before it.
* It **closes** at the first later row of a different sign (a flat, or a
  reversal). The closing row is not part of the episode; on a reversal it is the
  next episode's entry row.
* A same-sign size change does **not** open a new episode.
* An episode still open at the end of the schedule is **unterminated** and yields
  no labelled event.

### 2.2 Episode economics — C++ Fill-derived

The engine has recorded `BacktestResult::trades` (and `::fills`) since Phase 06.
Phase 15A adds only a **read-only CSV export** of those audit trails
(`cpp/include/quant_core/trade_export.hpp`, CLI flags `--trades-out=` and
`--fills-out=`). No execution, accounting, fill, roll, risk or portfolio
semantics change, and the JSON result line is byte-identical with or without the
flags — which is asserted by a test, not merely intended.

**Trade attribution.** A closed trade belongs to the episode that was open when
the trade *opened*. This is total, deterministic and order-independent, and it
places a roll's close-leg and re-open-leg (`close_reason == "roll"`) inside the
episode that spans the roll, which is what "this episode's economics" means.

**Costs are the full round turn.** `ClosedTrade::costs_usd` is, by the ledger's
own definition (`cpp/src/position_ledger.cpp`), the commission of the **closing
fill only**. The opening fill's commission is real, is in the engine's headline
`costs_usd` / `net_pnl_usd`, and is attributed to no `ClosedTrade`. Summing
`ClosedTrade.net_pnl_usd` would therefore understate every episode's cost by its
entry commission — biasing every label optimistic, in exactly the direction that
makes a filter look useful. Correcting the attribution inside the ledger would
change the official accounting path, so Phase 15 does not do that: it sums
commissions from the **fill** trail instead. Gross PnL still comes from the
engine's `gross_pnl_usd`. Python sums; it never prices.

**Commission splits per contract** (Phase 15A.1). Commission is a flat
USD-per-contract charge, so a fill's commission is not handed whole to one
episode:

* the fill's per-contract rate is `commission_usd / quantity`;
* the contracts that **closed** exposure are exactly the `quantity` of the
  `ClosedTrade` the engine emitted for that fill, and belong to the episode
  owning that trade;
* every remaining contract **opened** exposure, and belongs to the episode open
  at the fill instant.

That split is what makes a direct LONG → SHORT flip attribute correctly. The
engine flips through zero with **one fill of two contracts**: one closes the
long, one opens the short, and `ClosedTrade::costs_usd` records the *whole* fill
commission against the outgoing trade. Charging it all to the outgoing episode
would overstate its cost and understate the incoming episode's — biasing two
labels in opposite directions at every reversal. A roll needs no special case and
gets none: the close leg carries a `ClosedTrade` on the outgoing instrument and
the re-open leg carries none on the incoming one, so both resolve to the **same**
episode, because the primary target direction never changed.

Six deterministic synthetic cases pin this down: enter→exit, same-sign size
change, direct LONG→SHORT flip, latency-delayed exit, an episode spanning a roll,
and roll close → roll reopen → later strategy exit.

### 2.2b Economic reconciliation

Nothing is silently dropped. `reconcile_episode_economics` asserts, for a fully
closed run:

```
Σ episode gross PnL − Σ episode-attributed commissions  ==  BacktestResult.net_pnl_usd
```

within the existing numerical tolerance, with no unexplained residual — proved on
synthetic fixtures *and* against a real C++ run. When the final episode is
deliberately left unterminated, the closed **labelled** economics, the
**excluded/open** economics and the **engine** economics are reported separately,
and the difference must be exactly the open episode. Unattributed trades and
unattributed commission are their own reported terms, so an attribution bug
surfaces as a non-zero residual rather than as a quietly wrong label.

### 2.3 The label

```
label = 1  iff  Σ gross_pnl_usd(trades in episode) − Σ commission_usd(fills in episode)  >  0.0
label = 0  otherwise
```

Strictly greater than zero: a zero-PnL episode labels 0, because taking it spent
risk and cost for nothing.

Episodes that cannot be labelled are **excluded with a typed reason, never
labelled 0** — `UNTERMINATED_AT_CORPUS_END`, `NO_ATTRIBUTED_FILLS`,
`FEATURES_MISSING_AT_DECISION`. Silently labelling an unobservable outcome 0
would teach the model that "no fill" means "bad trade".

---

### 2.4 Model input allowlist

A `MetaLabelEvent` deliberately carries the causal decision-time inputs *and* the
realised episode outcome the label came from. That is right for an auditable
record and wrong for a model matrix, so the boundary is explicit:
`assert_model_inputs_allowed` accepts a column **only** if it is one of the frozen
feature set's ordered aliases or a declared categorical (`root_symbol`,
`primary_side`). It is an **allowlist, not a blacklist** — an audit field nobody
anticipated, a debugging column or a join artifact all fail closed.

Refused by name and by construction: `label`, `episode_net_pnl_usd`, episode
duration, `exit_timestamp`, future fills, future roll counts, future MAE/MFE, a
realised future regime. It runs before a single fit inside `run_nested_cv`, so a
forbidden column cannot reach a matrix even transiently, and
`assert_no_audit_field_is_allowlisted` catches the one way an allowlist can fail:
somebody adding a forward-looking column to the feature set itself.

---

## 3. `MetaLabelEvent` and the causal invariant

```
feature_timestamp  ≤  decision_timestamp  <  label_start  ≤  label_end
```

The **label may look forward** — that is what a supervised target is. **No
feature may.** `assert_event_causality` enforces this per event and is called on
every event the builder emits, so a leaky event cannot reach a training set even
by accident.

`label_end_timestamp` is the episode's **information horizon**: the last
`ts_close_ns` of an attributed fill, *not* the exit decision. Purge and embargo
use that horizon, because it is genuinely the last instant the label depends on.

Fields: `event_id`, `root_symbol`, `strategy_family`, `strategy_fingerprint`,
`primary_signal_timestamp`, `primary_side`, `primary_target`, `episode_index`,
`decision_timestamp`, `feature_timestamp`, `label_start_timestamp`,
`label_end_timestamp`, `label`, `label_definition`, `episode_net_pnl_usd`
(audit only — never a feature, never a selection input), `development_fold`, and
the source fingerprints (`meta_label_spec`, `feature_set`, `dataset`,
`target_schedule_hash`).

---

## 4. Meta-label action semantics

The action space is **closed**: `TAKE` / `SKIP`. The model may never invent a
position size, and a probability never becomes continuous leverage. Probability
is stored for calibration and audit only.

* `TAKE` → the episode's rows keep the primary strategy's own targets.
* `SKIP` → the episode's rows are rewritten to the neutral target `0`.
* Rows outside any episode are untouched.

The meta-labeled schedule has **exactly the same row timestamps** as the primary
schedule; only target values differ. That is what makes the primary-vs-meta
comparison run on exactly aligned evaluation support — same bars, same warm-up,
same absent-row policy — so any difference in result is attributable to the
TAKE/SKIP decisions and nothing else. `assert_aligned_evaluation_support` and
`assert_closed_action_space` check both properties.

**One decision per episode**, taken at entry. No mid-episode re-decision in the
MVP: that would need its own label definition (the outcome of a *partial*
episode), which is a new frozen research semantic.

**Note on the schedule fingerprint.** `targets.csv` carries a `stratdsl1:`
fingerprint validated by the C++ parser, and Phase 15 does not touch that frozen
boundary — so a meta-labeled schedule keeps the primary's fingerprint. The two
are still provably distinguishable: `schedule_hash()` covers target values and
differs as soon as one episode is skipped. ML identity lives in the ML artifacts
and the registry, never in the target-schedule fingerprint.

---

## 5. Feature universe

Every ML feature is a typed Phase 09 `FeatureSpec`. There is no way to hand this
layer an ad-hoc pandas column — `MLFeature` refuses an unregistered kind. Each
feature therefore inherits a deterministic canonical name and fingerprint, an
availability timestamp, a declared `PriceDomain`, and signal-safety provenance,
and `assert_feature_set_signal_safe` **re-derives** those from the registry
rather than trusting the declaration.

Ordered (order is semantic — it is the model matrix column order):

| alias | kind | params |
|---|---|---|
| `mom_20` | `cum_log_return` | window 20 |
| `mom_120` | `cum_log_return` | window 120 |
| `rvol_20` | `realized_vol` | window 20 |
| `rvol_60` | `realized_vol` | window 60 |
| `atr_14` | `atr` | window 14 |
| `zscore_20` | `zscore` | window 20 |
| `dist_ma_50` | `dist_from_ma` | window 50 |
| `donchian_pos_55` | `donchian_pos` | window 55 |
| `trend_strength_50_200` | `trend_strength` | fast 50, slow 200 |
| `vol_percentile_20_252` | `vol_percentile` | window 20, lookback 252 |
| `volume_zscore_20` | `volume_zscore` | window 20 |

Categoricals: `root_symbol`, `primary_side`.
Price domain: back-adjusted — signal-safe and point-in-time, and never a fill
price.

**Excluded on purpose.** *Carry / term structure*: `alpha_agent.features.carry`
is a separate curve-construction API, not a registered `FeatureSpec` kind, and
the acquired Plan-B dataset has no point-in-time deferred-contract curve for
every root. Carry is permitted only where PIT-safe curve data supports it; here
it does not. *Execution-plane fields*: never a feature, however convenient.

---

## 6. Regime

**Deterministic causal regime — the frozen reference baseline (Phase 15B.1b).**
One predeclared transformation, `independent_axis_tertiles_cartesian_product/1`:

- **Axis 0** — `realized_vol(20)`: empirical tertiles (quantile probs 1/3, 2/3,
  linear interpolation) fit on the **current training fold only** →
  LOW / MID / HIGH.
- **Axis 1** — signed `trend_strength(50,200)`: the same train-fold tertile rule
  on the signed value → BEARISH (fast MA well below slow) / NEUTRAL / BULLISH.
- The two inputs are bucketed **independently on their own scales** — no
  standardise-and-average, no PCA, no clustering, no performance tuning (their
  scales differ ~2000×, so a raw mean would be trend-only).
- **regime = Cartesian product**, axis-0-major:
  `state = vol_bucket · 3 + trend_bucket` → **9 states**, exposed as 9
  deterministic one-hot columns `regime_state_0 … regime_state_8`.
- A missing / non-finite regime input follows the typed feature-exclusion
  semantics (`FEATURES_MISSING_AT_DECISION`) — it is **never imputed to zero**;
  `_regime_state_columns` raises if a non-finite value reaches the model matrix.

The whole transformation — algorithm/version, ordered input specs, per-axis
bucket count, quantile rule and scope, `digitize` tie convention, Cartesian
combination, signed-trend axis semantics, output-state ordering — is bound into
`RegimeSpec.identity()` (`identity_payload()["deterministic_transformation"]`).
It is **one predeclared transformation, not 9 hypotheses**: BH stays 60, DSR
stays 3/8 and family breadth 380.

Cut points are fit through the fold-local preprocessing path — no whole-dataset
standardisation; no quantile boundary ever sees a validation or future row.

**The regime ablation (`kind = NONE`) is a BH trial in its own right** — 20
identities, byte-identical across the 15B.1b freeze.

**A regime state is never given a discretionary economic name in output.**
`assert_regime_label_is_not_economic` still refuses "risk-on" / "crisis" style
vocabulary for any learned partition.

---

## 7. Model families and the frozen grid

Deliberately small and predeclared. No AutoML, no Bayesian optimisation, no
Optuna, no adaptive grid expansion, no "try another parameter because this
result looks bad". No neural network.

| family | backend | grid |
|---|---|---|
| `LOGISTIC_L2` | `LogisticRegression(solver=lbfgs)`, L2 by default | `C ∈ {0.1, 1.0, 10.0}` (3) |
| `HIST_GRADIENT_BOOSTING` | `HistGradientBoostingClassifier(early_stopping=False)` | `max_depth ∈ {2,3}` × `learning_rate ∈ {0.05, 0.10}` × `max_iter ∈ {100, 300}` (8) |

**TAKE threshold: frozen at `0.50`. It is not a search dimension.**

```
probability_take >= 0.50  ->  TAKE
probability_take <  0.50  ->  SKIP
```

The inner loop selects exactly one thing — the model hyperparameter
configuration — by minimising mean inner-fold log loss; ties break on the frozen
grid's declaration order, which is predeclared and deterministic. A probability
never becomes continuous position size. Thresholds such as 0.55 / 0.60 may
return only as explicitly predeclared **decision-policy** hypotheses in a new
search family, never as a free parameter of this one and never chosen on PnL, and
the inner objective is never replaced by an economic one.

`MLModelSpec` refuses any hyperparameter point outside this grid and
`ModelSearchSpec` refuses any grid that is not the frozen one in declaration
order; a non-0.50 threshold is refused by `assert_frozen_take_threshold` in both
`MLTrainingSpec` and `MLExperimentSpec`. `early_stopping` is off because an internal validation
cut would be a hidden, unaudited split.

**The model family stays at exactly two.** One simple linear probabilistic model
versus one controlled nonlinear tree is the scientific comparison; the zoo is not
expanded before any Phase 15 result exists. `lightgbm`, `xgboost`, `catboost`,
`optuna`, `tensorflow` and `torch` are all forbidden, and
`assert_no_forbidden_ml_packages()` fails loudly if one becomes importable.

> **Environment (2026-09-09, one-time approved install):** Python 3.12.14,
> numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.0. Recorded in
> [`outputs/phase_15/ENVIRONMENT_PROVENANCE.json`](../outputs/phase_15/ENVIRONMENT_PROVENANCE.json).
> Versions are provenance, never semantics: they enter no fingerprint. The
> leakage / fold-isolation / determinism tests still run on a pure-numpy
> `SYNTHETIC_DETERMINISTIC` backend that the manifest **refuses for research
> use**, so those guarantees hold independently of any vendor library.

---

## 8. Nested chronological validation

```
OUTER fold k   train [ .......... ] |embargo| test [ 20xx ]
                     purged of any event whose LABEL WINDOW overlaps the test block
INNER fold j   the same construction, built INSIDE outer-train only
```

* **No random split, ever.** `refuse_random_split()` is a hard tripwire.
* **Pooled roots share one temporal boundary.** Training pools ES/NQ/CL/GC/ZN
  with root as a categorical *feature*; pooling is a modelling choice and must
  never become a temporal one. If 2023 is an outer test block, 2023 events of
  **every** root are excluded from training — not merely those of the root whose
  economics are later measured. The boundary is a timestamp, so this is true by
  construction; `assert_global_temporal_isolation` asserts it anyway, **per
  root**, on every outer *and* inner fold, and returns a per-root census as
  evidence. "True by construction" is exactly the claim a pooled design breaks
  later, and a silent breach would make every out-of-sample number meaningless.
* Outer test blocks: calendar years **2020, 2021, 2022, 2023, 2024** — chosen for
  auditability, so a reader can check the design by eye and no boundary was tuned
  to a result. Expanding origin.
* Inner folds: 3, expanding origin, built strictly inside the outer training
  window. The outer test block is never visible to selection — that is what makes
  the outer estimate honest.
* Embargo: **10 calendar days** either side of every test block.
* Gates: `min_train_days=365`, `min_train_events=100`, `min_test_events=20`.

**Purge and embargo are different, and both are needed.**

*Purge* removes a training event whose label window `[label_start, label_end]`
overlaps the test block. A meta-label resolves when the primary episode's last
fill closes, which can be months after the decision; without purging, a training
row's outcome would be partly determined by the very period being scored.

*Embargo* additionally removes training events whose label window falls inside a
buffer around the test block, covering serial dependence the label window does
not itself express — shared volatility state, a shared roll, a straddled macro
shock.

Both operate on the **label window**, never on the decision timestamp alone;
using the decision timestamp would leave exactly the long-horizon leakage that
matters most here.

Inner-loop objective: **mean inner-fold log loss**, minimised. The inner loop
never selects on economics, so the C++ economic evaluation stays a genuinely
out-of-sample question.

---

## 9. Metrics

Predeclared ML diagnostics: ROC-AUC, PR-AUC, log loss, Brier score, calibration,
precision/recall, class balance, prediction coverage, TAKE rate. **Plain accuracy
is never the objective.**

These are **diagnostic only**. The scientific question is economic:

> Does the meta-labeled strategy improve the deterministic primary under the same
> C++ execution and accounting semantics?

Improvement is never claimed from classification metrics alone.

---

## 10. Economic evaluation

```
OOF predictions -> closed TAKE/SKIP TargetSchedule -> C++ Quant Core
    -> Fill-derived PnL -> Reliability Validation -> Experiment Registry
```

Compared on exactly aligned evaluation support: **PRIMARY** vs **META-LABELED**.
Reported per trial: net PnL, daily Sharpe, trades, fills/turnover, cost
sensitivity, max drawdown, TAKE rate, and the economic delta vs primary.

Cost scenarios `1.0× / 1.5× / 2.0×` — the frozen Phase 13.5C plan, unchanged, so
the ML comparison sits on the same cost surface as the primary it is judged
against. The execution, risk and cost planes are inherited from Phase 13.5C
unchanged, including the truthful `PassThroughRiskManager` limitation.

---

## 11. Multiple testing

Phase 15 is a **separate** multiple-testing family, `phase_15_ml.all`. The 107
Phase 13.5C trials are historical and unchanged and are **never** folded into
this denominator.

| quantity | value |
|---|---|
| BH/FDR trials | **60** (40 headline + 20 regime ablation) |
| BH q | 0.10 |
| DSR effective trial count | **380** (superseded: 7,600) |
| DSR per-trial breadth | 3 (`LOGISTIC_L2`), 8 (`HIST_GRADIENT_BOOSTING`) |
| placebo in denominator | no (a null never competes for significance) |
| cross-root summaries in denominator | no (descriptive) |
| diagnostic reruns in denominator | no (`SensitivityEvidence`) |

Headline trials are `primary_family × root × model_family` = 4 × 5 × 2 = 40.
Ablation trials are the same pipeline with the regime removed, tree family only:
4 × 5 = 20. A regime ablation is a genuinely different configuration, so it is a
trial, not a free diagnostic.

**Why BH and DSR use different counts.** BH counts the out-of-sample economic
results *actually inspected* — 60. DSR uses the inspected-**configuration**
count, because DSR is precisely the tool that deflates for how much **search**
produced a result: each trial's inner loop chooses among a finite predeclared
hyperparameter grid, and pretending that search did not happen would overstate
every Sharpe.

**Why 380 and not 7,600.** The superseded Phase 15A figure was
`Σ_trials (outer folds × H × 4 thresholds)` = `5 × 4 × 380`. Both multipliers
were wrong:

* **×4** counted a threshold sweep that selected nothing — the inner objective is
  threshold-independent, so the four thresholds were never distinguishable by the
  selection rule;
* **×5** counted each outer fold's refit of the *same* configuration as a
  separately inspected configuration. Refitting one predeclared configuration on
  another CV fold is a **training operation**, not a new model hypothesis.

`380 = Σ_trials H(trial) = 20×3 + 20×8 + 20×8` is what the research procedure
actually inspects. The per-trial breadth (3 or 8) deflates one trial's Sharpe;
380 is the family-level breadth used when the best result of the whole Phase 15
search is what is being judged. The fold factor lives in the *fit* totals
(1,140 inner + 60 outer), where it belongs. Full derivation with every invariant
checked: [`outputs/phase_15/PRETRAIN_TRIAL_ACCOUNTING.json`](../outputs/phase_15/PRETRAIN_TRIAL_ACCOUNTING.json).

A diagnostic rerun is not a new hypothesis — unless it inspects a genuinely new
model configuration, in which case it *is* a new trial and must be predeclared
before it is run.

**Phase 15B.0 adjudication rule 1 — the family is always exactly 60.** A trial
that a typed refusal/failure (`INSUFFICIENT_TRAIN_EVENTS`, a
`DATA_QUALITY_FAILURE`, …) keeps from producing a valid statistical result is
**never dropped** from the BH denominator. It enters BH accounting at the
conservative `p = 1` — it cannot be significant — and its real typed
refusal/failure is preserved in the registry as first-class evidence. Dropping
it would shrink the denominator every time the small-data risk (§17.2) bites,
silently inflating everyone else's significance. `family_size_is_fixed`,
`refused_trial_bh_p_value = 1.0`; the length-60 input vector is built by
`alpha_agent.ml.phase_15_bh_family_p_values`.

---

## 12. Experiment registry

Phase 15 creates **no ML registry**. It uses the existing
`data/registry/experiments.sqlite` (schema v5, identity schema v2) exactly as
Phase 14 defined it, so a Research Agent asks one question of one memory. Under
schema v5 a Phase 15 family run is one immutable EXECUTION ATTEMPT of each of the
60 pre-run identities; the a31f571 run's 60 attempts are `INVALID_EXECUTION`
(feature-pipeline defect), and the corrected run lands as a new VALID attempt
under the SAME 60 identities -- no new hypothesis, BH family still 60.

Mandatory pre-run order, before any Phase 15B experiment:

1. construct the pre-run `experiment_identity`;
2. `registry.find_exact_duplicate()`;
3. `registry.find_related()` / `FailureMemory.lookup()`;
4. refuse silent reruns — cite the prior result instead;
5. append completed evidence after evaluation.

Identity mapping (all pre-run, so "already run?" is answerable before any fit):

| Phase 14 component | Phase 15 value |
|---|---|
| `strategy_fingerprint` | `ml_composite_fingerprint` — the primary spec(s) **plus** the whole meta-label pipeline |
| `strategy_family` | `ml_meta_label.<primary_family>` |
| `root_symbol` | the root whose economics are adjudicated |
| `parameter_variant_identity` | the **predeclared model-search procedure** (family, full frozen grid in declaration order, selection objective, tie-break, seed) + the frozen TAKE threshold (0.50) — never a selected point |
| `dataset_fingerprint` | Phase 13.5C dataset identity, unchanged |
| `split_identity` | the `NestedCVSpec` identity |
| `feature_spec_fingerprint` | the ML feature set's **own** identity |
| execution / cost / risk | Phase 13.5C planes, unchanged |

`target_schedule_hash` (the meta-labeled schedule) and `report_fingerprint`
remain **post-run provenance and never enter identity** — otherwise the same ML
hypothesis, recompiled, would masquerade as a new experiment and slip past
duplicate detection. The Phase 14.2 rule applies unchanged: identity uses the
actual proposed feature semantics, never an inherited copy.

### The search procedure is identity; the selected point is provenance (15A.2)

Under nested CV the hyperparameter is chosen **inside each outer training fold**.
So there is no single point before the run, and often no single point after it
either. What a Phase 15 trial actually claims is:

> *this primary, filtered by a model of **this family** selected from **this
> frozen grid** by **this objective** under **this tie-break** with **this seed**,
> improves risk-adjusted net economics.*

Every clause of that is knowable from the proposal, so every clause is identity —
carried by `ModelSearchSpec`, which refuses any grid that is not the frozen one,
in declaration order (the order **is** the tie-break rule).

| quantity | where it lives | is it identity? |
|---|---|---|
| model family, frozen grid, objective, tie-break, seed | `ModelSearchSpec` | **yes** — pre-run |
| hyperparameters an outer fold selected | `OuterFoldRecord.selected_hyperparameters` | no — post-fit provenance |
| per-fold selected model / artifact fingerprint | `OuterFoldRecord` | no — post-fit provenance |
| inner objective values | `OuterFoldRecord.inner_objective_value` | no — post-fit provenance |
| `target_schedule_hash`, `report_fingerprint` | registry row | no — post-run provenance |

Provenance is still *audited*: `assert_selection_is_within_search` refuses a fold
that reports a configuration the predeclared procedure could not reach, and
`run_nested_cv` refuses to execute a search whose objective or seed disagrees with
the declared one. The executed search must be the declared search.

`alpha_agent.ml.trials` maps each frozen manifest trial to its `MLExperimentSpec`
— rebuilding the primaries with the frozen Phase 13.5C factories, per the Phase
14.2 rule — so the **60 predeclared hypotheses map one-to-one onto 60
registry-facing identities**, not the 380 that keying on a point would produce.
The six inherited plane fingerprints are supplied explicitly as
`MLIdentityPlanes`; there is no default, because a plane fingerprint is either
known or the identity is not yet claimable.

### 12.1 The six identity planes, and where each one comes from (15A.3)

They are all real now. `alpha_agent.ml.planes.assert_planes_are_real` refuses a
placeholder outright, so the Phase 15A.2 hashing pivot `PENDING_PHASE_15B` — which
lives only inside `trial_discriminator` and reaches no artifact — cannot leak into
a recorded identity.

| plane | value | where it comes from |
|---|---|---|
| `dataset_fingerprint` | `mldatasetplane1:95716bd0…` | the five roots' committed Phase 13.5C dataset identities, **pooled** |
| `validation_spec_fingerprint` | `mlvalidationspec1:bfbd460c…` | **new**: the family-wide Phase 15 `MLValidationSpec` |
| `reliability_policy_fingerprint` | `valreliabilitypolicy1:b0682e86…` | **inherited unchanged** from Phase 13/13.5C |
| `execution_config_identity` | `execconfig1:5b89fcc1…` | Phase 13.5C corrected execution plane, unchanged |
| `cost_config_identity` | `costconfig1:7ad0adc1…` | Phase 13.5C cost plane, unchanged |
| `risk_identity` | `riskconfig1:a24167a3…` | Phase 13.5C risk plane, unchanged |

*Why the dataset plane is pooled.* A Phase 15 pipeline trains **one** model over
ES/NQ/CL/GC/ZN with root as a categorical feature, so the data a Phase 15
hypothesis is about is all five roots' bars, not only the root whose economics
are adjudicated. Phase 13.5C's execution bars span 2018-01-01 up to the exclusive
locked-holdout boundary — exactly the `PHASE_15_DEVELOPMENT_CORPUS` — so each
per-root identity is Phase 13.5C's *unchanged*, and the plane only records that
all five are in scope at once. No bar was re-hashed and no new data exists.
All four Phase 13.5C baseline families' reports are read per root and a
disagreement raises: choosing one would be choosing which data the experiment was
about.

All 60 trials share all six planes, so the Phase 15A.2 uniqueness argument carries
over unchanged — two trials collide in `experiment_identity` **iff** they collide
in `trial_discriminator` — and it is now also checked directly on the complete
identities.

### 12.2 The pre-run registry preflight, run before any fit (15A.3)

All 60 identities were constructed from configuration alone and queried against
`data/registry/experiments.sqlite`: **60 queried, 0 exact duplicates, clear to
run**, with the nearest prior Phase 13.5C evidence recorded per trial.
`target_schedule_hash` and `report_fingerprint` are `NULL` in every row — they are
post-run provenance and are never back-filled to make a row look complete.
Artifact: [`outputs/phase_15/PRERUN_REGISTRY_PREFLIGHT.json`](../outputs/phase_15/PRERUN_REGISTRY_PREFLIGHT.json).

---

## 12A. The validation and adjudication planes (Phase 15A.3)

`alpha_agent.ml.adjudication` obeys one rule: **every gate is either explicitly
predeclared by this protocol, or inherited unchanged from the frozen Phase 13
default this protocol says it does not replace.** Nothing was chosen here, and in
particular nothing was chosen from ML performance — no model has been fitted, so
there is none to choose from, which is exactly why the planes are frozen *now*.

### The `ReliabilityPolicy` is Phase 13's, unchanged

Section 0 says Phase 15 "replaces nothing … not the `ReliabilityPolicy`". That is
made checkable rather than left as prose:
`assert_reliability_policy_inherited_unchanged` requires the Phase 15 policy to
fingerprint **identically** to the value committed in all 21 Phase 13.5C
validation reports. Only the cosmetic `policy_name` differs, and `identity()`
drops it by construction.

| gate | value | source |
|---|---|---|
| `fdr_q_threshold` | 0.10 | **PROTOCOL** — §11 BH q; checked to *agree* with the inherited default |
| `null_p_value_max` | 0.05 | inherited Phase 13 default |
| `dsr_min` | 0.95 | inherited Phase 13 default |
| `max_cost_degradation` | 0.5 | inherited; read off the unchanged 1.5× / 2.0× reruns |
| `fold_consistency_min` | 0.5 | inherited; evaluated over the five nested-CV outer blocks |
| `require_positive_oos_net_pnl` | true | inherited Phase 13 default |
| `minimum_sample` | 60 / 20 / 10 / 3 / 20 | inherited Phase 13 default |
| `require_regime_stability` / `require_cross_market` / `require_ablation_mechanism_value` | false | inherited; regime and cross-root evidence stay descriptive, as in Phase 13.5C |

`RELIABILITY_GATE_SOURCES` names the origin of **every** field, and
`assert_reliability_policy_gates_are_predeclared` refuses to run if the map and
`ReliabilityPolicy` disagree — an allowlist, so a gate added in a later phase
cannot enter the Phase 15 reliability plane with no stated provenance.

If the protocol's BH `q` and the inherited `fdr_q_threshold` ever disagreed, that
would be an unresolved protocol conflict and
`assert_protocol_and_policy_agree` raises rather than picking one. They agree.

### The `MLValidationSpec` is Phase 15's own, and family-wide

A validation spec names the corpus, the split, the cost surface, the null plan and
the multiple-testing family, and all five are Phase 15 quantities. It is a new
typed object rather than a reuse of the Phase 13 `ValidationSpec` for one
structural reason: the Phase 13 spec is **per strategy** (it carries a
`strategy_fingerprint` and a `ParameterNeighbourhood`), and a per-trial validation
fingerprint would break the shared-plane premise §12.1 rests on.

Everything it carries is sourced: the corpus and nested-CV design from the frozen
manifest; the 1.0× / 1.5× / 2.0× cost plan and the $100,000 capital base from
Phase 13.5C unchanged (asserted equal to `frozen_cost_plan()` by identity); the
BH/DSR family derived from the manifest rather than retyped; the placebo plan
(20 draws, conditional on gating-null rejection, never in the denominator).

**The null plan carries only the gating null.** §15 declares one — the centred
block bootstrap, "requiring no extra C++ runs" — and §14 budgets **240** C++
evaluations. The Phase 13 schedule time-shift null runs one C++ backtest *per
replicate*, so including it would need order 12,000 more runs and would contradict
the frozen accounting. `assert_null_plan_fits_frozen_compute_budget` enforces that
link instead of leaving it in a comment, and also refuses a plan with no gating
null at all.

**There is no economic hyperparameter-plateau gate, and there could not be one.**
The frozen budget evaluates economics for the *selected* pipeline only (60 primary
baselines + 180 meta-labeled runs), so no per-hyperparameter-point out-of-sample
economics exist to build a Phase 13 `ParameterStabilityResult` from. §16.8 lists a
"hyperparameter plateau check" under adjudication but declares no threshold for
one, so it is recorded as what it can be — a diagnostic over inner-fold objective
values — and `parameter_stability` is passed to `evaluate_policy` as `None`, which
is the frozen Phase 13 behaviour for absent evidence. The policy's own
`param_stability_*` thresholds stay at their inherited Phase 13 values, so if a
later predeclared phase does produce that evidence it is judged by a threshold
nobody invented after the fact. Declaring an *economic* plateau gate would be a
new frozen research semantic and needs approval, not a code change; see §17
limitation 10.

`MLValidationSpec` refuses, at construction: a moved TAKE threshold, a lowered
event gate, a corpus that reaches 2025, a claim that the holdout was accessed, a
verdict gated on ML classification metrics, a parameter-stability gate with no
predeclared threshold, and a gated `economic_delta_vs_primary` — which §10 reports
and declares no threshold for.

---

## 13. Reproducibility

Every training run records: model family, hyperparameters, random seed, training
window, evaluation window, feature fingerprints **and ordered feature identity**,
`MetaLabelSpec` fingerprint, preprocessing fingerprint, regime fingerprint, code
commit, dataset fingerprint, primary `StrategySpec` fingerprints, OOF prediction
hash, model artifact fingerprint, and the environment record.

Randomness is explicitly seeded. Same inputs + same seed ⇒ identical predictions
and an identical report fingerprint (asserted by test).

Typed models: `MetaLabelSpec`, `MLFeatureSetSpec`, `RegimeSpec`, `MLModelSpec`,
`ModelSearchSpec`, `MLTrainingSpec`, `MLExperimentSpec`, `MLIdentityPlanes`,
`MLValidationSpec`, `MLMultipleTestingPlan`, `MLPlaceboPlan`,
`MLBaselineComparisonPlan`, `MLPredictionFrame`, `MLTrainingReport`.
Pickle/joblib is never the *only* authoritative representation: the deterministic
metadata, version, parameters, ordered feature list and fingerprints are always
persisted alongside.

---

## 14. Compute estimate

Derived from the manifest itself (`scripts/phase_15a1_pretrain_audit.py`), not by
hand. Every quantity below is a **different** thing; conflating any two is how a
search's real breadth gets understated.

| # | quantity | count | derivation |
|---|---|---|---|
| 1 | BH/FDR economic hypotheses | **60** | pipelines(12) × roots(5) = 40 headline + 20 ablation |
| 2 | distinct fitted pipelines | **12** | primary families(4) × (model family, regime)(3) |
| 3 | model hyperparameter configurations | **76** | Σ over pipelines of H: 4×3 + 4×8 + 4×8 |
| 3b | threshold configurations | **1** | frozen at 0.50, not searched |
| 4 | inner-CV fits | **1,140** | Σ (outer 5 × inner 3 × H) = 15 × 76 |
| 5 | outer OOF-producing fits | **60** | pipelines(12) × outer folds(5) |
| — | **model fits, total** | **1,200** | 1,140 + 60 |
| 6 | **DSR inspected configurations** | **380** | Σ over 60 trials of H(trial) × 1 |
| 6b | configuration × fold inspections | 1,900 | bookkeeping only — *never* a DSR term |
| 7 | **C++ economic evaluations** | **240** | 60 primary baseline + 180 meta-labeled |
| 8 | conditional placebo (ceiling) | 1,200 | trials(60) × draws(20), expected ≈ 0 |
| — | diagnostic reruns planned | 0 | any that prove necessary are `SensitivityEvidence` |

A fitted pipeline is per `(primary_family, model_family, regime_kind)`: the model
is trained **pooled across the five roots** with root as a categorical feature,
then evaluated per root. Five per-root trials share one fitted pipeline but are
five separately inspected economic results, so all five stay in the BH
denominator.

Placebo runs are conditional on a trial rejecting its gating null. Given the
Phase 13.5C prior (0 PASS in 107 trials) the expected placebo count is near zero;
1,200 is the ceiling, not a plan.

---

## 15. Nulls and placebo

* **Gating null**: centred block bootstrap on the meta-labeled OOS daily net
  return series — the frozen Phase 13 machinery, reused, requiring no extra C++
  runs.
* **Placebo**: a random meta-labeler matched on the realised TAKE rate,
  C++-executed, 20 draws, baseline cost, conditional on the gating null being
  rejected. This is the control that separates "the model chose well" from "the
  model simply traded less".

---

## 16. Phase 15B execution plan

Phase 15B **must not begin until the dependency question is resolved** (§17).

> **Phase 15B.1 built the whole execution layer for this plan** and proved it
> end to end on deterministic synthetic fixtures. `alpha_agent.ml.phase_15b_runner`
> drives `.corpus` → `.engine_io` → `.feature_matrix` → `.economics` /
> `.diagnostics` → `.placebo` → `.family_adjudication` → `.registry_append` →
> `.phase_15b_report`, and `scripts/phase_15b_research.py` exposes `--preflight`,
> `--estimate-only`, `--synthetic-integration` and `--run-real`. The real data
> path (`.phase_15b_real_path`: the frozen Phase 13.5C `_Phase135cAdapter`, the
> Feature Engine on the daily signal series, and `CppEngineRunner` with
> `--trades-out`/`--fills-out`) is built but not invoked; `--run-real` (Phase
> 15B.2) is a mechanical invocation. 15B.1 touched no real corpus, fitted no real
> model, wrote no production registry row and never touched 2025. Record:
> [`outputs/phase_15/PRE_REAL_RUN_INTEGRATION_REPORT.json`](../outputs/phase_15/PRE_REAL_RUN_INTEGRATION_REPORT.json).

0. **Done (Phase 15A.3 + 15B.0).** The validation / adjudication planes are
   frozen, all six identity planes are real, the 60 predeclared trials carry 60
   complete pre-run identities, and the registry preflight ran clean. Nothing
   below may change a gate.
1. **Approve the ML dependency.** Install `scikit-learn` (and `scipy`), or
   decline and stop. This is the only network action Phase 15B needs.
2. **Build the event corpus, offline.** For each of the 20 primaries, compile the
   frozen Phase 13.5C schedule over 2018–2024, run the C++ CLI once with
   `--trades-out=` and `--fills-out=`, extract episodes, attribute trades and
   fills, and emit `MetaLabelEvent`s. 20 C++ runs. **Report the per-scope event
   counts and stop if the minimum-event gates fail** — see §17.
3. **Build the feature matrix** through the Feature Engine at each decision
   timestamp; exclude events with a missing feature, with the typed reason.
4. **Register intent.** Already done pre-run and committed: build the
   `MLExperimentSpec` with `experiment_spec_for_trial` supplying the frozen
   `phase_15_identity_planes()`, compute the pre-run identity, and query
   `find_exact_duplicate` / `find_related` — `phase_15_registry_preflight` does
   all 60 at once. Re-run it and refuse any duplicate; it needs no fitted model,
   which is what makes it worth doing before step 5.
5. **Train.** Run the nested chronological protocol per pipeline. Emit OOF
   predictions and the training report. Assert OOF provenance and fold isolation
   before looking at anything.
6. **Evaluate ML diagnostics** (calibration first). Diagnostic only.
7. **Evaluate economics.** Build meta-labeled schedules, run the C++ core at all
   three cost scenarios, alongside the primary baselines on the same support.
8. **Adjudicate** under the frozen `MLValidationSpec` and the inherited
   `ReliabilityPolicy` (§12A) — no gate may be chosen, tuned or added at this
   point. BH/FDR over 60 at q = 0.10 — one trial per pre-run identity, so the
   denominator and the registry agree by construction — DSR at 380 inspected
   configurations (3 or 8 per trial) against `dsr_min = 0.95`, the gating null at
   `null_p_value_max = 0.05`, cost sensitivity against `max_cost_degradation =
   0.5`, outer-fold consistency against `fold_consistency_min = 0.5`, plus
   descriptive regime and cross-root evidence. There is no threshold plateau to
   check — the threshold is frozen, not searched — and the hyperparameter plateau
   check is a diagnostic over inner-fold objective values, not a gate (§12A).
   **Phase 15B.0 adjudication rule 2**: a base `PASS` is kept only if the ML
   filter also demonstrably *adds value* — `meta daily Sharpe > primary daily
   Sharpe` on exactly aligned support (§4), **and** the placebo (step 9) clears
   `empirical p <= 0.05`. Otherwise the trial is `REJECT` with a Phase-15 reason
   code. A typed refusal yields `REFUSED` and stays in the family at `p = 1`.
   `alpha_agent.ml.adjudicate_phase_15_trial`.
9. **Placebo**, run for every trial that rejected its gating null (so, every
   base `PASS`). 20 TAKE-rate-matched draws, C++-executed, baseline cost;
   `p = (1 + #{placebo daily Sharpe >= observed}) / (20 + 1)`. A control, never
   a BH hypothesis.
10. **Append to the registry** — every trial, including every failure. Failures
    are first-class evidence and are never deleted. The per-outer-fold selected
    hyperparameters are appended as provenance on the result, never as a second
    experiment.
11. **Report.** 2025 remains locked; the final holdout is a later decision that
    requires its own freeze and human approval.

---

## 17. Limitations and unresolved issues

**1. The model family is deliberately narrow.** `scikit-learn` and `scipy` are
installed under a one-time approved install; nothing else is. Two families — one
linear probabilistic model, one controlled nonlinear tree — is statistical
control, not a claim that LightGBM or a boosted alternative would be worse. It is
a real limitation on how much of the model space Phase 15 explores, and it is
accepted on purpose: expanding the zoo before any Phase 15 result exists would
trade search discipline for breadth exactly where discipline matters most.

**2. Event counts are small, and this is the dominant scientific risk.** Phase
13.5C recorded 10–53 closed trades per (root, family) over the two-year
validation window. Extrapolated over the seven-year corpus, a single (family,
root) primary yields roughly 30–90 *episodes* — far too few to train an honest
classifier. That is precisely why the model is **pooled across the five roots**,
giving order 150–450 events per family. Even pooled, this is a small-data
problem, and the predeclared gates (`min_train_events=100`,
`min_test_events=20`) may well refuse some families outright. That refusal is a
legitimate, predeclared outcome recorded as an insufficient-evidence failure —
**not** a reason to relax the gate after seeing it.

**3. Meta-labeling cannot rescue a strategy with no edge.** A filter can only
reallocate exposure among signals the primary already generates. Phase 13.5C
found no primary with validated alpha, so the prior on Phase 15 producing a PASS
is low, and a PASS on this evidence base should attract more scepticism than
enthusiasm, not less.

**4. Skipping reduces trade count, which weakens statistical power.** A
meta-labeled strategy trades strictly less than its primary. Fewer trades means a
noisier Sharpe on the same window — so a favourable point estimate is easier to
obtain and harder to trust. This is the specific failure mode the TAKE-rate-matched
placebo exists to detect.

**5. One decision per episode.** No mid-episode re-decision, no early exit, no
sizing. Deliberate: each of those needs its own label definition, which is a new
frozen research semantic requiring approval.

**6. Costs.** Episode costs are the full round-turn commission from the fill
trail, which is correct under the frozen assumptions (slippage 0, spread 0). If a
future phase turns on slippage or spread, the fill trail carries `slippage_ticks`
but the episode cost model here does not price it — that would be a new semantic.

**7. `HoldoutAccessError` inside a pydantic validator surfaces as
`ValidationError`.** `HoldoutAccessError` subclasses `AssertionError` so the Phase
14 registry can catch it, and pydantic wraps `AssertionError`. The message is
unambiguous either way, and every builder guards *first* so normal callers get
the typed error; the in-model check is defence in depth. Worth knowing when
writing an `except` clause.

**8. Silver Bullet is out of scope** for the MVP: its native-1m cadence implies a
different label horizon, purge/embargo scale and event density. Meta-labeling it
is a separate predeclared family for a later phase.

**9. Carry / term-structure features are unavailable**, not merely unused: the
acquired dataset has no PIT-safe deferred-contract curve for every root.

**10. There is no economic hyperparameter-plateau gate.** §16.8 lists a
"hyperparameter plateau check" under adjudication, but the frozen protocol
declares no threshold for one, and the frozen compute budget (240 C++ evaluations
= 60 primary baselines + 180 meta-labeled runs) produces no per-hyperparameter-point
out-of-sample economics to build a Phase 13 `ParameterStabilityResult` from. Phase
15A.3 therefore records the check as a **diagnostic** over inner-fold objective
values and gates on nothing, which is the only reading consistent with the frozen
artifacts — inventing a threshold would be exactly the post-hoc rule-making the
whole design exists to prevent. It is a real limitation, and it is the one gate a
sceptical reader might reasonably want and not find: Phase 13.5C's canonical
trials *were* judged against a parameter neighbourhood, and Phase 15's are not.
Adding one is a new frozen research semantic — it needs a predeclared threshold, a
predeclared evidence source, and its own C++ budget — and it must be declared
**before** Phase 15B training, never after a result is seen. The policy's
inherited `param_stability_*` values are left untouched so that if such evidence
ever is produced, it is judged by a threshold nobody invented after the fact.

---

## 18. Phase 15A / 15A.1 / 15A.2 / 15A.3 verification summary

| claim | how it is verified |
|---|---|
| labels look forward, features do not | `assert_event_causality`, per event, on every built event |
| feature availability ≤ decision | causal invariant test |
| preprocessing sees training fold only | `FoldFittedTransform` records its fit rows; fit outside them raises |
| no whole-dataset fit | `assert_no_whole_dataset_fit` |
| no random split | `refuse_random_split` tripwire |
| purge/embargo prevent label leakage | `assert_fold_is_causal`; longer horizon ⇒ strictly fewer training rows |
| OOF never from a model trained on that row | `assert_oof_provenance` |
| deterministic seed ⇒ deterministic output | prediction hash + report fingerprint equality |
| different semantics ⇒ different identity | 8 parametrised identity cases |
| identity is constructible before fitting | `MLExperimentSpec` holds no model point; identity computed from configuration alone |
| changing the frozen search changes identity | 7 procedure perturbations: family, seed, grid extended / trimmed / reordered, objective, tie-break |
| a post-fit selected hyperparameter changes nothing | every grid point checked absent from identity, composite fingerprint, variant identity and label |
| a non-frozen search cannot be constructed | `ModelSearchSpec` and `MLExperimentSpec` both refuse an expanded / trimmed / reordered grid |
| a selection outside the search is refused | `assert_selection_is_within_search`; `run_nested_cv` refuses a mismatched objective or seed |
| 60 trials ⇒ 60 distinct pre-run identities | full identities under a fixed plane set, **and** plane-free discriminators; committed in `PRETRAIN_IDENTITY_MAP.json` |
| duplicate query works before fitting | `query_registry_before_running` on an empty registry, from the proposal alone |
| artifact preserves ordered feature identity | `artifact_metadata()["feature_order"]` |
| Phase 13 registry entries unchanged | schema v5 leaves the identity formula and every Phase 13/14 authoritative result unchanged; 167 authoritative hypotheses (107 + 60) |
| 2025 guard fires | timestamp, event, timeline, manifest and exclusive-bound cases |
| no network | `forbid_network` tripwire; nothing in Phase 15A/15A.1/15A.2 performs market-data I/O |
| threshold is frozen at 0.50 | `assert_frozen_take_threshold`; no `threshold_grid` survives anywhere; every OOF row carries 0.50 |
| the inner loop selects hyperparameters only | `INNER_SELECTION_DIMENSIONS`; report counts distinct configurations, not fold repeats |
| the search accounting is internally consistent | `trial_accounting_payload` asserts 15 invariants and refuses to emit an inconsistent artifact |
| the superseded manifests are preserved | `3c28e643…` (15A) and `82983cf4…` (15A.1) still on disk with their original schema versions; the freeze scripts refuse to rewrite them |
| every reliability gate names its source | `RELIABILITY_GATE_SOURCES` is exhaustive by assertion; an unsourced gate fails closed |
| the `ReliabilityPolicy` is Phase 13's, unchanged | fingerprint equality with the value in all 21 committed Phase 13.5C validation reports |
| a tuned gate is refused | a loosened `dsr_min` changes the policy fingerprint and raises |
| a protocol/policy disagreement is escalated, not resolved | `assert_protocol_and_policy_agree` raises on a mismatched BH `q` |
| the null plan fits the frozen compute budget | a C++-rerunning null is refused against the 240-evaluation accounting; a plan with no gating null is refused too |
| the cost surface is Phase 13.5C's, unchanged | identity equality with `phase_13_5c_matrix.frozen_cost_plan()` |
| a lowered event gate / moved threshold / gated diagnostic is refused | `MLValidationSpec` validators, six cases |
| no plane is a placeholder | `assert_planes_are_real` refuses `PENDING…`, and every plane is a real `<prefix>:<sha256>` |
| a moved plane moves the identity | all six planes perturbed, one at a time |
| 60 trials ⇒ 60 COMPLETE distinct pre-run identities | `assert_sixty_complete_identities`; committed in `PRERUN_IDENTITY_MAP_COMPLETE.json` |
| post-run provenance stays NULL pre-run | `target_schedule_hash` / `report_fingerprint` `None` in all 60 rows |
| the duplicate query runs before any fit | 60 queried against the real registry, 0 duplicates, `clear_to_run` |
| the Phase 13.5C registry is unchanged | schema v5; the 107 Phase 13.5C authoritative results and 21 supersessions are byte-identical |
| superseded accounting is preserved | `outputs/phase_15/PRETRAIN_TRIAL_ACCOUNTING.json` still reads `phase: 15A.1` |
| event gates unchanged, refusal is typed | `min_train_events=100` / `min_test_events=20`; `MLRefusalReason.INSUFFICIENT_TRAIN_EVENTS` |
| episode attribution is exact | six deterministic cases: enter/exit, same-sign resize, LONG→SHORT flip, latency-delayed exit, roll-spanning, two rolls then exit |
| roll fills stay in the same episode | roll close + reopen never create a synthetic episode |
| economics reconcile with the engine | Σ episode gross − Σ attributed commission == `net_pnl_usd`, on synthetic and real C++ runs |
| an open episode explains the residual exactly | closed labelled / excluded-open / engine economics reported separately |
| audit and label fields cannot enter `X` | explicit **allowlist**; 15 audit fields parametrised; unforeseen columns fail closed |
| pooled roots share one temporal boundary | `assert_global_temporal_isolation` per root, per outer and inner fold |
| no real training in 15A / 15A.1 / 15A.2 | no model is fitted to the real event corpus; only synthetic fixtures |
| no forbidden ML package | `assert_no_forbidden_ml_packages`; lightgbm/xgboost/catboost/optuna/tensorflow/torch all absent |
| the C++ export is additive | identical JSON with and without `--trades-out` / `--fills-out` |
| export reconciles with official economics | Σ gross − Σ commission == engine `net_pnl_usd` |
