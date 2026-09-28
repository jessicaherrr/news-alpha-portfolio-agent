# Scientific Validation

How a hypothesis becomes a trusted result — and why "no trade" is a valid,
expected outcome. Full framework detail lives in
`docs/RELIABILITY_VALIDATION.md`; registry mechanics live in
`docs/EXPERIMENT_REGISTRY.md`. This document is the concise public entry
point.

## Screening vs. validation

Screening (factor diagnostics, information coefficient) is a cheap filter
that narrows the candidate space. It is **not** evidence of a tradeable edge.
Only a hypothesis that survives the full validation gate — walk-forward OOS,
multiple-testing correction, holdout — can be called `PASS`. Conflating the
two is one of the most common ways backtests lie to their authors; this
system keeps them structurally separate.

## Train / validation / locked holdout

Every research program declares three non-overlapping periods:

- **Train/discovery** — where hypotheses are formed and screened
- **Validation** — where the walk-forward OOS evaluation runs
- **Locked holdout (2025+)** — never touched by research, features, ML
  training, or the LLM research agent, under any circumstance

The holdout boundary is enforced structurally: any code path touching
`ts >= 2025-01-01` fails loudly rather than silently degrading. The Research
Agent may never inspect the holdout before a final, explicitly-approved
evaluation phase.

## Multiple-testing correction

Testing many hypotheses/configurations inflates the false-positive rate if
left uncorrected. This system applies Benjamini–Hochberg / FDR correction
across the full predeclared family of trials — including trials that failed
to execute for engineering reasons, which enter the family at the
conservative `p = 1` rather than being silently dropped. Sensitivity reruns
and cross-market summaries are evidence, not trials, and are never counted in
the correction family.

## Deflated Sharpe ratio and null/placebo tests

Point-estimate Sharpe ratios are optimistic under search. The framework
applies a deflated Sharpe ratio adjustment and runs explicit null/placebo
tests — including take-rate-matched placebo comparisons for ML-filtered
signals — so a "PASS" reflects a result that beats a randomized control, not
just a positive backtest.

## Cost sensitivity and regime stability

A validated result must also survive: (1) reasonable perturbation of
commission/slippage assumptions, and (2) evaluation across distinct market
regimes rather than one favorable window. A strategy that only works at zero
cost, or only in one regime, does not pass.

## Parameter plateaus over isolated optima

The framework prefers a broad, stable region of good parameters over a single
sharp optimum — an isolated best point is much more likely to be a fitting
artifact than a real effect.

## Verdict authority

The LLM proposes hypotheses. It never computes PnL, decides fills, or issues
a pass/fail verdict — those are deterministic outputs of the C++ engine and
the typed validation gate. A `ReliabilityPolicy` and `ValidationSpec` define
the statistical semantics once, and changing them is treated as a frozen
research-semantics decision requiring explicit sign-off, not a routine code
change.

## Failure memory

Every experiment — including ones that failed for a genuine engineering
reason (`INVALID_EXECUTION`) versus a real scientific rejection — is
permanently recorded in the append-only experiment registry. An engineering
failure may be re-attempted under the same scientific identity; a real
scientific correction requires a new identity with an explicit
supersession/correction edge to the old one. Nothing is ever deleted or
silently overwritten.

## "No eligible portfolio" is a valid result

When no candidate signal or portfolio clears the validation gate, the honest
output is "no eligible portfolio" or "insufficient evidence" — not a forced
recommendation. The current registry state (queryable via the Experiment Log
page or `alpha_agent.registry`) reflects this: strategies that did not clear
the bar are recorded as first-class evidence, not discarded.

## Event-conditioned evidence limitations

News-alpha signals are conditioned on specific, often rare events. Sample
sizes per mechanism/path are frequently small, which the validation gate's
minimum-sample requirements enforce explicitly rather than allowing a
low-n result to appear more confident than it is. A rejection due to
insufficient sample size is reported as exactly that — not disguised as a
directional negative result.

## Current mixed-market limitation

The primary validated research program (Phase 13.5C / 15B) covers five CME
futures markets (ES, NQ, CL, GC, ZN) over 2018–2024. Cross-market and
cross-asset-domain (ETF/equity) generalization has not been independently
re-validated at the same depth — treat cross-domain claims as exploratory
until backed by their own validation record in the registry.
