"""Agent runtime-integration release, section 1 -- the holdout / observational-
context boundary.

This is a security boundary, tested adversarially:

A) a genuine historical SCIENTIFIC payload naming a locked-holdout date --
   bare ``YYYY-MM-DD``, or the date component of a full ISO-8601 datetime
   (``...T...Z`` / ``...T...+00:00``) -- must be rejected. The old guard's
   ``\\b``-anchored regex silently failed to match the datetime forms (no word
   boundary between a digit and a following ``T``); a scientific payload using
   full timestamps sailed straight through undetected.
B) a correctly-typed, current-day ``OBSERVATIONAL_CONTEXT_ONLY provider=IBKR
   feed=DELAYED ...`` note may reach the Research Agent's knowledge base (a
   `ResearchContext` / `OrchestratorConfig` built today, whose wall-clock is
   >= 2025-01-01) while staying excluded from every OTHER scientific plane --
   never disabling the guard to get there.

A pre-existing, independent defect this fix exposed (an ordinary
``created_at`` / ``generated_at`` bookkeeping wall-clock field being swept into
a holdout scan) is also covered, since it is the same "does this guard
correctly separate wall-clock bookkeeping from market data" property.
"""
from __future__ import annotations

import pytest
from alpha_agent.agents.context import ResearchContext, build_research_context
from alpha_agent.registry.holdout_guard import (
    BOOKKEEPING_TIMESTAMP_KEYS,
    HoldoutAccessError,
    assert_no_holdout_market_data,
)
from alpha_agent.ui import market_context

_UNIVERSE = ("NQ",)


def _sample_note(*, root: str = "NQ", provider: str = "IBKR", as_of: str) -> str:
    return (
        f"OBSERVATIONAL_CONTEXT_ONLY provider={provider} feed=DELAYED root={root} "
        f"last=20123.25 as_of={as_of} -- delayed market color for conversational "
        "context only; not scientific evidence, not a feature, not used in any "
        "validation or paper-eligibility decision."
    )


# ---------------------------------------------------------------------------
# A) scientific holdout payloads are still rejected -- including the ISO
#    datetime forms the old \b-anchored regex silently missed.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        "2025-06-15",
        "2025-06-15T10:30:00Z",
        "2025-06-15T10:30:00+00:00",
        "window ends 2025-06-15T00:00:00Z sharp",
        "2026-01-01T00:00:00Z",
    ],
)
def test_scientific_holdout_dates_are_rejected_bare_and_iso_datetime(value):
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data(value)


def test_scientific_holdout_datetime_rejected_inside_research_context_inputs():
    with pytest.raises(HoldoutAccessError):
        build_research_context(
            objective="uses 2025-06-15T10:30:00Z as a feature timestamp",
            market_universe=_UNIVERSE,
        )


def test_scientific_holdout_datetime_rejected_in_knowledge_base_too():
    """The knowledge_base exemption is content-gated -- a real scientific date
    inside it is still rejected, it does not get a free pass just for being at
    the declared path."""
    with pytest.raises(HoldoutAccessError):
        build_research_context(
            objective="t",
            market_universe=_UNIVERSE,
            knowledge_base=["NQ traded heavily on 2025-06-15T10:30:00Z."],
        )


# ---------------------------------------------------------------------------
# B) a correctly-tagged current-day observational note is NOT scientific data
#    and must be able to reach the agent's context even when today is
#    >= 2025-01-01.
# ---------------------------------------------------------------------------
def test_tagged_current_day_ibkr_note_reaches_research_context(monkeypatch):
    note = _sample_note(as_of="2026-09-12T14:23:00+00:00")
    ctx = build_research_context(objective="t", market_universe=_UNIVERSE, knowledge_base=[note])
    assert note in ctx.knowledge_base
    # and the frozen model_validator on ResearchContext itself also accepts it
    assert isinstance(ctx, ResearchContext)


def test_real_market_context_note_survives_the_holdout_guard(monkeypatch):
    """End-to-end with the actual note builder (not a hand-written string),
    frozen at today's real wall-clock via the module under test."""
    fixed_quote = _FakeQuote(root="NQ", last=20123.25, as_of="2026-09-12T14:23:00+00:00")
    monkeypatch.setattr(market_context, "get_quote", lambda root: fixed_quote)
    note = market_context.observational_context_note("NQ")
    assert note is not None
    ctx = build_research_context(objective="t", market_universe=_UNIVERSE, knowledge_base=[note])
    assert note in ctx.knowledge_base


class _FakeQuote:
    def __init__(self, *, root: str, last: float, as_of: str):
        from datetime import datetime

        from alpha_agent.marketdata.quote import FeedState

        self.root_symbol = root
        self.last = last
        self.bid = None
        self.ask = None
        self.volume = None
        self.provider = "IBKR"
        self.feed_state = FeedState.DELAYED
        self.received_timestamp = datetime.fromisoformat(as_of)


# ---------------------------------------------------------------------------
# adversarial: the exemption must not become a general bypass
# ---------------------------------------------------------------------------
def test_observational_note_outside_the_declared_path_is_still_scanned():
    """The same well-formed note, appearing somewhere OTHER than a declared
    knowledge_base path, gets no exemption -- proves this is a path-gated
    provenance rule, not a global string whitelist."""
    note = _sample_note(as_of="2026-09-12T14:23:00+00:00")
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data({"objective": note})


def test_observational_note_cannot_smuggle_a_second_real_date():
    """Appending genuine scientific content after the fixed closing sentence
    breaks the exact-match grammar, so the whole string is scanned normally
    and the embedded 2025 date is still caught."""
    note = _sample_note(as_of="2026-09-12T14:23:00+00:00") + " also see 2025-06-15T00:00:00Z"
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data(
            note,
            path="$.research_context_inputs.knowledge_base[0]",
            observational_context_paths=frozenset({"$.research_context_inputs.knowledge_base"}),
        )


def test_observational_note_replacing_the_real_date_with_a_holdout_date_is_rejected():
    """A note that is otherwise well-formed but whose own `as_of` genuinely is
    a locked-holdout SCIENTIFIC date is still just a well-formed observational
    note by grammar -- the exemption is about PROVENANCE (this is delayed
    color, not evidence), not about which calendar date it names. This is
    intentional: an as_of in the far future is exactly what "today" looks like
    from inside this repo, and the whole point of the exemption is that a
    correctly-tagged note is never scientific regardless of its date. Recorded
    here so a future reader does not "fix" this into a second holdout check
    that would break test B above."""
    note = _sample_note(as_of="2026-09-12T14:23:00+00:00")
    assert not build_research_context(
        objective="t", market_universe=_UNIVERSE, knowledge_base=[note]
    ) is None


def test_unknown_provider_never_exempted():
    note = _sample_note(provider="FAKEBROKER", as_of="2026-09-12T14:23:00+00:00")
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data(
            note,
            path="$.research_context_inputs.knowledge_base[0]",
            observational_context_paths=frozenset({"$.research_context_inputs.knowledge_base"}),
        )


def test_no_declared_path_means_the_guard_is_fully_closed_by_default():
    """Every existing call site that has not opted in keeps working exactly as
    before -- an empty `observational_context_paths` default never grants the
    exemption anywhere."""
    note = _sample_note(as_of="2026-09-12T14:23:00+00:00")
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data(note, path="$.anything")


# ---------------------------------------------------------------------------
# the independent bookkeeping-timestamp defect this fix exposed
# ---------------------------------------------------------------------------
def test_bookkeeping_timestamp_keys_are_never_scanned_by_default():
    for key in BOOKKEEPING_TIMESTAMP_KEYS:
        # would raise if the guard still descended into this key's value
        assert_no_holdout_market_data({key: "2026-09-12T00:00:00Z"})


def test_bookkeeping_exemption_is_opt_out_able():
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data(
            {"created_at": "2026-09-12T00:00:00Z"}, bookkeeping_timestamp_keys=frozenset()
        )


def test_unrelated_key_with_a_holdout_date_still_rejected():
    """Sanity check that the bookkeeping exemption is narrow: a sibling key
    with the same shape of value but a different name is unaffected."""
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data({"as_of_ts": "2025-06-15T00:00:00Z"})
