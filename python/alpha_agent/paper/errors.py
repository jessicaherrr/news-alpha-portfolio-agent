"""Typed errors for Phase 21 paper trading. Every refusal here is loud and
specific -- there is no code path that silently downgrades an ineligible
strategy into a runnable paper trade."""
from __future__ import annotations


class PaperTradingError(RuntimeError):
    """Base class for every Phase 21 paper-trading control-flow failure."""


class PaperTradingEligibilityError(PaperTradingError):
    """A strategy may not be routed to paper trading yet.

    Raised when the registry has no experiment matching the given key, the
    experiment's authoritative result is not an unambiguous committed PASS
    (CLAUDE.md: only after a strategy passes frozen validation), the
    experiment has been superseded, or its ``StrategySpec`` cannot be
    deterministically reconstructed and fingerprint-verified from committed
    registry provenance (Phase 21 MVP scope -- see
    :mod:`alpha_agent.paper.eligibility`).
    """


class PaperRunStateError(PaperTradingError):
    """An operation was attempted against a paper run in the wrong state (e.g.
    stepping a STOPPED run, or double-stopping one)."""


class PaperDataWindowExhausted(PaperTradingError):
    """The run's predeclared replay window has no further trading days to
    advance into. Raised, never silently absorbed, so a caller cannot mistake
    "out of data" for "nothing happened this step"."""


class PaperReplayDivergence(PaperTradingError):
    """A step's full-window replay disagrees with the ledger's already-committed
    fill history for the same run (Phase 21.1).

    Every paper-trading step replays the WHOLE window from scratch, so the
    historical PREFIX of fills a later (longer) replay produces must be
    byte-identical to the fills an earlier (shorter) replay already committed.
    Raised when the new replay has fewer fills than already committed, or any
    committed fill's fields differ from the corresponding replayed fill.
    Raising this means NOTHING was appended to the ledger and the run's
    watermark/status were not advanced -- see
    :func:`alpha_agent.paper.replay_integrity.assert_replay_prefix_consistent`
    and :meth:`alpha_agent.paper.ledger.PaperLedger.commit_step`.
    """
