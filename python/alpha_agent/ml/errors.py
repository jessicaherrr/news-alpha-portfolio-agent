"""Typed, fail-loud errors for the Phase 15 ML layer.

Every one of these is a research-integrity condition. None is ever downgraded to
a warning, and none is ever caught-and-continued inside the framework.
"""
from __future__ import annotations


class MLProtocolError(RuntimeError):
    """Base class for a Phase 15 protocol violation."""


class MLDependencyMissing(MLProtocolError):
    """A predeclared model family needs a package that is not installed.

    Installing it requires network access and therefore explicit human approval
    (CLAUDE.md autonomous-execution STOP condition 1). The framework never
    attempts an install and never silently substitutes another model.
    """


class LeakageError(MLProtocolError):
    """A feature, transform or label reached information it may not see."""


class FoldIsolationError(MLProtocolError):
    """A fitted object crossed a fold boundary, or an OOF prediction came from a
    model that was trained on that row."""


class RandomSplitForbidden(MLProtocolError):
    """A shuffled / random / K-fold split was requested for market time series."""


class InsufficientEventsError(MLProtocolError):
    """A scope has too few labelled events to be trained honestly.

    Carries a typed :class:`~alpha_agent.ml.enums.MLRefusalReason` so the refusal
    is a machine-readable outcome (``INSUFFICIENT_TRAIN_EVENTS``, ...) rather
    than a prose message a caller has to parse. The predeclared gates are frozen:
    a scope below them is REFUSED, never trained on a lowered bar.
    """

    def __init__(self, message: str, *, reason: object | None = None, detail: dict | None = None):
        super().__init__(message)
        self.reason = reason
        self.detail = dict(detail or {})


class ManifestFrozenError(MLProtocolError):
    """An attempt to grow or mutate the frozen candidate manifest."""


class NetworkAccessForbidden(MLProtocolError):
    """Phase 15A performs no network I/O of any kind."""


class ForbiddenModelInput(LeakageError):
    """A column outside the explicit model-input ALLOWLIST reached the matrix.

    The label may look forward; the features may not. Audit/label-only fields --
    the label itself, realised episode PnL or duration, the exit timestamp,
    future fills, future roll counts, future MAE/MFE, a realised future regime --
    are never model inputs, and the protection is an allowlist rather than a
    blacklist so an unforeseen field fails closed.
    """


class ForbiddenDependency(MLProtocolError):
    """A model/search package outside the approved Phase 15 set is present or used.

    The Phase 15 model family is deliberately two: one simple linear
    probabilistic model and one controlled nonlinear tree. Expanding the model
    zoo before any Phase 15 result exists would trade statistical control for
    breadth.
    """
