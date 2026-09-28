"""Alpha Discovery campaign, Part L -- 2025 holdout lifecycle tests (task
spec section 77). Every test below uses SYNTHETIC epoch labels
("test_2099", "test_2100", ...) -- never "2025" -- so the real epoch's
lifecycle file is never created or touched by the test suite."""
from __future__ import annotations

import ast
import inspect

import pytest
from alpha_agent.holdout import (
    AUTHORIZATION_ENV_VAR,
    FinalHoldoutManifest,
    HoldoutAuthorizationError,
    HoldoutLifecycle,
    HoldoutLifecycleError,
    HoldoutState,
)


def _manifest(epoch: str) -> FinalHoldoutManifest:
    return FinalHoldoutManifest(
        epoch_label=epoch, git_commit="deadbeef", code_version="test",
        registry_digest="digest1:test", candidate_experiment_identities=("experiment1:a",),
        strategy_spec_fingerprints=("stratfp1:a",), validation_policy_fingerprint="valreliabilitypolicy1:test",
        execution_config_identity="execconfig1:test", cost_config_identity="costconfig1:test",
        data_identity="valdataset2:test", allowed_strategy_families=("tsmom",),
    )


def test_module_has_no_market_data_loading_code():
    """Structural proof this is pure bookkeeping (module docstring's own
    claim): no import of the market-data layer anywhere."""
    import alpha_agent.holdout.lifecycle as mod

    tree = ast.parse(inspect.getsource(mod))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    forbidden = ("alpha_agent.data", "alpha_agent.adapters", "databento")
    for m in imported:
        assert not m.startswith(forbidden), f"holdout lifecycle module must not import {m}"


def test_default_state_is_sealed_and_not_accessed(tmp_path):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    record = lc.get("test_2099")
    assert record.state is HoldoutState.SEALED
    assert record.accessed is False
    assert record.status() == {"epoch": "test_2099", "status": "SEALED", "accessed": False}


def test_freeze_final_manifest_moves_to_final_eval_ready(tmp_path):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    record = lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))
    assert record.state is HoldoutState.FINAL_EVAL_READY
    assert record.accessed is False  # frozen != accessed
    assert record.manifest is not None


def test_freeze_final_manifest_requires_sealed(tmp_path):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))
    with pytest.raises(HoldoutLifecycleError):
        lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))  # already FINAL_EVAL_READY


def test_freeze_final_manifest_rejects_mismatched_epoch_label(tmp_path):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    with pytest.raises(HoldoutLifecycleError):
        lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2100"))


def test_consume_requires_final_eval_ready(tmp_path, monkeypatch):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    monkeypatch.setenv(AUTHORIZATION_ENV_VAR, "1")
    with pytest.raises(HoldoutLifecycleError):
        lc.consume("test_2099", authorization_token="1")  # still SEALED


def test_consume_requires_matching_authorization_token(tmp_path, monkeypatch):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))
    monkeypatch.delenv(AUTHORIZATION_ENV_VAR, raising=False)
    with pytest.raises(HoldoutAuthorizationError):
        lc.consume("test_2099", authorization_token="1")  # env var not set at all
    monkeypatch.setenv(AUTHORIZATION_ENV_VAR, "1")
    with pytest.raises(HoldoutAuthorizationError):
        lc.consume("test_2099", authorization_token="wrong-token")  # mismatched token


def test_consume_succeeds_with_matching_token(tmp_path, monkeypatch):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))
    monkeypatch.setenv(AUTHORIZATION_ENV_VAR, "explicit-authorization-value")
    record = lc.consume("test_2099", authorization_token="explicit-authorization-value")
    assert record.state is HoldoutState.CONSUMED
    assert record.accessed is True
    assert record.consumed_at


def test_consumed_cannot_return_to_sealed(tmp_path, monkeypatch):
    """Task spec section 62/77: CONSUMED cannot go back to SEALED. There is
    no method that allows this -- proven by every state-changing method
    refusing a non-matching precondition."""
    lc = HoldoutLifecycle(base_dir=tmp_path)
    lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))
    monkeypatch.setenv(AUTHORIZATION_ENV_VAR, "x")
    lc.consume("test_2099", authorization_token="x")
    with pytest.raises(HoldoutLifecycleError):
        lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))  # not SEALED anymore


def test_roll_forward_requires_consumed(tmp_path):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    with pytest.raises(HoldoutLifecycleError):
        lc.roll_forward("test_2099", new_epoch_label="test_2101")


def test_roll_forward_creates_a_fresh_sealed_successor(tmp_path, monkeypatch):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))
    monkeypatch.setenv(AUTHORIZATION_ENV_VAR, "x")
    lc.consume("test_2099", authorization_token="x")
    old, new = lc.roll_forward("test_2099", new_epoch_label="test_2101")
    assert old.state is HoldoutState.ROLLED_FORWARD
    assert old.successor_epoch_label == "test_2101"
    assert new.state is HoldoutState.SEALED
    assert new.accessed is False
    assert new.manifest is None


def test_roll_forward_never_retroactively_marks_a_used_epoch_pristine(tmp_path, monkeypatch):
    """Task spec section 65: never call the OLD epoch unseen again."""
    lc = HoldoutLifecycle(base_dir=tmp_path)
    lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))
    monkeypatch.setenv(AUTHORIZATION_ENV_VAR, "x")
    lc.consume("test_2099", authorization_token="x")
    old, _ = lc.roll_forward("test_2099", new_epoch_label="test_2101")
    assert old.accessed is True  # stays True forever -- never resets


def test_roll_forward_refuses_a_successor_with_existing_history(tmp_path, monkeypatch):
    lc = HoldoutLifecycle(base_dir=tmp_path)
    lc.freeze_final_manifest("test_2099", manifest=_manifest("test_2099"))
    monkeypatch.setenv(AUTHORIZATION_ENV_VAR, "x")
    lc.consume("test_2099", authorization_token="x")
    lc.freeze_final_manifest("test_2101", manifest=_manifest("test_2101"))  # successor already has history
    with pytest.raises(HoldoutLifecycleError):
        lc.roll_forward("test_2099", new_epoch_label="test_2101")


def test_manifest_hash_is_stable_and_content_addressed():
    a = _manifest("test_2099")
    b = _manifest("test_2099")
    assert a.manifest_hash() == b.manifest_hash()
    c = _manifest("test_2100")
    assert a.manifest_hash() != c.manifest_hash()


def test_manifest_is_frozen_immutable():
    from pydantic import ValidationError

    m = _manifest("test_2099")
    with pytest.raises(ValidationError):
        m.epoch_label = "test_9999"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The real "2025" epoch: never touched by this suite, and the honest status
# this campaign must report.
# ---------------------------------------------------------------------------


def test_real_2025_epoch_has_never_been_touched_by_this_repository():
    """No lifecycle file for the real "2025" epoch exists in this checkout --
    the campaign never created one, matching 'DO NOT consume real 2025'."""
    from alpha_agent.holdout.lifecycle import DEFAULT_LIFECYCLE_DIR

    path = DEFAULT_LIFECYCLE_DIR / "2025.json"
    assert not path.exists()


def test_real_2025_status_reads_sealed_not_accessed():
    lc = HoldoutLifecycle()
    record = lc.get("2025")
    assert record.state is HoldoutState.SEALED
    assert record.accessed is False
    assert record.status() == {"epoch": "2025", "status": "SEALED", "accessed": False}


def test_services_holdout_lifecycle_status_matches_the_lifecycle_module():
    from alpha_agent.ui import services

    status = services.holdout_lifecycle_status("2025")
    assert status == {"epoch": "2025", "status": "SEALED", "accessed": False}


def test_system_page_shows_the_honest_2025_status_line():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.system import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    captions = " ".join(c.value for c in at.caption)
    assert "2025 STATUS: SEALED" in captions
    assert "2025 ACCESSED: NO" in captions
