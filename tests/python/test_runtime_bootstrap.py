"""Tests for the HOME TERMINAL + RUNTIME CONNECTIVITY pass, mission section
1: a normal `streamlit run python/alpha_agent/ui/app.py` must discover a
local `.env`'s `DATABENTO_API_KEY` WITHOUT the user manually running
`source .env` first, and ordinary unit tests must stay offline regardless.

These tests deliberately never call the REAL `load_dotenv_if_available()`
unmocked -- doing so would load this checkout's actual `.env` (if any) into
this pytest process's environment for the rest of the session, which is
exactly the class of bug this pass fixes (see `tests/python/conftest.py`'s
own module docstring for the full incident). `test_databento_market_data_
provider_live.py` remains the one deliberate, explicit, real place that
happens for real.
"""
from __future__ import annotations

import pytest
from alpha_agent.marketdata.databento_schemas import DatabentoCapability
from alpha_agent.ui import app, databento_context


def test_bootstrap_wires_to_the_canonical_env_loader(monkeypatch):
    """`app.bootstrap_local_runtime_environment` -- called once, explicitly,
    at real app startup (`app.main`) -- is the ONE canonical place a local
    `.env` is loaded. Proven by spying on the loader it delegates to,
    WITHOUT ever actually loading a real `.env` into this test process."""
    from alpha_agent.knowledge import connector_support

    calls = []
    monkeypatch.setattr(connector_support, "load_dotenv_if_available", lambda: calls.append(1))
    app.bootstrap_local_runtime_environment()
    assert calls == [1]


def test_main_bootstraps_before_touching_streamlit_or_views(monkeypatch):
    """The bootstrap call happens FIRST in `main()` -- before any page
    module (which may indirectly construct a Databento provider on first
    real use) is even imported. Verified by asserting bootstrap runs even
    when `streamlit` itself is unavailable (which makes `main()` raise
    immediately afterward) -- proving it is not gated behind that import."""
    from alpha_agent.knowledge import connector_support

    calls = []
    monkeypatch.setattr(connector_support, "load_dotenv_if_available", lambda: calls.append(1))

    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "streamlit":
            raise ImportError("streamlit deliberately unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="Install UI extras"):
        app.main()
    assert calls == [1], "bootstrap must run even though the streamlit import failed right after it"


def test_ordinary_unit_tests_stay_offline_even_with_a_key_present(monkeypatch):
    """Mission section 1/20: "normal unit tests do not perform real
    Databento calls" -- even if some OTHER code path (or a leaked `.env`)
    put a real-shaped key into `os.environ`, `databento_context`'s public
    functions must still be the safe conftest.py default here, never the
    real provider. This is the actual regression a prior full-suite run
    exposed (one test hung 20+ minutes on an unmocked real call)."""
    monkeypatch.setenv("DATABENTO_API_KEY", "not-a-real-key-set-by-this-test")
    health = databento_context.health()
    assert health.capability == DatabentoCapability.NOT_CONNECTED
    assert "conftest.py" in health.detail
    # The default must also never fabricate a snapshot/OHLCV/contract.
    assert databento_context.market_snapshot("NQ") is None
    assert databento_context.recent_ohlcv("NQ") is None
    assert databento_context.resolve_display_contract("NQ") is None


@pytest.mark.real_databento_context
def test_conftest_default_does_not_apply_under_the_opt_out_marker(monkeypatch):
    """Sanity check on the opt-out mechanism itself: a test explicitly
    marked `real_databento_context` gets the REAL `databento_context.health`
    (still offline here -- no key, no client -- but genuinely NOT_CONNECTED
    via the real code path, not the conftest.py stand-in)."""
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    health = databento_context.health()
    assert health.capability == DatabentoCapability.NOT_CONNECTED
    assert "conftest.py" not in health.detail


def test_live_smoke_file_gates_on_the_real_environment_variable(monkeypatch):
    """Mission section 1/20: "dedicated live tests must still explicitly
    test real networking [when explicitly enabled]". This does not run any
    live test -- it proves the skip predicate those tests rely on
    (`test_databento_market_data_provider_live._HAS_KEY`) tracks
    `DATABENTO_API_KEY` at import/collection time, which is what makes that
    file's `@pytest.mark.skipif` correctly enable them when a real key is
    present and skip them (the safe default) otherwise.

    The module's own `load_dotenv_if_available()` call is stubbed out before
    reloading it, so this test never re-triggers a real `.env` load (which
    would re-leak this checkout's real key back into `os.environ`, defeating
    the very `monkeypatch.delenv` below and re-creating the class of bug
    this pass fixes) -- this test's own env var, set directly, is the only
    thing `_HAS_KEY` is asked to react to here."""
    import importlib

    from alpha_agent.knowledge import connector_support

    monkeypatch.setattr(connector_support, "load_dotenv_if_available", lambda: None)
    live_mod = importlib.import_module("test_databento_market_data_provider_live")

    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    importlib.reload(live_mod)
    assert live_mod._HAS_KEY is False
    assert live_mod._SKIP_REASON

    monkeypatch.setenv("DATABENTO_API_KEY", "not-a-real-key-set-by-this-test")
    importlib.reload(live_mod)
    assert live_mod._HAS_KEY is True

    # Restore everything (env + the stubbed loader) and reload once more so
    # later tests in this session see the module's real-environment state.
    monkeypatch.undo()
    importlib.reload(live_mod)


def test_dotenv_never_printed_or_logged(monkeypatch, capsys):
    """`bootstrap_local_runtime_environment` must never print/log anything
    -- `load_dotenv_if_available` is documented never to read the file's
    content into a string this process displays; this asserts the bootstrap
    call itself produces no stdout/stderr output regardless."""
    from alpha_agent.knowledge import connector_support

    monkeypatch.setattr(connector_support, "load_dotenv_if_available", lambda: None)
    app.bootstrap_local_runtime_environment()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_env_file_is_gitignored():
    """`.env` must never be a candidate for `git add` -- this is a static,
    offline check of `.gitignore` itself, not a git call."""
    from pathlib import Path

    gitignore = (Path(__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")
    assert any(line.strip() in (".env", "/.env") for line in gitignore.splitlines())


def test_bootstrap_module_never_hardcodes_or_leaks_a_key_literal():
    """Static guard: the bootstrap module's own source never contains a
    literal that looks like a real credential -- it only ever delegates to
    the environment loader, never embeds a value."""
    import ast

    from alpha_agent.ui import app as app_module

    with open(app_module.__file__, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "DATABENTO_API_KEY=" not in node.value
            assert not (len(node.value) > 24 and node.value.isalnum())
    assert "os.environ[" not in source or "DATABENTO_API_KEY" not in source, (
        "app.py must never directly assign into os.environ for a secret key -- "
        "it only ever calls the shared loader"
    )
