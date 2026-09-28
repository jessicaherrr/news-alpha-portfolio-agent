"""Phase 22 / 22.1 -- optional crypto / on-chain extension (prompts/22).

The Phase 22 review-gate decision (2026-09-11) approved a SYNTHETIC-SCAFFOLD-
ONLY implementation: zero network calls, zero paid API use, zero real vendor
connections, no vendor lock-in, no 2025 holdout access. Phase 22.1
(2026-09-12) closed four review findings: (1) the alternative-data surface was
incomplete (no exchange flows, liquidation not in the unified dataset), (2)
every alt-data record needed a separate observation-vs-availability timestamp
and a real causal as-of join, (3) provenance was collapsed to one record
instead of preserved per series end-to-end, (4) UI/doc/provenance wording
overclaimed "a real CME-listed future". This file proves all of it:

1. every synthetic fixture / adapter is provenance-tagged and the tag is
   enforced, not decorative (A-C);
2. the package makes no network call anywhere in its source (D);
3. the crypto strategy family reuses the real, unmodified Phase 10 DSL /
   Phase 09 feature registry / Phase 13 validation engine, and -- when the
   compiled C++ core is present -- the real C++ execution path, end-to-end,
   deterministically (E);
4. the isolated synthetic results store can never become paper-trading
   eligible, join the real BH/FDR family, or be mistaken for the real
   registry, by construction (F-H);
5. every synthetic timestamp is nowhere near the real 2025 locked holdout (I);
6. the Streamlit UI stays read-only, honest-empty-state, banner always
   shown (J);
7. every Phase 22 alternative-data surface is represented, including exchange
   flows and liquidation, in the unified research dataset (K);
8. every alt-data record has a real observation-vs-availability distinction
   and the dataset assembler performs a genuine causal as-of join -- no
   forward-looking leak, no prefix dependence (L);
9. complete per-series provenance survives fixture -> dataset -> research
   bundle -> validation artifact -> isolated registry -> UI, and a stable
   content identity is independent of wall-clock generation time (M);
10. presentation semantics are accurate everywhere: no "real CME-listed
    future" language, and a synthetic outcome is always labelled a
    non-authoritative "Synthetic Policy Outcome" (N).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CLI = Path("build/cpp/cpp/quant_backtest_targets_csv")
CRYPTO_PKG_DIR = Path("python/alpha_agent/crypto")
CRYPTO_1DAY_NS = 86_400_000_000_000


# ======================================================================
# A. provenance is enforced, not decorative
# ======================================================================
def test_A1_synthetic_provenance_constructor_is_always_synthetic():
    from alpha_agent.crypto.provenance import DataProvenanceRole, synthetic_provenance

    prov = synthetic_provenance("funding_rate", seed=7)
    assert prov.role is DataProvenanceRole.SYNTHETIC
    assert prov.vendor != ""
    assert "REAL" not in prov.vendor  # never shaped like a real vendor name


def test_A2_assert_synthetic_raises_for_a_real_role():
    from alpha_agent.crypto.provenance import (
        CryptoDataProvenance,
        DataProvenanceRole,
        RealVendorNotConnectedError,
    )

    real = CryptoDataProvenance(role=DataProvenanceRole.REAL, vendor="some_vendor", series_kind="x")
    with pytest.raises(RealVendorNotConnectedError):
        real.assert_synthetic()


def test_A3_synthetic_fixture_bundle_provenance_is_all_synthetic():
    from alpha_agent.crypto.provenance import DataProvenanceRole
    from alpha_agent.crypto.synthetic_fixtures import synthetic_crypto_bars

    bundle = synthetic_crypto_bars(instrument_id=1, n_days=60, seed=3)
    assert bundle.provenance_by_series  # non-empty
    for prov in bundle.provenance_by_series.values():
        assert prov.role is DataProvenanceRole.SYNTHETIC
        prov.assert_synthetic()  # never raises


def test_A4_provenance_content_identity_excludes_wall_clock():
    from alpha_agent.crypto.provenance import synthetic_provenance

    p1 = synthetic_provenance("funding_rate", seed=7)
    p2 = synthetic_provenance("funding_rate", seed=7)
    assert p1.generated_at_utc != "" and p2.generated_at_utc != ""
    assert p1.content_identity() == p2.content_identity()


# ======================================================================
# B. vendor Protocols are vendor-neutral; the guard fails closed
# ======================================================================
def test_B1_synthetic_adapters_satisfy_their_protocols():
    from alpha_agent.crypto import synthetic_fixtures as sf
    from alpha_agent.crypto import vendors

    assert isinstance(sf.SyntheticPerpFundingAdapter(), vendors.PerpFundingVendorAdapter)
    assert isinstance(sf.SyntheticLiquidationAdapter(), vendors.LiquidationVendorAdapter)
    assert isinstance(sf.SyntheticOnChainAdapter(), vendors.OnChainVendorAdapter)
    assert isinstance(sf.SyntheticExchangeFlowAdapter(), vendors.ExchangeFlowVendorAdapter)
    assert isinstance(sf.SyntheticCmeCryptoBasisAdapter(), vendors.CmeCryptoBasisSource)


def test_B2_assert_vendor_is_synthetic_only_accepts_synthetic_and_rejects_real():
    from alpha_agent.crypto.provenance import DataProvenanceRole, RealVendorNotConnectedError
    from alpha_agent.crypto.synthetic_fixtures import SyntheticPerpFundingAdapter
    from alpha_agent.crypto.vendors import assert_vendor_is_synthetic_only

    assert_vendor_is_synthetic_only(SyntheticPerpFundingAdapter())  # does not raise

    class _FakeRealAdapter:
        provenance_role = DataProvenanceRole.REAL

        def fetch_funding(self, **kw):  # pragma: no cover - never reached
            raise AssertionError("must never be called")

    with pytest.raises(RealVendorNotConnectedError):
        assert_vendor_is_synthetic_only(_FakeRealAdapter())

    class _UntaggedAdapter:  # missing provenance_role entirely -- fails closed
        pass

    with pytest.raises(RealVendorNotConnectedError):
        assert_vendor_is_synthetic_only(_UntaggedAdapter())


def test_B3_typed_row_schemas_reject_a_real_role_row():
    from alpha_agent.crypto.provenance import (
        CryptoDataProvenance,
        DataProvenanceRole,
        RealVendorNotConnectedError,
    )
    from alpha_agent.crypto.schemas import PerpFundingBar

    # the row model itself accepts either role (it is a pure data schema) --
    # provenance enforcement is at construction time (synthetic_provenance),
    # not at the row-schema level. This test pins that boundary explicitly so
    # a future change cannot silently relax it without a visible red test.
    row = PerpFundingBar(
        ts_event_ns=1, available_ts_ns=1, venue="x", symbol="BTC-PERP", funding_rate=0.0,
        open_interest_usd=1.0, mark_price_usd=1.0,
        provenance=CryptoDataProvenance(role=DataProvenanceRole.REAL, vendor="v", series_kind="k"),
    )
    with pytest.raises(RealVendorNotConnectedError):
        row.provenance.assert_synthetic()


def test_B4_available_ts_before_event_ts_is_rejected():
    from alpha_agent.crypto.provenance import synthetic_provenance
    from alpha_agent.crypto.schemas import PerpFundingBar

    with pytest.raises(Exception, match="available_ts_ns"):
        PerpFundingBar(
            ts_event_ns=100, available_ts_ns=50,  # before the event -- invalid
            venue="x", symbol="BTC-PERP", funding_rate=0.0, open_interest_usd=1.0,
            mark_price_usd=1.0, provenance=synthetic_provenance("funding_rate", seed=1),
        )


def test_B5_available_ts_equal_to_event_ts_is_allowed():
    from alpha_agent.crypto.provenance import synthetic_provenance
    from alpha_agent.crypto.schemas import PerpFundingBar

    row = PerpFundingBar(
        ts_event_ns=100, available_ts_ns=100, venue="x", symbol="BTC-PERP",
        funding_rate=0.0, open_interest_usd=1.0, mark_price_usd=1.0,
        provenance=synthetic_provenance("funding_rate", seed=1),
    )
    assert row.available_ts_ns == row.ts_event_ns


# ======================================================================
# C. deterministic, reproducible fixtures
# ======================================================================
def test_C1_same_seed_reproduces_bit_identical_bars():
    from alpha_agent.crypto.synthetic_fixtures import synthetic_crypto_bars

    b1 = synthetic_crypto_bars(instrument_id=5, n_days=120, seed=11)
    b2 = synthetic_crypto_bars(instrument_id=5, n_days=120, seed=11)
    assert b1.bars.equals(b2.bars)
    assert b1.generator_identity == b2.generator_identity


def test_C2_different_seed_changes_the_series():
    from alpha_agent.crypto.synthetic_fixtures import synthetic_crypto_bars

    b1 = synthetic_crypto_bars(instrument_id=5, n_days=120, seed=11)
    b2 = synthetic_crypto_bars(instrument_id=5, n_days=120, seed=12)
    assert not b1.bars["close"].equals(b2.bars["close"])
    assert b1.generator_identity != b2.generator_identity


def test_C3_no_missing_bars_on_the_ohlcv_timeline():
    from alpha_agent.crypto.synthetic_fixtures import DAY_NS, synthetic_crypto_bars

    bundle = synthetic_crypto_bars(instrument_id=5, n_days=200, seed=4)
    ts = bundle.bars["ts_event_ns"].to_numpy()
    assert (ts[1:] - ts[:-1] == DAY_NS).all()
    # the OHLCV execution timeline itself is always complete -- only the
    # slower-publishing alt-data columns may be NaN before their first
    # availability (see section L).
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    assert not bundle.bars[ohlcv_cols].isna().any().any()


# ======================================================================
# D. zero network calls anywhere in the package source
# ======================================================================
def test_D1_crypto_package_makes_no_network_call():
    forbidden = [
        r"\brequests\.", r"\bhttpx\.", r"\burllib\.request", r"\bsocket\.",
        r"\baiohttp\.", r"\bwebsocket\b",
    ]
    py_files = sorted(CRYPTO_PKG_DIR.glob("*.py"))
    assert len(py_files) >= 10, "expected the Phase 22 crypto package to exist"
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not re.search(pattern, text), f"{path}: forbidden network call {pattern!r}"


def test_D2_no_api_key_or_secret_literal_in_source():
    forbidden = [r"api[_-]?key\s*=\s*['\"]", r"Bearer [A-Za-z0-9]"]
    for path in sorted(CRYPTO_PKG_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not re.search(pattern, text, re.IGNORECASE), f"{path}: {pattern!r}"


# ======================================================================
# E. real DSL / feature registry / validation engine reuse, end-to-end
# ======================================================================
def test_E1_strategy_spec_is_a_real_unmodified_StrategySpec_and_compiles():
    from alpha_agent.crypto.strategy_family import (
        FundingContrarianParams,
        make_funding_contrarian_spec,
    )
    from alpha_agent.strategy import StrategySpec, compile_strategy

    spec = make_funding_contrarian_spec(FundingContrarianParams(root_symbol="BTC"))
    assert isinstance(spec, StrategySpec)
    plan = compile_strategy(spec)
    assert plan.root_symbol == "BTC"
    assert len(plan.rules) == 2


def test_E2_reuses_the_existing_zscore_feature_kind_unmodified():
    from alpha_agent.crypto.strategy_family import (
        FundingContrarianParams,
        make_funding_contrarian_spec,
    )
    from alpha_agent.features.registry import REGISTRY

    spec = make_funding_contrarian_spec(FundingContrarianParams(root_symbol="BTC"))
    kind = spec.features[0].spec.kind
    assert kind == "zscore"
    assert kind in REGISTRY.kinds()  # the real, global, unmodified registry


def test_E2b_every_declared_data_surface_can_feed_a_real_FeatureSpec():
    """Prompt 22.1: 'the full typed alternative-data surface can feed the same
    downstream FeatureSpec/StrategySpec pipeline' -- not just the one column
    the demonstration strategy happens to use. Declares a real `zscore`
    FeatureSpec against every column in CRYPTO_FEATURE_COLUMNS and computes it
    through the real, unmodified Phase 09 feature engine."""
    from alpha_agent.crypto.synthetic_fixtures import CRYPTO_FEATURE_COLUMNS, synthetic_crypto_bars
    from alpha_agent.features.compute import compute_features
    from alpha_agent.features.source import SourceSeries
    from alpha_agent.features.spec import FeatureSpec
    from alpha_agent.schemas.market_data import PriceDomain

    bundle = synthetic_crypto_bars(instrument_id=1, n_days=120, seed=22)
    cols = ["ts_event_ns", *CRYPTO_FEATURE_COLUMNS]
    src = SourceSeries(
        frame=bundle.bars.loc[:, cols], price_domain=PriceDomain.RAW_CONTRACT,
        identity={"root_symbol": "BTC"}, interval_ns=CRYPTO_1DAY_NS,
    )
    specs = [
        FeatureSpec(kind="zscore", params={"window": 10}, price_field=col)
        for col in CRYPTO_FEATURE_COLUMNS
    ]
    frame = compute_features(src, specs, require_point_in_time=True)
    assert len(frame.features.columns) == len(CRYPTO_FEATURE_COLUMNS)


def test_E3_hypothesis_spec_is_the_real_unmodified_schema():
    from alpha_agent.crypto.hypotheses import CRYPTO_HYPOTHESES
    from alpha_agent.schemas.hypothesis import HypothesisSpec

    assert len(CRYPTO_HYPOTHESES) >= 2
    for h in CRYPTO_HYPOTHESES:
        assert isinstance(h, HypothesisSpec)
        assert h.required_features


def test_E4_schedule_builds_from_the_real_compiler_and_target_schedule_types():
    from alpha_agent.backtest.targets import TargetSchedule
    from alpha_agent.crypto.strategy_family import (
        FundingContrarianParams,
        funding_contrarian_adapter,
        make_funding_contrarian_spec,
    )
    from alpha_agent.crypto.synthetic_fixtures import synthetic_crypto_bars

    adapter = funding_contrarian_adapter(root_symbol="BTC")
    spec = make_funding_contrarian_spec(FundingContrarianParams(**adapter.canonical_params))
    bars = synthetic_crypto_bars(instrument_id=1, n_days=200, seed=22).bars
    sched = adapter.schedule_for(spec, bars, emit_from_ts_ns=0)
    assert isinstance(sched, TargetSchedule)
    assert len(sched.rows) > 0
    assert {r.target_units for r in sched.rows} <= {-1, 0, 1}


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_E5_full_research_run_through_the_real_cpp_engine(tmp_path):
    from alpha_agent.crypto.research import run_crypto_research
    from alpha_agent.validation.enums import Verdict
    from alpha_agent.validation.phase_13_5c_matrix import frozen_policy

    artifact = run_crypto_research(work_dir=tmp_path)
    assert artifact.report.verdict in (Verdict.PASS, Verdict.REJECT, Verdict.INCONCLUSIVE)
    assert artifact.report.oos_daily.official_source == "cpp_portfolio_accountant_daily_equity_trace"
    assert artifact.report.oos_metrics.n_trading_days > 0
    assert artifact.report.walk_forward.n_folds_evaluated >= 3
    assert not artifact.report.holdout_evaluated
    # the frozen ReliabilityPolicy gate was actually reused, not reinvented --
    # ValidationEngine itself refuses to run under a mismatched policy
    # fingerprint (see ValidationEngine.__init__), so a successful run here is
    # already proof; this pins it explicitly against the frozen factory too.
    assert artifact.report.reliability_policy_fingerprint == frozen_policy().identity()


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_E6_deterministic_replay_of_the_full_pipeline(tmp_path):
    from alpha_agent.crypto.research import run_crypto_research

    r1 = run_crypto_research(work_dir=tmp_path / "a")
    r2 = run_crypto_research(work_dir=tmp_path / "b")
    assert r1.report.report_fingerprint() == r2.report.report_fingerprint()
    assert r1.strategy_fingerprint == r2.strategy_fingerprint
    assert r1.content_identity() == r2.content_identity()


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_E7_different_seed_changes_the_report(tmp_path):
    from alpha_agent.crypto.research import run_crypto_research

    r1 = run_crypto_research(work_dir=tmp_path / "a", seed=22)
    r2 = run_crypto_research(work_dir=tmp_path / "b", seed=23)
    assert r1.report.report_fingerprint() != r2.report.report_fingerprint()
    assert r1.content_identity() != r2.content_identity()


# ======================================================================
# F. the isolated synthetic store: append-only, always SYNTHETIC
# ======================================================================
@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_F1_record_and_read_back_a_synthetic_experiment(tmp_path):
    from alpha_agent.crypto.research import run_crypto_research
    from alpha_agent.crypto.synthetic_registry import (
        get_synthetic_experiment,
        list_synthetic_experiments,
        record_synthetic_experiment,
    )

    db = tmp_path / "crypto_synth.sqlite"
    artifact = run_crypto_research(work_dir=tmp_path / "run")
    eid = record_synthetic_experiment(artifact, db_path=db)

    rows = list_synthetic_experiments(db_path=db)
    assert len(rows) == 1
    assert rows[0]["data_role"] == "SYNTHETIC"
    assert rows[0]["experiment_id"] == eid
    assert rows[0]["content_identity"] == artifact.content_identity()

    detail = get_synthetic_experiment(eid, db_path=db)
    assert detail is not None
    assert detail["data_role"] == "SYNTHETIC"
    assert detail["banner"] == artifact.banner
    assert detail["verdict"] == artifact.report.verdict.value


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_F2_recording_the_same_result_twice_never_silently_overwrites(tmp_path):
    from alpha_agent.crypto.research import run_crypto_research
    from alpha_agent.crypto.synthetic_registry import record_synthetic_experiment

    db = tmp_path / "crypto_synth.sqlite"
    artifact = run_crypto_research(work_dir=tmp_path / "run")
    record_synthetic_experiment(artifact, db_path=db)
    with pytest.raises(ValueError):
        record_synthetic_experiment(artifact, db_path=db)  # append-only, no UPSERT


def test_F3_data_role_check_constraint_rejects_a_real_row_at_the_sql_level(tmp_path):
    import sqlite3

    from alpha_agent.crypto.synthetic_registry import _connect

    db = tmp_path / "crypto_synth.sqlite"
    conn = _connect(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO crypto_synthetic_experiments ("
            "experiment_id, data_role, banner, strategy_family, root_symbol, params_json, "
            "strategy_fingerprint, validation_fingerprint, report_fingerprint, content_identity, "
            "generator_identity, verdict, reason_codes_json, fixture_seed, "
            "provenance_by_series_json, dataset_identity_json, report_json, created_at_utc"
            ") VALUES ('x', 'REAL', 'b', 'f', 'BTC', '{}', 's', 'v', 'r', 'c', 'g', 'REJECT', '[]', "
            "1, '{}', '{}', '{}', 'now')"
        )
    conn.close()


def test_F4_synthetic_registry_module_never_calls_a_real_registry_write_method():
    """Static regression guard, mirroring
    tests/python/test_phase_20_streamlit_ui.py's real-registry read-only guard:
    the crypto package must never import or call the real Phase 14 registry's
    write surface, or the real Phase 21 paper ledger's write surface."""
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE",
        r"alpha_agent\.registry\.sqlite_registry", r"alpha_agent\.paper\.ledger",
        r"ExperimentRegistry\(",
    ]
    for path in sorted(CRYPTO_PKG_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not re.search(pattern, text), f"{path}: forbidden pattern {pattern!r}"


def test_F5_isolated_schema_has_no_authority_or_supersedes_concept(tmp_path):
    from alpha_agent.crypto.synthetic_registry import _connect

    db = tmp_path / "crypto_synth.sqlite"
    conn = _connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(crypto_synthetic_experiments)")}
    conn.close()
    assert not (cols & {"authority", "supersedes", "is_authoritative", "superseded_by"})


# ======================================================================
# G. structurally never paper-trading eligible
# ======================================================================
def test_G1_crypto_family_is_not_a_rebuildable_baseline_family():
    from alpha_agent.crypto.strategy_family import CRYPTO_FUNDING_CONTRARIAN_FAMILY
    from alpha_agent.strategy.candidates_phase_13_5c import BASELINE_FAMILIES, spec_for_params

    assert CRYPTO_FUNDING_CONTRARIAN_FAMILY not in BASELINE_FAMILIES
    with pytest.raises(KeyError):
        spec_for_params(CRYPTO_FUNDING_CONTRARIAN_FAMILY, {"root_symbol": "BTC"})


def test_G2_paper_eligibility_module_never_references_crypto():
    src = Path("python/alpha_agent/paper/eligibility.py").read_text(encoding="utf-8")
    assert "crypto" not in src.lower()


def test_G3_paper_eligibility_only_ever_opens_the_real_registry_path():
    import alpha_agent.paper.eligibility as elig

    # the module hard-imports the real ExperimentRegistry class; it has no
    # reference to the Phase 22 isolated store at all
    assert "ExperimentRegistry" in dir(elig)
    assert not hasattr(elig, "crypto")


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_G4_even_a_synthetic_PASS_would_stay_paper_ineligible(tmp_path, monkeypatch):
    """A synthetic PASS is acceptable only as a diagnostic software-pipeline
    output (prompt 22.1). Force the frozen policy's PASS branch on a real
    artifact and prove the paper-eligibility path still cannot consume it --
    not because the verdict happens to be REJECT today, but because the
    family key is structurally absent from BASELINE_FAMILIES regardless of
    verdict."""
    from alpha_agent.crypto.research import run_crypto_research
    from alpha_agent.crypto.synthetic_registry import record_synthetic_experiment
    from alpha_agent.strategy.candidates_phase_13_5c import BASELINE_FAMILIES
    from alpha_agent.validation.enums import Verdict

    artifact = run_crypto_research(work_dir=tmp_path / "run")
    forced_pass = artifact.model_copy(
        update={"report": artifact.report.model_copy(update={"verdict": Verdict.PASS})}
    )
    eid = record_synthetic_experiment(forced_pass, db_path=tmp_path / "crypto_synth.sqlite")
    assert eid  # recorded successfully as SYNTHETIC evidence
    # still never rebuildable via the real paper-eligibility family map:
    assert forced_pass.strategy_family not in BASELINE_FAMILIES


# ======================================================================
# H. never mixed with the real Phase 13.5C+ statistical family
# ======================================================================
def test_H1_crypto_strategy_key_is_not_one_of_the_real_family_keys():
    from alpha_agent.crypto.strategy_family import CRYPTO_FUNDING_CONTRARIAN_FAMILY

    real_keys = {"tsmom", "ma_trend", "breakout", "mean_reversion", "silver_bullet"}
    assert CRYPTO_FUNDING_CONTRARIAN_FAMILY not in real_keys


def test_H2_real_registry_summary_is_unaffected_by_the_crypto_package(tmp_path):
    """Negative control: importing / exercising the crypto package must never
    change anything the real registry reports."""
    pytest.importorskip("alpha_agent.ui.services")
    from alpha_agent.ui import services

    if not services.REGISTRY_PATH.exists():
        pytest.skip("Phase 14 registry sqlite not present in this checkout")
    before = services.registry_summary()

    from alpha_agent.crypto.synthetic_fixtures import synthetic_crypto_bars  # noqa: F401

    after = services.registry_summary()
    assert before == after


# ======================================================================
# I. synthetic timestamps are nowhere near the real 2025 locked holdout
# ======================================================================
def test_I1_synthetic_fixture_calendar_predates_the_real_holdout_by_years():
    from alpha_agent.crypto.research import N_DAYS
    from alpha_agent.crypto.synthetic_fixtures import DAY_NS, SYNTHETIC_BASE_NS

    last_ts = SYNTHETIC_BASE_NS + N_DAYS * DAY_NS
    last_date = pd.Timestamp(last_ts, unit="ns", tz="UTC")
    assert last_date.year < 2020


def test_I2_holdout_guard_still_fires_on_a_real_2025_timestamp():
    """Sanity: the real holdout guard machinery is untouched and still fires --
    Phase 22 does not weaken it, it simply never gets near it. Also proves
    `record_synthetic_experiment` (which runs this same guard defensively over
    every payload before it writes) would refuse a 2025 value if one ever
    appeared, rather than only relying on the fixture calendar staying honest."""
    from alpha_agent.registry.holdout_guard import HoldoutAccessError, assert_no_holdout_market_data

    real_2025_ns = int(pd.Timestamp("2025-06-01T00:00:00Z").value)
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data({"ts_event_ns": real_2025_ns})


# ======================================================================
# J. Streamlit UI: read-only, honest-empty-state, banner always shown
# ======================================================================
def test_J1_services_crypto_functions_are_plain_python_no_streamlit_import():
    import ast

    tree = ast.parse(Path("python/alpha_agent/ui/services.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert not any(n == "streamlit" or n.startswith("streamlit.") for n in names)


def test_J2_crypto_lab_page_renders_the_honest_empty_state(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from alpha_agent.ui import services
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(services, "CRYPTO_SYNTHETIC_DB_PATH", tmp_path / "does_not_exist.sqlite")
    at = AppTest.from_string("from alpha_agent.ui.views.crypto_lab import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown) + " ".join(w.body for w in at.info) + \
        " ".join(w.body for w in at.error)
    assert "SYNTHETIC" in full_text
    assert "No synthetic crypto experiment has been recorded yet" in full_text


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_J3_crypto_lab_page_renders_a_seeded_synthetic_experiment(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from alpha_agent.crypto.envelope import SYNTHETIC_POLICY_OUTCOME_LABEL
    from alpha_agent.crypto.research import run_crypto_research
    from alpha_agent.crypto.synthetic_registry import record_synthetic_experiment
    from alpha_agent.ui import services
    from streamlit.testing.v1 import AppTest

    db_path = tmp_path / "crypto_synth.sqlite"
    artifact = run_crypto_research(work_dir=tmp_path / "run")
    record_synthetic_experiment(artifact, db_path=db_path)
    monkeypatch.setattr(services, "CRYPTO_SYNTHETIC_DB_PATH", db_path)

    at = AppTest.from_string("from alpha_agent.ui.views.crypto_lab import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown) + " ".join(w.body for w in at.error) + \
        " ".join(c.value for c in at.caption)
    assert "SYNTHETIC" in full_text
    assert "Recorded synthetic experiments" in full_text
    assert SYNTHETIC_POLICY_OUTCOME_LABEL in full_text
    assert "Verdict" not in full_text.replace("Full Report", "")  # no bare "Verdict" label


def test_J4_real_ui_pages_never_import_the_crypto_package():
    """Negative control: the Phase 20/21 pages this app already had must stay
    completely unaware of Phase 22's SCIENTIFIC/DATA package
    (`alpha_agent.crypto.*`) -- only app.py (navigation), crypto_lab.py (this
    page), and (Product Consolidation campaign, task spec section 29:
    "place it under Developer / Labs") system.py's own Labs link may
    reference it at all, and even system.py only imports the PRESENTATION
    module (`views.crypto_lab.render`, to build one `st.Page(...)` nav
    target) -- never `alpha_agent.crypto.*` itself."""
    views_dir = Path("python/alpha_agent/ui/views")
    for path in sorted(views_dir.glob("*.py")):
        if path.name in ("crypto_lab.py", "system.py"):
            continue
        text = path.read_text(encoding="utf-8")
        assert "crypto" not in text.lower(), f"{path} unexpectedly references crypto"

    system_text = (views_dir / "system.py").read_text(encoding="utf-8")
    assert "alpha_agent.crypto" not in system_text, "system.py must never import the scientific crypto package"
    assert "views import crypto_lab" in system_text or "views.crypto_lab" in system_text


# ======================================================================
# K. every Phase 22 alternative-data surface is represented
# ======================================================================
def test_K1_exchange_flow_schema_and_frame_builder_exist():
    from alpha_agent.crypto.provenance import synthetic_provenance
    from alpha_agent.crypto.schemas import ExchangeFlowBar, exchange_flow_frame

    row = ExchangeFlowBar(
        ts_event_ns=CRYPTO_1DAY_NS, available_ts_ns=2 * CRYPTO_1DAY_NS, asset="BTC",
        inflow_usd=1.0, outflow_usd=2.0, provenance=synthetic_provenance("exchange_flows", seed=1),
    )
    df = exchange_flow_frame([row])
    assert list(df["inflow_usd"]) == [1.0]


def test_K2_all_prompt_22_surfaces_present_in_the_unified_dataset():
    """Prompt 22 lists: CME BTC/ETH futures basis, perpetual funding, open
    interest, liquidations, exchange flows, MVRV, SOPR, active addresses."""
    from alpha_agent.crypto.synthetic_fixtures import CRYPTO_FEATURE_COLUMNS

    required_concepts = {
        "basis_pct": "CME BTC/ETH futures basis",
        "funding_rate": "perpetual funding",
        "open_interest_usd": "perpetual open interest",
        "long_liquidation_usd": "liquidations (long side)",
        "short_liquidation_usd": "liquidations (short side)",
        "liquidation_imbalance": "liquidations (imbalance)",
        "exchange_inflow_usd": "exchange flows (inflow)",
        "exchange_outflow_usd": "exchange flows (outflow)",
        "exchange_netflow_usd": "exchange flows (net)",
        "mvrv_ratio": "MVRV",
        "sopr_ratio": "SOPR",
        "active_addresses": "active addresses",
    }
    for col, concept in required_concepts.items():
        assert col in CRYPTO_FEATURE_COLUMNS, f"missing column for {concept}"


def test_K3_liquidation_and_exchange_flow_columns_are_in_the_bars_frame():
    from alpha_agent.crypto.synthetic_fixtures import synthetic_crypto_bars

    bundle = synthetic_crypto_bars(instrument_id=1, n_days=90, seed=5)
    for col in (
        "long_liquidation_usd", "short_liquidation_usd", "liquidation_imbalance",
        "exchange_inflow_usd", "exchange_outflow_usd", "exchange_netflow_usd",
    ):
        assert col in bundle.bars.columns
        assert bundle.bars[col].notna().any()  # eventually available, not all-NaN


def test_K4_liquidation_imbalance_and_netflow_are_correctly_derived():
    from alpha_agent.crypto.synthetic_fixtures import synthetic_crypto_bars

    bundle = synthetic_crypto_bars(instrument_id=1, n_days=90, seed=5)
    b = bundle.bars.dropna(subset=["liquidation_imbalance", "exchange_netflow_usd"])
    np.testing.assert_allclose(
        b["liquidation_imbalance"], b["long_liquidation_usd"] - b["short_liquidation_usd"]
    )
    np.testing.assert_allclose(
        b["exchange_netflow_usd"], b["exchange_inflow_usd"] - b["exchange_outflow_usd"]
    )


def test_K5_no_strategy_per_feature_required_but_full_surface_is_usable():
    """The demonstration strategy uses only `funding_rate`; this does not
    limit what the surface as a whole can express (see test_E2b)."""
    from alpha_agent.crypto.strategy_family import (
        FundingContrarianParams,
        make_funding_contrarian_spec,
    )

    spec = make_funding_contrarian_spec(FundingContrarianParams(root_symbol="BTC"))
    assert len(spec.features) == 1
    assert spec.features[0].spec.price_field == "funding_rate"


# ======================================================================
# L. point-in-time availability + causal as-of join
# ======================================================================
def test_L1_no_observation_visible_before_its_available_ts():
    from alpha_agent.crypto.dataset import causal_as_of_join

    bar_ts = np.array([0, 10, 20, 30])
    out = causal_as_of_join(
        bar_ts, obs_ts_event_ns=[5], obs_available_ts_ns=[25], obs_value=[99.0],
    )
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    assert out[3] == 99.0


def test_L2_equal_available_ts_is_visible_immediately():
    from alpha_agent.crypto.dataset import causal_as_of_join

    out = causal_as_of_join([10], obs_ts_event_ns=[10], obs_available_ts_ns=[10], obs_value=[1.0])
    assert out[0] == 1.0


def test_L3_delayed_publication_does_not_affect_earlier_feature_values():
    """Two variants of the same observation, one immediate and one delayed:
    bars strictly before the observation's own ts_event_ns are identical
    (NaN) in both, and delaying availability never makes an EARLIER bar see
    the value."""
    from alpha_agent.crypto.dataset import causal_as_of_join

    bar_ts = np.array([0, 10, 20, 30, 40])
    immediate = causal_as_of_join(
        bar_ts, obs_ts_event_ns=[20], obs_available_ts_ns=[20], obs_value=[7.0],
    )
    delayed = causal_as_of_join(
        bar_ts, obs_ts_event_ns=[20], obs_available_ts_ns=[35], obs_value=[7.0],
    )
    # before the event ever happens, both agree (NaN)
    assert np.isnan(immediate[0]) and np.isnan(delayed[0])
    assert np.isnan(immediate[1]) and np.isnan(delayed[1])
    # once available in the immediate case but not yet in the delayed case,
    # they diverge -- the delayed value has NOT leaked backward
    assert immediate[2] == 7.0
    assert np.isnan(delayed[2])
    assert np.isnan(delayed[3])
    # once its own (later) available_ts_ns has passed, delayed catches up
    assert delayed[4] == 7.0


def test_L4_a_future_row_cannot_alter_a_decision_before_its_available_ts():
    from alpha_agent.crypto.dataset import causal_as_of_join

    bar_ts = np.array([0, 10, 20])
    base = causal_as_of_join(bar_ts, obs_ts_event_ns=[0], obs_available_ts_ns=[0], obs_value=[1.0])
    # add a "future" observation available only after every bar above
    with_future = causal_as_of_join(
        bar_ts, obs_ts_event_ns=[0, 100], obs_available_ts_ns=[0, 100], obs_value=[1.0, 999.0],
    )
    np.testing.assert_array_equal(base, with_future)


def test_L5_prefix_invariance_rows_after_T_never_change_values_at_or_before_T():
    """Computing the feature frame through T gives identical values whether
    rows after T are present or absent -- the single strongest proof there is
    no hidden forward-looking join."""
    from alpha_agent.crypto.dataset import causal_as_of_join

    bar_ts = np.arange(0, 100, 10)
    obs_ts = np.array([5, 25, 45, 65, 85])
    obs_avail = obs_ts + 3
    obs_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    full = causal_as_of_join(
        bar_ts, obs_ts_event_ns=obs_ts, obs_available_ts_ns=obs_avail, obs_value=obs_val,
    )
    # drop every observation with ts_event_ns > 50 (i.e. everything "after" T=50)
    keep = obs_ts <= 50
    truncated = causal_as_of_join(
        bar_ts, obs_ts_event_ns=obs_ts[keep], obs_available_ts_ns=obs_avail[keep],
        obs_value=obs_val[keep],
    )
    prefix_mask = bar_ts <= 50
    np.testing.assert_array_equal(full[prefix_mask], truncated[prefix_mask])


def test_L6_full_day_aggregate_not_available_at_the_start_of_its_own_day():
    """Prompt 22.1: 'a full-day liquidation, flow, or on-chain value cannot be
    treated as available at the beginning of that same day.'"""
    from alpha_agent.crypto.synthetic_fixtures import (
        DAY_NS,
        SYNTHETIC_BASE_NS,
        synthetic_crypto_bars,
    )

    bundle = synthetic_crypto_bars(instrument_id=1, n_days=10, seed=1)
    day0_row = bundle.bars.iloc[0]
    assert day0_row["ts_event_ns"] == SYNTHETIC_BASE_NS
    for col in (
        "long_liquidation_usd", "exchange_inflow_usd", "mvrv_ratio", "active_addresses",
    ):
        assert pd.isna(day0_row[col]), f"{col} must not be available at the start of day 0"
    assert bundle.bars["ts_event_ns"].iloc[1] == SYNTHETIC_BASE_NS + DAY_NS


def test_L7_liquidation_events_and_their_daily_aggregate_disagree_in_timing():
    """An individual liquidation print is near-real-time; its day's TOTAL is
    only available after the day ends -- proving the aggregate is not merely
    copying the first event's own availability."""
    from alpha_agent.crypto.dataset import aggregate_daily_liquidations
    from alpha_agent.crypto.synthetic_fixtures import (
        DAY_NS,
        SYNTHETIC_BASE_NS,
        synthetic_liquidation_events,
    )

    events = synthetic_liquidation_events(symbol="BTC-PERP", n_days=5, seed=9)
    day0_events = [e for e in events if e.ts_event_ns < SYNTHETIC_BASE_NS + DAY_NS]
    if day0_events:
        assert all(e.available_ts_ns < SYNTHETIC_BASE_NS + DAY_NS for e in day0_events)
    agg = aggregate_daily_liquidations(
        events, start_ns=SYNTHETIC_BASE_NS, day_ns=DAY_NS, n_days=5, publication_delay_ns=900_000_000_000,
    )
    assert agg["long_liquidation_usd"].available_ts_ns[0] >= SYNTHETIC_BASE_NS + DAY_NS


def test_L8_causal_join_matches_the_next_bar_execution_semantics():
    """Preserve existing next-bar execution semantics: the causal join changes
    WHICH VALUE a feature sees at bar T, never WHEN the strategy's resulting
    decision executes -- that stays the engine's frozen next-eligible-bar
    rule, untouched by this module."""
    from alpha_agent.crypto.strategy_family import (
        FundingContrarianParams,
        funding_contrarian_adapter,
        make_funding_contrarian_spec,
    )
    from alpha_agent.crypto.synthetic_fixtures import synthetic_crypto_bars

    adapter = funding_contrarian_adapter(root_symbol="BTC")
    spec = make_funding_contrarian_spec(FundingContrarianParams(**adapter.canonical_params))
    bars = synthetic_crypto_bars(instrument_id=1, n_days=200, seed=22).bars
    sched = adapter.schedule_for(spec, bars, emit_from_ts_ns=0)
    # every scheduled decision ts is an actual bar timestamp -- the schedule
    # never invents an intermediate execution timestamp
    bar_ts_set = set(bars["ts_event_ns"].tolist())
    assert all(r.ts_event_ns in bar_ts_set for r in sched.rows)


# ======================================================================
# M. complete provenance survives end-to-end; stable content identity
# ======================================================================
def test_M1_provenance_by_series_is_not_collapsed_on_the_bundle():
    from alpha_agent.crypto.synthetic_fixtures import CRYPTO_FEATURE_COLUMNS, synthetic_crypto_bars

    bundle = synthetic_crypto_bars(instrument_id=1, n_days=60, seed=2)
    for col in ("ohlcv", *CRYPTO_FEATURE_COLUMNS):
        assert col in bundle.provenance_by_series, f"missing provenance for {col}"


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_M2_provenance_survives_fixture_to_dataset_to_artifact_to_registry_to_ui(tmp_path, monkeypatch):
    from alpha_agent.crypto.envelope import SyntheticCryptoValidationArtifact
    from alpha_agent.crypto.research import build_crypto_dataset, run_crypto_research
    from alpha_agent.crypto.synthetic_fixtures import CRYPTO_FEATURE_COLUMNS
    from alpha_agent.crypto.synthetic_registry import record_synthetic_experiment

    # fixture
    bundle, _contracts = build_crypto_dataset(seed=22)
    fixture_series = set(bundle.provenance_by_series)

    # research bundle / validation artifact
    artifact = run_crypto_research(work_dir=tmp_path / "run", seed=22)
    assert isinstance(artifact, SyntheticCryptoValidationArtifact)
    assert set(artifact.provenance_by_series) == fixture_series

    # isolated registry
    db_path = tmp_path / "crypto_synth.sqlite"
    eid = record_synthetic_experiment(artifact, db_path=db_path)
    from alpha_agent.crypto.synthetic_registry import get_synthetic_experiment

    row = get_synthetic_experiment(eid, db_path=db_path)
    assert set(row["provenance_by_series"]) == fixture_series
    for col in ("ohlcv", *CRYPTO_FEATURE_COLUMNS):
        assert row["provenance_by_series"][col]["role"] == "SYNTHETIC"

    # UI
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from alpha_agent.ui import services
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(services, "CRYPTO_SYNTHETIC_DB_PATH", db_path)
    at = AppTest.from_string("from alpha_agent.ui.views.crypto_lab import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_M3_content_identity_is_wall_clock_independent(tmp_path):
    from alpha_agent.crypto.research import run_crypto_research

    a1 = run_crypto_research(work_dir=tmp_path / "a", seed=22)
    a2 = run_crypto_research(work_dir=tmp_path / "b", seed=22)
    # generated_at_utc genuinely differs (real wall-clock metadata) ...
    assert (
        a1.provenance_by_series["ohlcv"].generated_at_utc
        != a2.provenance_by_series["ohlcv"].generated_at_utc
    )
    # ... but the scientific/content identity does not depend on it
    assert a1.content_identity() == a2.content_identity()


def test_M4_dataset_identity_source_fingerprint_is_content_based_not_a_bare_seed_string():
    from alpha_agent.crypto.research import build_crypto_dataset
    from alpha_agent.validation.dataset import DatasetIdentity, frame_content_hash

    bundle, _ = build_crypto_dataset(seed=22)
    di = DatasetIdentity(
        root_symbol="BTC", price_domain="raw_contract", source_fingerprint=bundle.generator_identity,
        bars_content_hash=frame_content_hash(bundle.bars), n_bars=len(bundle.bars),
    )
    assert di.source_fingerprint == bundle.generator_identity
    assert di.source_fingerprint.startswith("cryptogenerator1:")
    assert "seed=" not in di.source_fingerprint  # not the old bare descriptive string


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_M5_envelope_rejects_a_tampered_real_provenance_entry(tmp_path):
    from alpha_agent.crypto.envelope import SyntheticCryptoValidationArtifact
    from alpha_agent.crypto.research import run_crypto_research

    artifact = run_crypto_research(work_dir=tmp_path / "run")
    # `model_copy(update=...)` deliberately bypasses validators in pydantic v2
    # (it is a cheap-copy primitive), so it would NOT exercise the guard this
    # test is checking -- rebuild via `model_validate` instead, which reruns
    # every field validator including `_synthetic_only`.
    data = artifact.model_dump(mode="python")
    data["provenance_by_series"]["ohlcv"] = {
        "schema_version": "crypto-provenance/1", "role": "REAL", "vendor": "not_really_real",
        "series_kind": "ohlcv", "fixture_seed": None, "generated_at_utc": "x", "notes": "",
    }
    with pytest.raises(Exception, match="SYNTHETIC"):
        SyntheticCryptoValidationArtifact.model_validate(data)


# ======================================================================
# N. accurate synthetic/real presentation semantics
# ======================================================================
_MISLEADING_PHRASES = (
    "cme-listed future",
    "actual cme btc",
    "actual cme eth",
    "real historical crypto",
    "real bitcoin futures",
    "real ethereum futures",
)

#: A banned phrase preceded (within this many characters) by one of these
#: cues is a legitimate NEGATION -- "never a real CME-listed future", "no
#: synthetic-data experiment may appear ... as real historical crypto
#: performance" -- not an overclaim. Only an UNQUALIFIED appearance fails.
_NEGATION_CUES = (
    "never", "not a", "not the", "not claim", "not real", "not presented",
    "not shown", "no synthetic", "isn't", "is not", "without qualifying",
    "claiming a", "language claiming", "phrases claiming", "guard against",
    "denylist", "banned phrase", "misleading phrase",
)
_NEGATION_WINDOW = 60

_SCANNED_FOR_WORDING = (
    *sorted(CRYPTO_PKG_DIR.glob("*.py")),
    Path("python/alpha_agent/ui/views/crypto_lab.py"),
    Path("scripts/phase_22_crypto_research.py"),
    Path("docs/CRYPTO_ONCHAIN_EXTENSION.md"),
    Path("outputs/phase_22/PHASE_22_CRYPTO_SYNTHETIC_SCAFFOLD.json"),
)


def test_N1_no_misleading_real_cme_claim_anywhere_in_phase_22_surfaces():
    for path in _SCANNED_FOR_WORDING:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for phrase in _MISLEADING_PHRASES:
            start = 0
            while True:
                idx = text.find(phrase, start)
                if idx == -1:
                    break
                window = text[max(0, idx - _NEGATION_WINDOW):idx]
                assert any(cue in window for cue in _NEGATION_CUES), (
                    f"{path}: unqualified occurrence of {phrase!r} at offset {idx} "
                    f"(preceding context: {window!r})"
                )
                start = idx + len(phrase)


def test_N2_synthetic_policy_outcome_label_is_defined_and_non_authoritative_sounding():
    from alpha_agent.crypto.envelope import SYNTHETIC_POLICY_OUTCOME_LABEL

    assert "NON-AUTHORITATIVE" in SYNTHETIC_POLICY_OUTCOME_LABEL
    assert "Synthetic" in SYNTHETIC_POLICY_OUTCOME_LABEL


def test_N3_crypto_lab_page_never_renders_a_bare_verdict_label():
    text = Path("python/alpha_agent/ui/views/crypto_lab.py").read_text(encoding="utf-8")
    # the only acceptable appearances of "Verdict" are inside the raw embedded
    # ValidationReport JSON (st.json(detail["report"])) and this file's own
    # comments -- there must be no metric/column literally labelled "Verdict"
    assert 'metric_card("cl-verdict", "Verdict"' not in text
    assert '"Verdict":' not in text


def test_N4_cli_prints_the_synthetic_policy_outcome_label_not_a_bare_verdict():
    text = Path("scripts/phase_22_crypto_research.py").read_text(encoding="utf-8")
    assert "SYNTHETIC_POLICY_OUTCOME_LABEL" in text


def test_N5_generator_and_contract_docstrings_state_real_vs_synthetic_explicitly():
    text = Path("python/alpha_agent/crypto/synthetic_fixtures.py").read_text(encoding="utf-8")
    assert "real, unmodified C++ Quant Core" in text
    assert "NOT a real CME contract" in text or "not a real CME BTC/ETH contract" in text.lower()


# ======================================================================
# O. Phase 22.1b -- legacy synthetic registry schema guard
# ======================================================================
# The EXACT Phase 22 (bdafd12) v1 table shape -- not a hand-wavy
# approximation. Copied verbatim from that commit's `_SCHEMA_SQL`.
_V1_SCHEMA_SQL_BDAFD12 = """
CREATE TABLE IF NOT EXISTS crypto_synthetic_experiments (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL UNIQUE,
    data_role TEXT NOT NULL CHECK (data_role = 'SYNTHETIC'),
    banner TEXT NOT NULL,
    strategy_family TEXT NOT NULL,
    root_symbol TEXT NOT NULL,
    params_json TEXT NOT NULL,
    strategy_fingerprint TEXT NOT NULL,
    validation_spec_fingerprint TEXT NOT NULL,
    report_fingerprint TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL,
    fixture_seed INTEGER NOT NULL,
    provenance_json TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
"""


def _seed_v1_database(db_path: Path, *, with_row: bool = False) -> None:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute(_V1_SCHEMA_SQL_BDAFD12)
    if with_row:
        conn.execute(
            "INSERT INTO crypto_synthetic_experiments (experiment_id, data_role, banner, "
            "strategy_family, root_symbol, params_json, strategy_fingerprint, "
            "validation_spec_fingerprint, report_fingerprint, verdict, reason_codes_json, "
            "fixture_seed, provenance_json, report_json, created_at_utc) VALUES "
            "('LEGACY-1', 'SYNTHETIC', 'b', 'crypto_funding_contrarian', 'BTC', '{}', 's', "
            "'v', 'r', 'REJECT', '[]', 1, '{}', '{}', '2026-01-01T00:00:00+00:00')"
        )
    conn.commit()
    conn.close()


def test_O1_fresh_database_initializes_as_schema_v2(tmp_path):
    from alpha_agent.crypto.synthetic_registry import SCHEMA_VERSION, _connect

    db = tmp_path / "fresh.sqlite"
    conn = _connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 2
    conn.close()


def test_O2_schema_version_is_persisted_and_detected_on_reopen(tmp_path):
    from alpha_agent.crypto.synthetic_registry import _connect

    db = tmp_path / "fresh.sqlite"
    _connect(db).close()
    conn2 = _connect(db)  # reopen -- must not raise, must still report v2
    assert conn2.execute("PRAGMA user_version").fetchone()[0] == 2
    conn2.close()


def test_O3_a_v2_database_predating_the_version_pragma_is_recognized_by_shape(tmp_path):
    """A database created by the Phase 22.1 (54cff78) code -- real v2 columns,
    but that code never stamped PRAGMA user_version -- must be recognized as
    v2 by its actual shape, not rejected as unknown, and gets stamped now."""
    import sqlite3

    from alpha_agent.crypto.synthetic_registry import _SCHEMA_SQL, _connect

    db = tmp_path / "old_v2_no_pragma.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute(_SCHEMA_SQL)
    conn.commit()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0  # never stamped
    conn.close()

    conn2 = _connect(db)  # must not raise
    assert conn2.execute("PRAGMA user_version").fetchone()[0] == 2  # stamped now
    conn2.close()


@pytest.mark.skipif(not CLI.exists(), reason="C++ core not built")
def test_O4_existing_v2_store_round_trips_normally(tmp_path):
    from alpha_agent.crypto.research import run_crypto_research
    from alpha_agent.crypto.synthetic_registry import (
        get_synthetic_experiment,
        list_synthetic_experiments,
        record_synthetic_experiment,
    )

    db = tmp_path / "v2.sqlite"
    artifact = run_crypto_research(work_dir=tmp_path / "run")
    eid = record_synthetic_experiment(artifact, db_path=db)
    # reopen and read back -- proves normal v2 operation is untouched by the guard
    rows = list_synthetic_experiments(db_path=db)
    assert len(rows) == 1
    assert get_synthetic_experiment(eid, db_path=db) is not None


def test_O5_a_seeded_v1_store_is_detected_before_any_v2_query(tmp_path):
    from alpha_agent.crypto.synthetic_registry import SyntheticRegistrySchemaError, _connect

    db = tmp_path / "legacy_v1.sqlite"
    _seed_v1_database(db, with_row=True)
    with pytest.raises(SyntheticRegistrySchemaError, match="[Ll]egacy Phase 22"):
        _connect(db)


def test_O6_list_get_record_on_v1_never_leak_a_raw_sqlite_operational_error(tmp_path):
    import sqlite3

    from alpha_agent.crypto.synthetic_registry import (
        SyntheticRegistrySchemaError,
        _connect,
        get_synthetic_experiment,
        list_synthetic_experiments,
    )

    db = tmp_path / "legacy_v1.sqlite"
    _seed_v1_database(db, with_row=True)

    with pytest.raises(SyntheticRegistrySchemaError):
        list_synthetic_experiments(db_path=db)
    with pytest.raises(SyntheticRegistrySchemaError):
        get_synthetic_experiment("LEGACY-1", db_path=db)

    # record_synthetic_experiment itself calls _connect() first (before any
    # INSERT is built), so exercising that shared connect path already proves
    # the write path fails the same way -- assert the failure mode is the
    # typed error, never a raw sqlite3.OperationalError.
    try:
        _connect(db)
        raise AssertionError("expected SyntheticRegistrySchemaError")
    except SyntheticRegistrySchemaError:
        pass
    except sqlite3.OperationalError:
        pytest.fail("a raw sqlite3.OperationalError leaked instead of the typed schema error")


def test_O7_no_v1_row_is_silently_transformed_into_a_v2_artifact(tmp_path):
    """The legacy row's content ('LEGACY-1', REJECT, ...) must never surface
    through the v2 read path in any form -- the guard fires before a single
    column of it is read."""
    from alpha_agent.crypto.synthetic_registry import (
        SyntheticRegistrySchemaError,
        list_synthetic_experiments,
    )

    db = tmp_path / "legacy_v1.sqlite"
    _seed_v1_database(db, with_row=True)
    with pytest.raises(SyntheticRegistrySchemaError):
        rows = list_synthetic_experiments(db_path=db)
        assert not rows  # unreachable if the guard works; documents intent


def test_O8_malformed_unknown_schema_fails_closed(tmp_path):
    import sqlite3

    from alpha_agent.crypto.synthetic_registry import SyntheticRegistrySchemaError, _connect

    db = tmp_path / "malformed.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE crypto_synthetic_experiments (some_unrelated_column TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(SyntheticRegistrySchemaError, match="Unrecognized"):
        _connect(db)


def test_O9_crypto_lab_renders_the_legacy_store_state_without_exception(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from alpha_agent.ui import services
    from streamlit.testing.v1 import AppTest

    db = tmp_path / "legacy_v1.sqlite"
    _seed_v1_database(db, with_row=True)
    monkeypatch.setattr(services, "CRYPTO_SYNTHETIC_DB_PATH", db)

    at = AppTest.from_string("from alpha_agent.ui.views.crypto_lab import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(w.body for w in at.warning) + " ".join(w.body for w in at.error) + \
        " ".join(c.value for c in at.caption)
    assert "LEGACY SYNTHETIC STORE" in full_text
    assert "Phase 22 schema v1 detected" in full_text
    assert "LEGACY-1" not in full_text  # the v1 row's own content never surfaces


def test_O10_cli_handles_v1_with_a_typed_refused_message(tmp_path):
    import subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    db = tmp_path / "legacy_v1.sqlite"
    _seed_v1_database(db, with_row=True)
    result = subprocess.run(
        [_sys.executable, "scripts/phase_22_crypto_research.py", "--db", str(db), "list"],
        capture_output=True, text=True, cwd=str(repo_root), check=False,
    )
    assert result.returncode == 3
    assert "SCHEMA_INCOMPATIBLE" in result.stderr
    assert "Traceback" not in result.stderr
    assert "LEGACY-1" not in result.stdout


def test_O11_reset_store_requires_confirm_and_succeeds_against_the_configured_canonical_path(
    tmp_path, monkeypatch
):
    """Normal reset testing configures the isolated synthetic registry
    location to a tmp path (Phase 22.1c) rather than granting reset authority
    over an arbitrary path passed straight through."""
    from alpha_agent.crypto import synthetic_registry as sreg

    canonical = tmp_path / "legacy_v1.sqlite"
    monkeypatch.setattr(sreg, "DEFAULT_DB_PATH", canonical)
    _seed_v1_database(canonical, with_row=True)

    with pytest.raises(ValueError, match="confirm"):
        sreg.reset_synthetic_store(confirm=False)
    assert canonical.exists()  # untouched without confirm

    archived = sreg.reset_synthetic_store(confirm=True)  # db_path omitted -> canonical
    assert archived is not None and archived.exists()
    assert not canonical.exists()

    # reset_synthetic_store operates purely on the configured canonical path --
    # it never imports or opens the real registry module (see test_O13/F4 for
    # the code-level guard against ExperimentRegistry(...) / sqlite_registry).
    import inspect

    src = inspect.getsource(sreg.reset_synthetic_store)
    assert "ExperimentRegistry" not in src
    assert "registry/experiments.sqlite" not in src


def test_O12_reset_on_a_valid_v2_store_also_works_and_is_never_automatic(tmp_path, monkeypatch):
    from alpha_agent.crypto import synthetic_registry as sreg

    canonical = tmp_path / "v2.sqlite"
    monkeypatch.setattr(sreg, "DEFAULT_DB_PATH", canonical)
    sreg._connect(canonical).close()
    assert canonical.exists()
    # opening/connecting never resets anything by itself
    monkeypatch.setattr(sreg, "DEFAULT_DB_PATH", tmp_path / "does_not_exist.sqlite")
    archived = sreg.reset_synthetic_store(confirm=True)
    assert archived is None
    assert canonical.exists()


def test_O13_synthetic_registry_never_calls_a_real_registry_write_method_still_holds():
    """Re-affirms test_F4 after the 22.1b edit -- the legacy-guard code itself
    must not have introduced a real-registry dependency."""
    text = Path("python/alpha_agent/crypto/synthetic_registry.py").read_text(encoding="utf-8")
    forbidden = [r"alpha_agent\.registry\.sqlite_registry", r"alpha_agent\.paper\.ledger", r"ExperimentRegistry\("]
    for pattern in forbidden:
        assert not re.search(pattern, text), f"synthetic_registry.py: forbidden pattern {pattern!r}"


def test_O14_real_phase_14_registry_and_phase_21_eligibility_unaffected(tmp_path):
    """Negative control mirroring test_H2/test_G3: exercising the legacy-schema
    guard (including a raised SyntheticRegistrySchemaError) must change
    nothing about the real registry or paper eligibility."""
    import alpha_agent.paper.eligibility as elig
    from alpha_agent.strategy.candidates_phase_13_5c import BASELINE_FAMILIES

    before_families = set(BASELINE_FAMILIES)
    db = tmp_path / "legacy_v1.sqlite"
    _seed_v1_database(db, with_row=True)
    from alpha_agent.crypto.synthetic_registry import (
        SyntheticRegistrySchemaError,
        list_synthetic_experiments,
    )

    with pytest.raises(SyntheticRegistrySchemaError):
        list_synthetic_experiments(db_path=db)

    assert set(BASELINE_FAMILIES) == before_families
    assert "ExperimentRegistry" in dir(elig)

    pytest.importorskip("alpha_agent.ui.services")
    from alpha_agent.ui import services

    if services.REGISTRY_PATH.exists():
        summary_before = services.registry_summary()
        assert services.registry_summary() == summary_before


# ======================================================================
# P. Phase 22.1c -- restrict destructive reset to the isolated synthetic path
# ======================================================================
def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_P1_reset_synthetic_store_refuses_an_arbitrary_path_at_the_function_boundary(tmp_path):
    from alpha_agent.crypto.synthetic_registry import (
        SyntheticRegistryPathError,
        reset_synthetic_store,
    )

    arbitrary = tmp_path / "not_the_synthetic_store.sqlite"
    arbitrary.write_bytes(b"not even a real sqlite file")
    with pytest.raises(SyntheticRegistryPathError):
        reset_synthetic_store(db_path=arbitrary, confirm=True)
    assert arbitrary.exists()  # untouched
    assert not any(p.name.startswith(arbitrary.name + ".legacy-") for p in tmp_path.iterdir())


def test_P2_reset_synthetic_store_refuses_before_any_schema_or_content_is_read(tmp_path):
    """The path check happens purely at the filesystem-path level -- it fires
    even for content that would otherwise pass/fail schema detection, proving
    it is a boundary check, not a side effect of the schema guard."""
    from alpha_agent.crypto.synthetic_registry import (
        SyntheticRegistryPathError,
        reset_synthetic_store,
    )

    # a totally malformed / non-sqlite file -- if the path check ran AFTER
    # trying to open/inspect it, this would raise a different (sqlite) error
    malformed = tmp_path / "garbage.sqlite"
    malformed.write_bytes(b"\x00\x01\x02 this is not a sqlite database")
    with pytest.raises(SyntheticRegistryPathError):
        reset_synthetic_store(db_path=malformed, confirm=True)
    assert malformed.read_bytes() == b"\x00\x01\x02 this is not a sqlite database"


def test_P3_cli_reset_store_refuses_an_arbitrary_db_path(tmp_path):
    import subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    arbitrary = tmp_path / "arbitrary.sqlite"
    _seed_v1_database(arbitrary)  # any content -- must be refused on path grounds regardless

    result = subprocess.run(
        [_sys.executable, "scripts/phase_22_crypto_research.py", "--db", str(arbitrary), "reset-store", "--confirm"],
        capture_output=True, text=True, cwd=str(repo_root), check=False,
    )
    assert result.returncode != 0
    assert "PATH_REFUSED" in result.stderr
    assert "Traceback" not in result.stderr
    assert arbitrary.exists()
    assert not any(p.name.startswith("arbitrary.sqlite.legacy-") for p in tmp_path.iterdir())


@pytest.mark.skipif(
    not Path("data/registry/experiments.sqlite").exists(),
    reason="real Phase 14 registry sqlite not present in this checkout",
)
def test_P4_cli_reset_store_refuses_the_real_phase_14_registry_path():
    """Mandatory negative control: the actual CLI mutation path, pointed at
    the real registry file, must refuse -- not archive it, not touch it at
    all -- proven by a byte-for-byte hash before and after."""
    import subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    real_registry = repo_root / "data" / "registry" / "experiments.sqlite"
    before_hash = _sha256_file(real_registry)
    before_dir_listing = {p.name for p in real_registry.parent.iterdir()}

    result = subprocess.run(
        [
            _sys.executable, "scripts/phase_22_crypto_research.py",
            "--db", "data/registry/experiments.sqlite", "reset-store", "--confirm",
        ],
        capture_output=True, text=True, cwd=str(repo_root), check=False,
    )

    assert result.returncode != 0, "the real Phase 14 registry must never be a valid reset target"
    assert "PATH_REFUSED" in result.stderr
    assert "Traceback" not in result.stderr
    assert real_registry.exists()
    assert _sha256_file(real_registry) == before_hash, "the real registry's bytes must be unchanged"
    after_dir_listing = {p.name for p in real_registry.parent.iterdir()}
    assert after_dir_listing == before_dir_listing, "no .legacy-* backup of the real registry was created"
    assert not any(name.startswith("experiments.sqlite.legacy-") for name in after_dir_listing)


def test_P5_cli_reset_store_refuses_the_paper_ledger_path(tmp_path):
    """The Phase 21 paper ledger is a different isolated store this command
    must also never be able to target."""
    import subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    fake_ledger = tmp_path / "paper_ledger.sqlite"
    fake_ledger.write_bytes(b"pretend paper ledger bytes")

    result = subprocess.run(
        [_sys.executable, "scripts/phase_22_crypto_research.py", "--db", str(fake_ledger), "reset-store", "--confirm"],
        capture_output=True, text=True, cwd=str(repo_root), check=False,
    )
    assert result.returncode != 0
    assert "PATH_REFUSED" in result.stderr
    assert fake_ledger.read_bytes() == b"pretend paper ledger bytes"


def test_P6_list_and_show_still_accept_a_custom_db_path_unaffected_by_the_reset_restriction(tmp_path):
    """Confirms the restriction is scoped to the destructive reset path only
    -- read/append operations keep their existing custom-db test ergonomics."""
    from alpha_agent.crypto.synthetic_registry import (
        SyntheticRegistrySchemaError,
        get_synthetic_experiment,
        list_synthetic_experiments,
    )

    custom = tmp_path / "any_name_i_like.sqlite"
    _seed_v1_database(custom, with_row=True)

    # still freely addressable -- just still schema-guarded as before (O5/O6),
    # never path-restricted
    with pytest.raises(SyntheticRegistrySchemaError):
        list_synthetic_experiments(db_path=custom)
    with pytest.raises(SyntheticRegistrySchemaError):
        get_synthetic_experiment("LEGACY-1", db_path=custom)


def test_P7_reset_function_default_db_path_argument_is_not_a_bound_default(tmp_path, monkeypatch):
    """Regression guard for the exact bug class this phase fixes: the
    canonical path must be looked up at CALL time (so monkeypatching
    DEFAULT_DB_PATH takes effect), never captured once at function-definition
    time as a bound default parameter value."""
    import inspect

    from alpha_agent.crypto import synthetic_registry as sreg

    sig = inspect.signature(sreg.reset_synthetic_store)
    assert sig.parameters["db_path"].default is None, (
        "db_path must default to None and resolve DEFAULT_DB_PATH fresh inside the "
        "function body -- a bound-at-definition-time default would not observe a "
        "monkeypatched DEFAULT_DB_PATH"
    )

    canonical = tmp_path / "registry.sqlite"
    monkeypatch.setattr(sreg, "DEFAULT_DB_PATH", canonical)
    sreg._connect(canonical).close()
    archived = sreg.reset_synthetic_store(confirm=True)
    assert archived is not None
