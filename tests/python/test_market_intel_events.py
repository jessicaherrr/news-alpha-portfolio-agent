"""Market Intelligence Data Completion Pass, Checkpoint F -- scheduled
event calendars (FOMC, EIA weekly reports), importance rules, next-event
computation, and the EventStore. Every connector test injects fake HTML via
monkeypatched `http_get_text` -- NO real network call.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from alpha_agent.market_intel import importance
from alpha_agent.market_intel.connectors import eia as eia_mod
from alpha_agent.market_intel.connectors import fed as fed_mod
from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
from alpha_agent.market_intel.store import EventStore
from alpha_agent.marketdata.capability import CapabilityState

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# ScheduledMarketEvent -- Section 23/26
# ---------------------------------------------------------------------------


def _event(**overrides) -> ScheduledMarketEvent:
    base = {
        "event_id": "test:1", "name": "Test Event", "source_name": "Test Source", "source_url": "https://example.gov",
        "scheduled_at": NOW + timedelta(days=1), "timezone": "America/New_York", "category": "OTHER",
        "importance": EventImportance.MEDIUM, "importance_rule": "unmapped-category-default-medium/1",
        "mapping_reason": "test", "retrieved_at": NOW,
    }
    base.update(overrides)
    return ScheduledMarketEvent(**base)


def test_scheduled_at_must_be_timezone_aware():
    import pytest

    with pytest.raises(ValueError, match="aware"):
        ScheduledMarketEvent(
            event_id="x", name="n", source_name="s", source_url="https://x.gov",
            scheduled_at=datetime(2026, 9, 16, 14, 30),  # noqa: DTZ001 -- deliberately naive, testing rejection
            timezone="America/New_York",
            category="OTHER", importance=EventImportance.LOW, importance_rule="r", mapping_reason="m",
            retrieved_at=NOW,
        )


def test_actual_release_at_defaults_to_none_never_fabricated():
    assert _event().actual_release_at is None


# ---------------------------------------------------------------------------
# importance -- Section 25
# ---------------------------------------------------------------------------


def test_fomc_and_macro_releases_are_always_high():
    for category in ("FOMC_POLICY", "CPI_PPI_EMPLOYMENT", "PETROLEUM", "NATURAL_GAS", "USDA_GRAIN_OILSEED"):
        imp, rule = importance.importance_for_category(category)
        assert imp is EventImportance.HIGH
        assert rule


def test_unmapped_category_defaults_to_medium_not_low_or_high():
    imp, rule = importance.importance_for_category("SOME_UNKNOWN_CATEGORY")
    assert imp is EventImportance.MEDIUM
    assert rule == "unmapped-category-default-medium/1"


# ---------------------------------------------------------------------------
# FOMC calendar -- real page structure fixture (verified live 2026-09-14)
# ---------------------------------------------------------------------------

_FOMC_HTML = """
<div class="panel panel-default"><div class="panel-heading"><h4><a id="1">2026 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting">
<div class="fomc-meeting__month col-xs-5"><strong>September</strong></div>
<div class="fomc-meeting__date col-xs-4">15-16</div>
</div>
<div class="row fomc-meeting">
<div class="fomc-meeting__month col-xs-5"><strong>October</strong></div>
<div class="fomc-meeting__date col-xs-4">27-28</div>
</div>
</div>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="2">2027 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting">
<div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
<div class="fomc-meeting__date col-xs-4">26-27</div>
</div>
</div>
"""


def test_fomc_calendar_parses_real_page_structure(monkeypatch):
    monkeypatch.setattr(fed_mod, "http_get_text", lambda url, **kw: _FOMC_HTML)
    connector = fed_mod.FedEventCalendarConnector()
    events = connector.fetch_upcoming(now=NOW, horizon_days=200)
    assert [e.event_id for e in events] == ["fomc:2026-09-16", "fomc:2026-10-28", "fomc:2027-01-27"]
    assert all(e.importance is EventImportance.HIGH for e in events)
    assert all("ES" in e.affected_products for e in events)


def test_fomc_calendar_excludes_past_meetings():
    monkeypatch_html = _FOMC_HTML
    import alpha_agent.market_intel.connectors.fed as f

    orig = f.http_get_text
    f.http_get_text = lambda url, **kw: monkeypatch_html
    try:
        connector = f.FedEventCalendarConnector()
        # "now" is AFTER the September meeting -- it must not appear.
        later = datetime(2026, 9, 20, tzinfo=UTC)
        events = connector.fetch_upcoming(now=later, horizon_days=200)
        assert "fomc:2026-09-16" not in [e.event_id for e in events]
        assert "fomc:2026-10-28" in [e.event_id for e in events]
    finally:
        f.http_get_text = orig


def test_fomc_calendar_respects_horizon_days(monkeypatch):
    monkeypatch.setattr(fed_mod, "http_get_text", lambda url, **kw: _FOMC_HTML)
    connector = fed_mod.FedEventCalendarConnector()
    events = connector.fetch_upcoming(now=NOW, horizon_days=10)  # only September meeting is this close
    assert [e.event_id for e in events] == ["fomc:2026-09-16"]


def test_fomc_meeting_end_day_handles_asterisk_and_single_day():
    assert fed_mod._meeting_end_day("27-28") == 28
    assert fed_mod._meeting_end_day("17-18*") == 18
    assert fed_mod._meeting_end_day("14") == 14


def test_fomc_calendar_health_reflects_real_connectivity(monkeypatch):
    def _raise(*a, **kw):
        from alpha_agent.market_intel.http_support import MarketIntelHttpError

        raise MarketIntelHttpError("down")

    monkeypatch.setattr(fed_mod, "http_get_text", _raise)
    assert fed_mod.FedEventCalendarConnector().health() is CapabilityState.NOT_CONNECTED


# ---------------------------------------------------------------------------
# EIA weekly report calendars -- real page structure fixture
# ---------------------------------------------------------------------------

#: Real exception row, verified live from
#: https://www.eia.gov/petroleum/supply/weekly/schedule.php (2026-09-14):
#: the week ending September 4, 2026 (Labor Day) releases Thursday
#: September 10 at 12:00 p.m. ET instead of the normal Wednesday.
_EIA_PETROLEUM_SCHEDULE_HTML = """
<p>Release Schedule</p>
<p>The standard release time and day of the week will be at 10:30 a.m. eastern time on Wednesdays.</p>
<table>
<tr><td>December 27, 2024</td><td>January 2, 2025</td><td>Thursday</td><td>11:00 a.m.</td><td>New Year's Day</td></tr>
<tr><td>September 4, 2026</td><td>September 10, 2026</td><td>Thursday</td><td>12:00 p.m.</td><td>Labor Day</td></tr>
</table>
"""


def test_eia_petroleum_calendar_uses_default_wednesday_when_no_exception(monkeypatch):
    monkeypatch.setattr(eia_mod, "http_get_text", lambda url, **kw: "<p>Release Schedule</p><p>no exceptions here</p>")
    connector = eia_mod.EiaEventCalendarConnector()
    events = connector.fetch_upcoming(now=NOW, horizon_days=14)
    petroleum = [e for e in events if "Petroleum" in e.name]
    assert len(petroleum) == 1
    assert petroleum[0].scheduled_at.weekday() == 2  # Wednesday
    assert petroleum[0].scheduled_at.hour == 14 and petroleum[0].scheduled_at.minute == 30  # 10:30 ET = 14:30 UTC in Sept


def test_eia_petroleum_calendar_applies_a_real_holiday_shift_exception(monkeypatch):
    """September 2026's real EIA exception (from the live page, Section 24)
    shifts the normal Wednesday release to Thursday because of Labor Day."""
    monkeypatch.setattr(eia_mod, "http_get_text", lambda url, **kw: _EIA_PETROLEUM_SCHEDULE_HTML)
    connector = eia_mod.EiaEventCalendarConnector()
    now = datetime(2026, 9, 8, tzinfo=UTC)  # the Tuesday before Labor Day week
    events = connector.fetch_upcoming(now=now, horizon_days=10)
    petroleum = next(e for e in events if "Petroleum" in e.name)
    from datetime import date

    assert petroleum.scheduled_at.date() == date(2026, 9, 10)  # shifted to Thursday, not Wednesday the 9th
    assert petroleum.scheduled_at.hour == 16  # 12:00 p.m. ET = 16:00 UTC in September


def test_eia_calendar_never_fabricates_a_date_outside_the_horizon(monkeypatch):
    monkeypatch.setattr(eia_mod, "http_get_text", lambda url, **kw: "<p>Release Schedule</p>")
    connector = eia_mod.EiaEventCalendarConnector()
    events = connector.fetch_upcoming(now=NOW, horizon_days=0)
    assert events == ()  # the next Wednesday/Thursday is always > 0 days out from a Monday


# ---------------------------------------------------------------------------
# EventStore -- Section 16
# ---------------------------------------------------------------------------


def test_event_store_upsert_replaces_not_duplicates(tmp_path: Path):
    store = EventStore(tmp_path / "events.sqlite")
    e1 = _event(event_id="fomc:2026-09-16", name="FOMC Policy Statement")
    e2 = _event(event_id="fomc:2026-09-16", name="FOMC Policy Statement (updated)")
    store.upsert_event(e1)
    store.upsert_event(e2)
    upcoming = store.list_upcoming(now=NOW, horizon_days=30)
    assert len(upcoming) == 1
    assert upcoming[0].name == "FOMC Policy Statement (updated)"


def test_event_store_excludes_events_outside_the_horizon(tmp_path: Path):
    store = EventStore(tmp_path / "events.sqlite")
    store.upsert_event(_event(event_id="near", scheduled_at=NOW + timedelta(days=5)))
    store.upsert_event(_event(event_id="far", scheduled_at=NOW + timedelta(days=100)))
    upcoming = store.list_upcoming(now=NOW, horizon_days=30)
    assert [e.event_id for e in upcoming] == ["near"]


def test_event_store_excludes_past_events(tmp_path: Path):
    store = EventStore(tmp_path / "events.sqlite")
    store.upsert_event(_event(event_id="past", scheduled_at=NOW - timedelta(days=1)))
    upcoming = store.list_upcoming(now=NOW, horizon_days=30)
    assert upcoming == []
