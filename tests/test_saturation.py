"""Tests for the stopping rule.

Two failures matter and they pull in opposite directions: stopping a plan that
still had things to find, and continuing one that has run dry. The patience
tests are where that tension is resolved.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from frugal.saturation import SaturationMonitor
from frugal.schema import Document, Provenance

NOW = datetime(2026, 9, 17, tzinfo=UTC)


def docs(*names: str) -> list[Document]:
    return [
        Document(
            title=name,
            url=f"https://example.com/{name}",
            provenance=Provenance("google", "q", 1, NOW),
        )
        for name in names
    ]


# --------------------------------------------------------------------------
# Counting what is new
# --------------------------------------------------------------------------


def test_a_first_batch_is_entirely_new() -> None:
    observation = SaturationMonitor().observe(docs("a", "b", "c"))
    assert observation.new_items == 3
    assert observation.marginal_novelty == 1.0


def test_previously_seen_items_do_not_count_as_new() -> None:
    monitor = SaturationMonitor()
    monitor.observe(docs("a", "b"))
    observation = monitor.observe(docs("a", "b", "c"))
    assert observation.new_items == 1
    assert monitor.unique_count == 3


def test_duplicates_within_one_batch_count_once() -> None:
    """Two engines returning the same page in one round is one piece of evidence."""
    observation = SaturationMonitor().observe(docs("a", "a", "b"))
    assert observation.new_items == 2


def test_equivalent_urls_are_one_item() -> None:
    monitor = SaturationMonitor()
    monitor.observe(docs("a"))
    tracked = Document(
        title="a",
        url="https://www.example.com/a?utm_source=x",
        provenance=Provenance("google_news", "q", 1, NOW),
    )
    assert monitor.observe([tracked]).new_items == 0


def test_unique_filters_without_recording() -> None:
    monitor = SaturationMonitor()
    monitor.observe(docs("a"))
    batch = docs("a", "b")
    assert [d.title for d in monitor.unique(batch)] == ["b"]
    assert monitor.unique_count == 1


def test_has_seen_reports_membership() -> None:
    monitor = SaturationMonitor()
    monitor.observe(docs("a"))
    assert monitor.has_seen(docs("a")[0])
    assert not monitor.has_seen(docs("z")[0])


# --------------------------------------------------------------------------
# Stopping, and not stopping
# --------------------------------------------------------------------------


def test_a_plan_that_runs_dry_stops() -> None:
    monitor = SaturationMonitor()
    monitor.observe(docs("a", "b", "c", "d"))
    monitor.observe(docs("e", "f", "g", "h"))
    assert not monitor.saturated
    monitor.observe(docs("a", "b", "c", "d"))
    assert monitor.saturated


def test_one_thin_batch_does_not_stop_a_live_plan() -> None:
    """With patience raised, a reformulation that landed badly is survivable.

    The default is one, because the planner observes a whole round -- several
    engines answering in parallel -- where a thin result is strong evidence. A
    caller observing single queries should raise it, and this is that caller.
    """
    monitor = SaturationMonitor(patience=2)
    monitor.observe(docs("a", "b", "c", "d"))
    monitor.observe(docs("a", "b", "c", "e"))
    monitor.observe(docs("w", "x", "y", "z"))
    assert not monitor.saturated


def test_a_productive_batch_resets_patience() -> None:
    monitor = SaturationMonitor(patience=2)
    monitor.observe(docs("a", "b"))
    monitor.observe(docs("a", "b"))
    monitor.observe(docs("c", "d"))
    monitor.observe(docs("a", "b"))
    assert not monitor.saturated


def test_a_plan_cannot_stop_before_the_minimum_batches() -> None:
    """Stopping after one batch never gives a second engine a chance."""
    monitor = SaturationMonitor(min_batches=3, patience=1)
    monitor.observe([])
    monitor.observe([])
    assert not monitor.saturated
    monitor.observe([])
    assert monitor.saturated


def test_empty_batches_saturate_rather_than_dividing_by_zero() -> None:
    monitor = SaturationMonitor()
    monitor.observe(docs("a", "b"))
    assert monitor.observe([]).marginal_novelty == 0.0
    monitor.observe([])
    assert monitor.saturated


def test_the_default_patience_stops_on_one_thin_round() -> None:
    """A round where every engine returned nothing new is not an unlucky query."""
    monitor = SaturationMonitor()
    monitor.observe(docs("a", "b"))
    monitor.observe(docs("a", "b"))
    assert monitor.saturated


def test_saturation_is_sticky() -> None:
    monitor = SaturationMonitor()
    for _ in range(4):
        monitor.observe(docs("a"))
    assert monitor.saturated
    assert monitor.observe(docs("x", "y", "z")).reason == "already stopped"
    assert monitor.saturated


def test_a_looser_threshold_stops_sooner() -> None:
    strict = SaturationMonitor(threshold=0.1)
    loose = SaturationMonitor(threshold=0.9)
    for _ in range(3):
        batch = docs("a", "b", "c", "new" + str(_))
        strict.observe(batch)
        loose.observe(batch)
    assert loose.saturated
    assert not strict.saturated


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_the_curve_records_cumulative_unique_evidence() -> None:
    """The benchmark's headline figure: evidence against effort."""
    monitor = SaturationMonitor()
    monitor.observe(docs("a", "b"))
    monitor.observe(docs("c"))
    monitor.observe(docs("a"))
    assert monitor.report().curve == (2, 3, 3)


def test_efficiency_is_unique_evidence_per_search() -> None:
    monitor = SaturationMonitor()
    monitor.observe(docs("a", "b"))
    monitor.observe(docs("c", "d"))
    assert monitor.report().efficiency == 2.0


def test_efficiency_of_an_unused_monitor_is_zero() -> None:
    assert SaturationMonitor().report().efficiency == 0.0


def test_the_report_records_where_the_plan_stopped() -> None:
    monitor = SaturationMonitor()
    for _ in range(4):
        monitor.observe(docs("a"))
    report = monitor.report()
    assert report.saturated
    assert report.stopped_at == 2


def test_a_plan_that_never_saturated_records_no_stop() -> None:
    monitor = SaturationMonitor()
    monitor.observe(docs("a", "b"))
    monitor.observe(docs("c", "d"))
    assert monitor.report().stopped_at is None


def test_the_report_serialises_for_the_benchmark() -> None:
    monitor = SaturationMonitor()
    monitor.observe(docs("a", "b"))
    payload = monitor.report().as_dict()
    assert payload["unique_items"] == 2
    assert payload["curve"] == [2]


def test_observations_explain_themselves() -> None:
    explanation = SaturationMonitor().observe(docs("a", "b")).explain()
    assert "2/2 new" in explanation
    assert "too early to stop" in explanation


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold"):
        SaturationMonitor(threshold=1.5)
    with pytest.raises(ValueError, match="patience"):
        SaturationMonitor(patience=0)
    with pytest.raises(ValueError, match="min_batches"):
        SaturationMonitor(min_batches=0)
