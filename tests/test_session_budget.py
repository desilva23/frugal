"""Tests for the process-wide spend ledger.

The per-call cap alone did not do the job its own comment claimed. "An agent
looping on a tool is the usual way a search bill becomes a surprise" was the
stated reason for it, and a per-call cap is precisely what a loop defeats: every
call built a fresh governor and discarded the tally on return, so a hundred calls
at twenty-five apiece spent two and a half thousand searches uncapped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frugal._integration import (
    DEFAULT_SESSION_BUDGET,
    MAX_BUDGET,
    SESSION_BUDGET_ENV,
    SessionBudgetExceeded,
    SpendLedger,
    _record,
    clamp,
    cost_metadata,
    ledger,
    reset_ledger,
    run_plan,
)

FIXTURES = str(Path(__file__).parent.parent / "benchmarks" / "fixtures")
QUESTION = "Has search interest in electric vehicles in India been rising or falling over time?"


@pytest.fixture(autouse=True)
def fresh_ledger() -> None:
    """Every test starts from a clean tally; the ledger is process-wide."""
    reset_ledger(limit=10)


# --------------------------------------------------------------------------
# A loop is bounded
# --------------------------------------------------------------------------


def test_a_call_cannot_exceed_what_the_session_has_left() -> None:
    assert clamp(MAX_BUDGET) == 10
    _record(8)
    assert clamp(MAX_BUDGET) == 2


def test_an_exhausted_session_permits_nothing() -> None:
    _record(10)
    assert clamp(MAX_BUDGET) == 0


def test_a_loop_is_stopped_rather_than_merely_slowed() -> None:
    """The case the per-call cap did not cover."""
    _record(10)
    with pytest.raises(SessionBudgetExceeded, match="10/10"):
        run_plan(QUESTION, budget=25, cache_dir=FIXTURES, replay=True)


def test_the_refusal_says_how_to_lift_it() -> None:
    _record(10)
    with pytest.raises(SessionBudgetExceeded) as excinfo:
        run_plan(QUESTION, budget=25, cache_dir=FIXTURES, replay=True)
    assert SESSION_BUDGET_ENV in str(excinfo.value)


def test_the_per_call_ceiling_still_applies_below_the_session_limit() -> None:
    reset_ledger(limit=1000)
    assert clamp(10_000) == MAX_BUDGET


# --------------------------------------------------------------------------
# What does and does not consume the allowance
# --------------------------------------------------------------------------


def test_cache_hits_do_not_consume_the_session_allowance() -> None:
    """The allowance bounds money, and a cached search cost none."""
    run_plan(QUESTION, budget=6, cache_dir=FIXTURES, replay=True)
    assert ledger().spent == 0


def test_a_completed_call_is_counted() -> None:
    run_plan(QUESTION, budget=6, cache_dir=FIXTURES, replay=True)
    assert ledger().calls == 1


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_cost_metadata_carries_the_session_total() -> None:
    """An agent seeing only one call's cost cannot manage a budget across many."""
    from frugal.cache import CacheMode, ResponseCache
    from frugal.client import SerpApiClient
    from frugal.planner import Planner

    client = SerpApiClient(cache=ResponseCache(FIXTURES, mode=CacheMode.REPLAY))
    payload = cost_metadata(Planner(client).run(QUESTION, budget=6))
    for field in (
        "session_searches_billed",
        "session_limit",
        "session_remaining",
        "session_calls",
    ):
        assert field in payload


def test_the_ledger_reports_what_is_left() -> None:
    _record(4)
    snapshot = ledger()
    assert snapshot.spent == 4
    assert snapshot.remaining == 6
    assert not snapshot.exhausted


def test_a_snapshot_does_not_move_when_more_is_spent() -> None:
    snapshot = ledger()
    _record(3)
    assert snapshot.spent == 0


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_the_limit_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SESSION_BUDGET_ENV, "42")
    reset_ledger()
    assert ledger().limit == 42


def test_a_malformed_override_does_not_uncap_spending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back to unlimited on a typo would be the worst possible default."""
    monkeypatch.setenv(SESSION_BUDGET_ENV, "lots")
    reset_ledger()
    assert ledger().limit == DEFAULT_SESSION_BUDGET


def test_an_unset_environment_uses_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SESSION_BUDGET_ENV, raising=False)
    reset_ledger()
    assert ledger().limit == DEFAULT_SESSION_BUDGET


def test_a_zero_limit_permits_nothing() -> None:
    reset_ledger(limit=0)
    assert clamp(5) == 0


def test_the_ledger_serialises_for_a_caller() -> None:
    assert set(SpendLedger(limit=10, spent=3, calls=2).as_dict()) == {
        "session_searches_billed",
        "session_limit",
        "session_remaining",
        "session_calls",
    }
