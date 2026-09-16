"""Tests for budget enforcement.

The budget is the constraint the project is organised around, so the bar is that
overspending is impossible rather than merely discouraged. The concurrency tests
matter most: a naive counter that reads the remaining budget, decides, then
increments will pass every single-threaded test in this file and still overspend
the moment a plan fans out.
"""

from __future__ import annotations

import threading

import pytest

from frugal.budget import BudgetGovernor
from frugal.errors import BudgetExceeded


def spend(governor: BudgetGovernor, times: int, engine: str = "google") -> None:
    for _ in range(times):
        with governor.reserve(engine=engine):
            pass


# --------------------------------------------------------------------------
# Basic accounting
# --------------------------------------------------------------------------


def test_a_committed_reservation_is_spent() -> None:
    governor = BudgetGovernor(limit=5)
    spend(governor, 3)
    assert governor.spent == 3
    assert governor.available == 2


def test_spending_the_whole_budget_exhausts_it() -> None:
    governor = BudgetGovernor(limit=2)
    spend(governor, 2)
    assert governor.exhausted
    assert not governor.can_afford()


def test_spending_past_the_limit_is_refused() -> None:
    governor = BudgetGovernor(limit=1)
    spend(governor, 1)
    with pytest.raises(BudgetExceeded), governor.reserve():
        pass


def test_a_refusal_reports_what_was_spent_and_allowed() -> None:
    governor = BudgetGovernor(limit=3)
    spend(governor, 3)
    with pytest.raises(BudgetExceeded) as excinfo, governor.reserve():
        pass
    assert excinfo.value.spent == 3
    assert excinfo.value.limit == 3
    assert "3/3" in str(excinfo.value)


def test_a_multi_unit_step_is_refused_whole_rather_than_part_spent() -> None:
    governor = BudgetGovernor(limit=5)
    spend(governor, 4)
    with pytest.raises(BudgetExceeded), governor.reserve(cost=3):
        pass
    assert governor.spent == 4


def test_a_zero_budget_permits_nothing() -> None:
    governor = BudgetGovernor(limit=0)
    assert governor.exhausted
    with pytest.raises(BudgetExceeded), governor.reserve():
        pass


def test_a_negative_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        BudgetGovernor(limit=-1)


def test_a_non_positive_cost_is_rejected() -> None:
    governor = BudgetGovernor(limit=5)
    for cost in (0, -1):
        with pytest.raises(ValueError, match="at least one unit"), governor.reserve(cost=cost):
            pass


# --------------------------------------------------------------------------
# What does and does not charge
# --------------------------------------------------------------------------


def test_a_released_reservation_charges_nothing() -> None:
    """A cache hit cost SerpApi nothing and must cost the budget nothing."""
    governor = BudgetGovernor(limit=3)
    with governor.reserve() as claim:
        claim.release()
    assert governor.spent == 0
    assert governor.available == 3
    assert governor.report().cache_hits == 1


def test_a_failed_search_charges_nothing() -> None:
    governor = BudgetGovernor(limit=3)
    with pytest.raises(RuntimeError), governor.reserve():
        raise RuntimeError("the search failed")
    assert governor.spent == 0
    assert governor.available == 3


def test_release_after_commit_does_not_refund() -> None:
    governor = BudgetGovernor(limit=3)
    with governor.reserve() as claim:
        claim.commit()
        claim.release()
    assert governor.spent == 1


def test_commit_is_idempotent() -> None:
    governor = BudgetGovernor(limit=3)
    with governor.reserve() as claim:
        claim.commit()
        claim.commit()
    assert governor.spent == 1


def test_reservations_are_held_while_in_flight() -> None:
    """Budget claimed by one worker must not be offered to another."""
    governor = BudgetGovernor(limit=2)
    with governor.reserve():
        assert governor.reserved == 1
        assert governor.available == 1
    assert governor.reserved == 0


# --------------------------------------------------------------------------
# Concurrency — where a naive counter fails
# --------------------------------------------------------------------------


def test_concurrent_workers_cannot_collectively_overspend() -> None:
    """The failure a read-decide-increment counter has and this does not."""
    limit = 10
    governor = BudgetGovernor(limit=limit)
    refused = 0
    refused_lock = threading.Lock()
    start = threading.Barrier(32)

    def worker() -> None:
        nonlocal refused
        start.wait()
        try:
            with governor.reserve():
                pass
        except BudgetExceeded:
            with refused_lock:
                refused += 1

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert governor.spent == limit
    assert refused == 32 - limit
    assert governor.reserved == 0


def test_concurrent_releases_return_budget_correctly() -> None:
    governor = BudgetGovernor(limit=8)
    start = threading.Barrier(8)

    def worker() -> None:
        start.wait()
        with governor.reserve() as claim:
            claim.release()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert governor.spent == 0
    assert governor.reserved == 0
    assert governor.available == 8


def test_a_slow_listener_does_not_stall_other_workers() -> None:
    """The listener fires outside the lock, so a terminal redraw cannot block."""
    entered = threading.Event()
    proceed = threading.Event()

    def listener(*, spent: int, limit: int, engine: str | None) -> None:
        entered.set()
        proceed.wait(timeout=5)

    governor = BudgetGovernor(limit=4, listener=listener)
    blocker = threading.Thread(target=lambda: spend(governor, 1))
    blocker.start()
    assert entered.wait(timeout=5)

    # The listener is still inside its callback; another worker must proceed.
    spend(governor, 1)
    assert governor.spent == 2

    proceed.set()
    blocker.join(timeout=5)


# --------------------------------------------------------------------------
# Projection — dry runs
# --------------------------------------------------------------------------


def test_projection_does_not_spend() -> None:
    governor = BudgetGovernor(limit=10)
    governor.project([1, 1, 1])
    assert governor.spent == 0


def test_projection_reports_a_plan_that_fits() -> None:
    projection = BudgetGovernor(limit=10).project([1, 1, 2])
    assert projection.fits
    assert projection.projected_spend == 4
    assert projection.headroom == 6
    assert "to spare" in projection.describe()


def test_projection_refuses_steps_that_do_not_fit() -> None:
    projection = BudgetGovernor(limit=3).project([1, 1, 1, 1, 1])
    assert not projection.fits
    assert len(projection.affordable) == 3
    assert len(projection.refused) == 2
    assert "would be refused" in projection.describe()


def test_projection_preserves_plan_order() -> None:
    """A later cheap step must not be promoted past an earlier expensive one."""
    projection = BudgetGovernor(limit=2).project([3, 1])
    assert projection.refused == (3,)
    assert projection.affordable == (1,)


def test_projection_accounts_for_what_is_already_spent() -> None:
    governor = BudgetGovernor(limit=5)
    spend(governor, 3)
    projection = governor.project([1, 1, 1])
    assert len(projection.affordable) == 2
    assert projection.projected_spend == 5


def test_projection_rejects_a_non_positive_cost() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        BudgetGovernor(limit=5).project([1, 0])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_report_breaks_spend_down_by_engine() -> None:
    governor = BudgetGovernor(limit=10)
    spend(governor, 2, engine="google")
    spend(governor, 1, engine="google_news")
    assert governor.report().by_engine == {"google": 2, "google_news": 1}


def test_a_plan_that_ran_out_records_the_budget_as_binding() -> None:
    governor = BudgetGovernor(limit=1)
    spend(governor, 1)
    with pytest.raises(BudgetExceeded), governor.reserve():
        pass
    report = governor.report()
    assert report.was_binding
    assert report.refusals == 1


def test_a_plan_that_stopped_itself_records_the_budget_as_not_binding() -> None:
    """The distinction that makes a benchmark readable: saturation, not exhaustion."""
    governor = BudgetGovernor(limit=10)
    spend(governor, 3)
    report = governor.report()
    assert not report.was_binding
    assert report.remaining == 7


def test_utilisation_is_the_fraction_consumed() -> None:
    governor = BudgetGovernor(limit=4)
    spend(governor, 1)
    assert governor.report().utilisation == 0.25


def test_utilisation_of_a_zero_budget_is_zero_not_an_error() -> None:
    assert BudgetGovernor(limit=0).report().utilisation == 0.0


def test_report_serialises_for_the_benchmark() -> None:
    governor = BudgetGovernor(limit=5)
    spend(governor, 2)
    payload = governor.report().as_dict()
    assert payload["spent"] == 2
    assert payload["remaining"] == 3
    assert payload["was_binding"] is False


def test_listener_sees_each_commit() -> None:
    seen: list[tuple[int, str | None]] = []
    governor = BudgetGovernor(
        limit=5, listener=lambda *, spent, limit, engine: seen.append((spent, engine))
    )
    spend(governor, 2, engine="google_news")
    assert seen == [(1, "google_news"), (2, "google_news")]


def test_listener_is_not_told_about_released_reservations() -> None:
    seen: list[int] = []
    governor = BudgetGovernor(
        limit=5, listener=lambda *, spent, limit, engine: seen.append(spent)
    )
    with governor.reserve() as claim:
        claim.release()
    assert seen == []
