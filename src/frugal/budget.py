"""Enforcing a hard limit on what a plan may spend.

A budget is the constraint the whole project is organised around: "answer this
in at most twelve searches" has to be a real bound, not advice the planner may
exceed when a step looks promising. Everything here exists to make overspending
impossible rather than merely discouraged.

Three properties matter.

*Spending is atomic.* Plans fan out across engines concurrently, and a naive
counter — read the remaining budget, decide, then increment — lets several
workers each see room for one more search and collectively spend past the limit.
Reservation and commit happen under one lock, so the limit holds however many
workers are in flight.

*Cache hits are free.* The budget measures money, not calls. A search served
from disk charged nothing to SerpApi and must charge nothing here, or the
planner will stop early believing it has spent what it has not.

*Refusals are informative.* A plan that stops because it ran out of budget
should be able to say what it wanted next and what that would have cost, because
that is the interesting part of the result — it is the evidence the budget was
binding rather than generous.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from frugal.errors import BudgetExceeded


class SpendListener(Protocol):
    """Notified whenever the budget actually moves.

    The live demo uses this to tick a cost counter as a plan executes, which is
    the clearest way to show that the budget is doing something.
    """

    def __call__(self, *, spent: int, limit: int, engine: str | None) -> None: ...


@dataclass(frozen=True, slots=True)
class Projection:
    """What a proposed sequence of steps would cost, computed without spending.

    Powers dry runs: the planner builds a full plan, asks what it would cost, and
    reports that before issuing anything. Finding out you overspent afterwards is
    the failure mode this exists to prevent.
    """

    affordable: tuple[int, ...]
    refused: tuple[int, ...]
    projected_spend: int
    limit: int

    @property
    def fits(self) -> bool:
        """Whether every proposed step fits within the budget."""
        return not self.refused

    @property
    def headroom(self) -> int:
        """Searches that would remain unspent."""
        return self.limit - self.projected_spend

    def describe(self) -> str:
        fitted = len(self.affordable)
        total = fitted + len(self.refused)
        if self.fits:
            return (
                f"all {total} steps fit: {self.projected_spend}/{self.limit} searches, "
                f"{self.headroom} to spare"
            )
        return (
            f"{fitted} of {total} steps fit within {self.limit} searches; "
            f"{len(self.refused)} would be refused"
        )


@dataclass(slots=True)
class BudgetReport:
    """What a plan actually spent, for the trace and the benchmark."""

    limit: int
    spent: int
    refusals: int
    cache_hits: int
    by_engine: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    @property
    def utilisation(self) -> float:
        """Fraction of the budget consumed, or 0.0 for a zero budget."""
        return self.spent / self.limit if self.limit else 0.0

    @property
    def was_binding(self) -> bool:
        """Whether the budget stopped the plan rather than the plan stopping itself.

        The distinction matters when reading a benchmark: a plan that finished
        with headroom stopped because its evidence saturated, which is the
        result worth reporting.
        """
        return self.refusals > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "spent": self.spent,
            "remaining": self.remaining,
            "refusals": self.refusals,
            "cache_hits": self.cache_hits,
            "utilisation": round(self.utilisation, 4),
            "was_binding": self.was_binding,
            "by_engine": dict(sorted(self.by_engine.items())),
        }


class Reservation:
    """A claim on part of the budget, held while a search is in flight.

    Reserved searches are unavailable to other workers immediately, then either
    committed when the search was actually billed or released when it was not —
    served from cache, or failed. Without the reserve step, two concurrent
    workers could each find room for the last search.
    """

    __slots__ = ("_governor", "_settled", "cost", "engine")

    def __init__(self, governor: BudgetGovernor, cost: int, engine: str | None) -> None:
        self._governor = governor
        self.cost = cost
        self.engine = engine
        self._settled = False

    @property
    def settled(self) -> bool:
        return self._settled

    def commit(self) -> None:
        """Convert the reservation into spend. Idempotent."""
        if self._settled:
            return
        self._settled = True
        self._governor._commit(self.cost, self.engine)

    def release(self) -> None:
        """Return the reservation unspent. Idempotent.

        Called when a search cost nothing — a cache hit — or when it failed.
        """
        if self._settled:
            return
        self._settled = True
        self._governor._release(self.cost, free=True)


class BudgetGovernor:
    """Tracks and enforces the search budget for one plan.

    Safe to share across threads. The usual interaction is :meth:`reserve`,
    which commits on a clean exit and releases if the block raises, so a failed
    search never charges:

    .. code-block:: python

        with governor.reserve(engine="google") as claim:
            response = client.search("google", q=query)
            if response.from_cache:
                claim.release()
    """

    def __init__(
        self,
        limit: int,
        *,
        listener: SpendListener | None = None,
    ) -> None:
        if limit < 0:
            raise ValueError(f"budget limit cannot be negative, got {limit}")
        self.limit = limit
        self._spent = 0
        self._reserved = 0
        self._refusals = 0
        self._cache_hits = 0
        self._by_engine: dict[str, int] = {}
        self._listener = listener
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    @property
    def reserved(self) -> int:
        """Searches claimed by in-flight requests but not yet billed."""
        with self._lock:
            return self._reserved

    @property
    def available(self) -> int:
        """Searches that could be started right now.

        Excludes reservations in flight. A worker asking what it may spend must
        not be told about budget another worker has already claimed.
        """
        with self._lock:
            return max(0, self.limit - self._spent - self._reserved)

    @property
    def exhausted(self) -> bool:
        return self.available == 0

    def can_afford(self, cost: int = 1) -> bool:
        """Whether ``cost`` searches could be started right now.

        Advisory only. A concurrent worker may claim the budget between this
        call and a reservation, which is why :meth:`reserve` re-checks under the
        lock rather than trusting an earlier answer.
        """
        return self.available >= cost

    # -- spending ---------------------------------------------------------

    @contextmanager
    def reserve(self, cost: int = 1, *, engine: str | None = None) -> Iterator[Reservation]:
        """Claim ``cost`` searches for the duration of the block.

        Commits on a clean exit; releases if the block raises, so a failed
        search is never billed. Call :meth:`Reservation.release` inside the block
        when the search turned out to be free.

        :raises BudgetExceeded: if the budget cannot cover ``cost``.
        :raises ValueError: if ``cost`` is not positive.
        """
        claim = self._open(cost, engine)
        try:
            yield claim
        except BaseException:
            claim.release()
            raise
        else:
            claim.commit()

    def _open(self, cost: int, engine: str | None) -> Reservation:
        if cost <= 0:
            raise ValueError(f"a search must cost at least one unit, got {cost}")
        with self._lock:
            if self.limit - self._spent - self._reserved < cost:
                self._refusals += 1
                raise BudgetExceeded(self._spent, self.limit)
            self._reserved += cost
        return Reservation(self, cost, engine)

    def _commit(self, cost: int, engine: str | None) -> None:
        with self._lock:
            self._reserved -= cost
            self._spent += cost
            if engine:
                self._by_engine[engine] = self._by_engine.get(engine, 0) + cost
            spent, limit = self._spent, self.limit
        if self._listener is not None:
            # Fired outside the lock: a listener that draws to a terminal should
            # not be able to stall every other worker.
            self._listener(spent=spent, limit=limit, engine=engine)

    def _release(self, cost: int, *, free: bool = False) -> None:
        with self._lock:
            self._reserved -= cost
            if free:
                self._cache_hits += 1

    # -- planning ---------------------------------------------------------

    def project(self, costs: Sequence[int]) -> Projection:
        """Report what ``costs`` would spend, without spending anything.

        Steps are considered in order and a step that does not fit is refused
        rather than skipped over in favour of a cheaper later one: a plan is a
        sequence, and reordering it here would describe a plan that will not run.
        """
        with self._lock:
            available = max(0, self.limit - self._spent - self._reserved)

        affordable: list[int] = []
        refused: list[int] = []
        running = 0

        for cost in costs:
            if cost <= 0:
                raise ValueError(f"a search must cost at least one unit, got {cost}")
            if running + cost <= available:
                affordable.append(cost)
                running += cost
            else:
                refused.append(cost)

        with self._lock:
            already = self._spent

        return Projection(
            affordable=tuple(affordable),
            refused=tuple(refused),
            projected_spend=already + running,
            limit=self.limit,
        )

    def report(self) -> BudgetReport:
        """Snapshot for the plan trace and the benchmark."""
        with self._lock:
            return BudgetReport(
                limit=self.limit,
                spent=self._spent,
                refusals=self._refusals,
                cache_hits=self._cache_hits,
                by_engine=dict(self._by_engine),
            )

    def __repr__(self) -> str:
        return f"BudgetGovernor(spent={self.spent}/{self.limit}, reserved={self.reserved})"
