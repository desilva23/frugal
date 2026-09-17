"""Shared plumbing for the framework adapters.

LangChain and LlamaIndex want the same three things from a plan — text to read,
provenance to cite, and the cost of getting it — in slightly different
containers. Keeping that conversion here means the two adapters differ only in
the container, which is the only thing they should differ in.

Private to the package: the frameworks are what callers import.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from frugal.cache import CacheMode, ResponseCache
from frugal.client import SerpApiClient
from frugal.errors import FrugalError
from frugal.planner import DEFAULT_BUDGET, Planner, PlanResult
from frugal.schema import Document, Evidence, Series

#: Ceiling on one call, whatever the caller asks for.
MAX_BUDGET = 25

#: Ceiling on everything an integration spends for as long as the process lives.
#:
#: The per-call cap alone did not do the job its own comment claimed. "An agent
#: looping on a tool is the usual way a search bill becomes a surprise" was the
#: stated reason for MAX_BUDGET, and a per-call cap is exactly what a loop
#: defeats: a hundred calls at twenty-five apiece is two and a half thousand
#: searches, uncapped and unreported, because every call built a fresh governor
#: and threw away the tally when it returned.
DEFAULT_SESSION_BUDGET = 200
SESSION_BUDGET_ENV = "FRUGAL_SESSION_BUDGET"


class SessionBudgetExceeded(FrugalError):
    """The process has spent its whole session allowance."""

    def __init__(self, spent: int, limit: int) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"session search budget exhausted: {spent}/{limit} searches spent by this "
            f"process. Raise it with {SESSION_BUDGET_ENV}, or start a new process."
        )


@dataclass(slots=True)
class SpendLedger:
    """Cumulative spend for one process.

    Per-call accounting tells an agent what a question cost. It does not tell
    anyone what the session cost, which is the figure whoever pays actually
    wants, and it cannot stop a loop. This is deliberately process-wide rather
    than per-client: the point is to bound what a program spends in total, so
    every integration shares one.
    """

    limit: int
    spent: int = 0
    calls: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_searches_billed": self.spent,
            "session_limit": self.limit,
            "session_remaining": self.remaining,
            "session_calls": self.calls,
        }


def _session_limit() -> int:
    raw = os.environ.get(SESSION_BUDGET_ENV, "").strip()
    if not raw:
        return DEFAULT_SESSION_BUDGET
    try:
        return max(0, int(raw))
    except ValueError:
        # A malformed override should not silently uncap spending.
        return DEFAULT_SESSION_BUDGET


_LEDGER = SpendLedger(limit=_session_limit())
_LEDGER_LOCK = threading.Lock()


def ledger() -> SpendLedger:
    """A snapshot of what this process has spent."""
    with _LEDGER_LOCK:
        return SpendLedger(limit=_LEDGER.limit, spent=_LEDGER.spent, calls=_LEDGER.calls)


def reset_ledger(limit: int | None = None) -> None:
    """Start the session accounting again. For tests and long-lived servers."""
    global _LEDGER
    with _LEDGER_LOCK:
        _LEDGER = SpendLedger(limit=_session_limit() if limit is None else limit)


def _record(spent: int) -> None:
    with _LEDGER_LOCK:
        _LEDGER.spent += spent
        _LEDGER.calls += 1


def clamp(budget: int) -> int:
    """Bound one call by the per-call ceiling and by what the session has left."""
    with _LEDGER_LOCK:
        remaining = max(0, _LEDGER.limit - _LEDGER.spent)
    return max(1, min(budget, MAX_BUDGET, remaining)) if remaining else 0


def page_content(item: Evidence) -> str:
    """The text a chain or index will actually read.

    A series is rendered as its summary rather than its points: the direction
    and magnitude are what answer a question, and fifty timestamped numbers
    would crowd a context window without adding to it.
    """
    if isinstance(item, Series):
        return item.summarise()
    if isinstance(item, Document):
        return "\n".join(part for part in (item.title, item.snippet) if part)
    return ""  # pragma: no cover - Evidence is a closed union


def provenance_metadata(item: Evidence) -> dict[str, Any]:
    """Where a result came from, so a chain can cite the search not the claim."""
    common: dict[str, Any] = {
        "engine": item.provenance.engine,
        "query": item.provenance.query,
        "rank": item.provenance.position,
        "retrieved_at": item.provenance.retrieved_at.isoformat(),
    }
    if isinstance(item, Series):
        return {**common, "kind": "series", "name": item.name, "change": item.trend()}
    assert isinstance(item, Document)
    return {
        **common,
        "kind": "document",
        "title": item.title,
        "source": item.url,
        "publisher": item.source,
    }


def cost_metadata(result: PlanResult) -> dict[str, Any]:
    """What the retrieval spent.

    Attached to the first result. A chain that ignores it is no worse off; one
    that reads it can report the cost, which a list of documents does not
    otherwise convey.
    """
    return {
        "searches_billed": result.searches_charged,
        "served_from_cache": result.cache_hits,
        "steps_planned": result.ceiling,
        "stopped_because": result.stopped_because,
        # What the whole process has spent, not just this call. An agent that
        # can only see one call's cost cannot manage a budget across many.
        **ledger().as_dict(),
    }


def run_plan(
    question: str, *, budget: int, cache_dir: str, replay: bool = False
) -> PlanResult:
    """Execute a plan for an adapter, bounded per call and per session."""
    allowance = clamp(budget)
    if allowance <= 0:
        snapshot = ledger()
        raise SessionBudgetExceeded(snapshot.spent, snapshot.limit)

    mode = CacheMode.REPLAY if replay else CacheMode.AUTO
    with SerpApiClient(cache=ResponseCache(cache_dir, mode=mode)) as client:
        result = Planner(client).run(question, budget=allowance)

    # Recorded after the fact so that cache hits, which cost nothing, do not
    # consume a session allowance that exists to bound money.
    _record(result.searches_charged)
    return result


def render_for_agent(result: PlanResult) -> str:
    """A plan rendered as text for an agent to read.

    Leads with what it cost. An agent choosing whether to search again should
    not have to infer the price of the last one.
    """
    if not result.evidence:
        return "No evidence was retrieved."

    engines = ", ".join(sorted({step.step.engine for step in result.steps}))
    session = ledger()
    lines = [
        f"Searched {engines}; {result.searches_charged} searches billed, "
        f"{result.cache_hits} from cache. "
        f"This session: {session.spent}/{session.limit} searches, "
        f"{session.remaining} remaining.",
        "",
    ]
    for index, item in enumerate(result.evidence, start=1):
        lines.append(f"[{index}] ({item.provenance.engine}) {page_content(item)}")
        if isinstance(item, Document) and item.url:
            lines.append(f"    {item.url}")
    return "\n".join(lines)


def preview_text(question: str, *, budget: int = DEFAULT_BUDGET) -> str:
    """What a question would cost, without issuing anything."""
    if not question.strip():
        return "No question was given."

    budget = clamp(budget)
    client = SerpApiClient(cache=ResponseCache(".frugal-cache", mode=CacheMode.REPLAY))
    dry = Planner(client).dry_run(question, budget=budget)
    engines = ", ".join(sorted({step.engine for step in dry.steps}))
    return (
        f"Would search {engines}, costing at most {dry.ceiling} searches "
        f"(budget {budget}). No search was issued. A real run usually costs less: "
        f"it stops early when the evidence stops improving."
    )


SEARCH_DESCRIPTION = (
    "Search across SerpApi's engines under a budget, routing the question to the "
    "engines that can answer it. Set budget to the most searches you are willing "
    "to spend. Returns evidence with the engine that produced each item, and what "
    "the call cost."
)

PLAN_DESCRIPTION = (
    "Report which engines a question would reach and the most it could cost, "
    "without issuing any search. Call this before searching when the budget "
    "matters."
)
