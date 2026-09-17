"""Shared plumbing for the framework adapters.

LangChain and LlamaIndex want the same three things from a plan — text to read,
provenance to cite, and the cost of getting it — in slightly different
containers. Keeping that conversion here means the two adapters differ only in
the container, which is the only thing they should differ in.

Private to the package: the frameworks are what callers import.
"""

from __future__ import annotations

from typing import Any

from frugal.cache import CacheMode, ResponseCache
from frugal.client import SerpApiClient
from frugal.planner import DEFAULT_BUDGET, Planner, PlanResult
from frugal.schema import Document, Evidence, Series

#: Ceiling on one call, whatever the caller asks for. An agent in a loop is the
#: usual way a search bill becomes a surprise, and an adapter that honours an
#: arbitrarily large budget is trusting a program not to have a bug in it.
MAX_BUDGET = 25


def clamp(budget: int) -> int:
    return max(1, min(budget, MAX_BUDGET))


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
    }


def run_plan(
    question: str, *, budget: int, cache_dir: str, replay: bool = False
) -> PlanResult:
    """Execute a plan for an adapter, with the budget clamped."""
    mode = CacheMode.REPLAY if replay else CacheMode.AUTO
    with SerpApiClient(cache=ResponseCache(cache_dir, mode=mode)) as client:
        return Planner(client).run(question, budget=clamp(budget))


def render_for_agent(result: PlanResult) -> str:
    """A plan rendered as text for an agent to read.

    Leads with what it cost. An agent choosing whether to search again should
    not have to infer the price of the last one.
    """
    if not result.evidence:
        return "No evidence was retrieved."

    engines = ", ".join(sorted({step.step.engine for step in result.steps}))
    lines = [
        f"Searched {engines}; {result.searches_charged} searches billed, "
        f"{result.cache_hits} from cache.",
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
