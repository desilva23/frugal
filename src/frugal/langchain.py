"""LangChain adapters.

Two ways in, because LangChain has two shapes for this and they are used
differently.

:class:`FrugalRetriever` is a ``BaseRetriever``, so it drops into an existing
retrieval chain where a vector store would go. The difference a caller sees is
that the documents come back with what they cost attached.

:func:`frugal_search_tool` is a tool an agent can choose to call, and
:func:`frugal_plan_tool` lets it ask what a question would cost before choosing.
That second one is the reason this adapter is worth having: a LangChain agent
can otherwise search as many times as it likes with no idea what it is spending.

Every document carries its engine, query and rank in ``metadata``, so a chain
that cites its sources can name the search that produced each one rather than
asserting the claim.

The SDK is an optional extra and imported lazily, for the same reason the MCP
server's is: someone using the planner directly should not have to install a
framework they are not using.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from frugal.cache import CacheMode, ResponseCache
from frugal.client import SerpApiClient
from frugal.errors import FrugalError
from frugal.planner import DEFAULT_BUDGET, Planner, PlanResult
from frugal.schema import Document as FrugalDocument
from frugal.schema import Evidence, Series

#: Ceiling on one call, whatever the caller asks for. An agent in a loop is the
#: usual way a search bill becomes a surprise.
MAX_BUDGET = 25


def _require_langchain() -> Any:
    """Import LangChain, with an error that says how to get it."""
    try:
        import langchain_core.documents as documents
        import langchain_core.retrievers as retrievers
        import langchain_core.tools as tools
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise FrugalError(
            "the LangChain adapters need langchain-core:\n\n"
            "    pip install 'frugal[langchain]'\n"
        ) from exc
    return documents, retrievers, tools


def _page_content(item: Evidence) -> str:
    """What a chain will read.

    A series is rendered as its summary rather than its raw points: the
    direction and magnitude are what answer a question, and fifty timestamped
    numbers would crowd the context window without adding to it.
    """
    if isinstance(item, Series):
        return item.summarise()
    assert isinstance(item, FrugalDocument)
    return "\n".join(part for part in (item.title, item.snippet) if part)


def _metadata(item: Evidence) -> dict[str, Any]:
    """Provenance, so a chain can cite the search rather than assert the claim."""
    common: dict[str, Any] = {
        "engine": item.provenance.engine,
        "query": item.provenance.query,
        "rank": item.provenance.position,
        "retrieved_at": item.provenance.retrieved_at.isoformat(),
    }
    if isinstance(item, Series):
        return {**common, "kind": "series", "name": item.name, "change": item.trend()}
    return {
        **common,
        "kind": "document",
        "title": item.title,
        "source": item.url,
        "publisher": item.source,
    }


def to_documents(result: PlanResult) -> list[Any]:
    """Convert a plan's evidence into LangChain documents.

    The first document's metadata carries the plan's cost. A chain that never
    looks at it is no worse off; one that does can report what the retrieval
    spent, which is not otherwise knowable from a list of documents.
    """
    documents, _, _ = _require_langchain()
    converted: list[Any] = []
    for index, item in enumerate(result.evidence):
        metadata = _metadata(item)
        if index == 0:
            metadata["plan_cost"] = {
                "searches_billed": result.searches_charged,
                "served_from_cache": result.cache_hits,
                "steps_planned": result.ceiling,
                "stopped_because": result.stopped_because,
            }
        converted.append(documents.Document(page_content=_page_content(item), metadata=metadata))
    return converted


def build_retriever(
    *,
    budget: int = DEFAULT_BUDGET,
    cache_dir: str = ".frugal-cache",
    replay: bool = False,
) -> Any:
    """A ``BaseRetriever`` that plans its searching.

    Drops in where a vector store would go. Unlike one, it decides which engines
    to query and stops when the evidence stops improving.
    """
    _, retrievers, _ = _require_langchain()
    budget = max(1, min(budget, MAX_BUDGET))
    mode = CacheMode.REPLAY if replay else CacheMode.AUTO

    class FrugalRetriever(retrievers.BaseRetriever):  # type: ignore[misc, name-defined]
        """Retrieves by planning search across SerpApi's engines."""

        def _get_relevant_documents(
            self, query: str, *, run_manager: Any = None
        ) -> list[Any]:
            if not query.strip():
                return []
            with SerpApiClient(cache=ResponseCache(cache_dir, mode=mode)) as client:
                return to_documents(Planner(client).run(query, budget=budget))

    return FrugalRetriever()


def _search(
    question: str,
    budget: int = DEFAULT_BUDGET,
    *,
    cache_dir: str = ".frugal-cache",
    replay: bool = False,
) -> str:
    """Run a plan and render it for an agent to read."""
    if not question.strip():
        return "No question was given."

    budget = max(1, min(budget, MAX_BUDGET))
    mode = CacheMode.REPLAY if replay else CacheMode.AUTO
    try:
        with SerpApiClient(cache=ResponseCache(cache_dir, mode=mode)) as client:
            result = Planner(client).run(question, budget=budget)
    except FrugalError as exc:
        # Returned rather than raised: an agent should be told what went wrong in
        # a form it can read and act on.
        return f"Search failed: {type(exc).__name__}: {exc}"

    if not result.evidence:
        return "No evidence was retrieved."

    lines = [
        f"Searched {', '.join(sorted({s.step.engine for s in result.steps}))}; "
        f"{result.searches_charged} searches billed, {result.cache_hits} from cache.",
        "",
    ]
    for index, item in enumerate(result.evidence, start=1):
        lines.append(f"[{index}] ({item.provenance.engine}) {_page_content(item)}")
        if isinstance(item, FrugalDocument) and item.url:
            lines.append(f"    {item.url}")
    return "\n".join(lines)


def _preview(question: str, budget: int = DEFAULT_BUDGET) -> str:
    """Report a plan's cost without issuing anything."""
    if not question.strip():
        return "No question was given."

    budget = max(1, min(budget, MAX_BUDGET))
    client = SerpApiClient(cache=ResponseCache(".frugal-cache", mode=CacheMode.REPLAY))
    dry = Planner(client).dry_run(question, budget=budget)
    engines = ", ".join(sorted({step.engine for step in dry.steps}))
    return (
        f"Would search {engines}, costing at most {dry.ceiling} searches "
        f"(budget {budget}). No search was issued. A real run usually costs less: "
        f"it stops early when the evidence stops improving."
    )


def frugal_search_tool(*, cache_dir: str = ".frugal-cache", replay: bool = False) -> Any:
    """A tool an agent can call to search under a budget.

    ``replay`` serves only from recorded responses and spends nothing, which is
    how this is demonstrated and tested without a key.
    """
    _, _, tools = _require_langchain()

    def search(question: str, budget: int = DEFAULT_BUDGET) -> str:
        return _search(question, budget, cache_dir=cache_dir, replay=replay)

    return tools.StructuredTool.from_function(
        func=search,
        name="frugal_search",
        description=(
            "Search across SerpApi's engines under a budget, routing the question "
            "to the engines that can answer it. Set budget to the most searches "
            "you are willing to spend. Returns evidence with the engine that "
            "produced each item, and what the call cost."
        ),
    )


def frugal_plan_tool() -> Any:
    """A tool an agent can call to price a question before searching."""
    _, _, tools = _require_langchain()
    return tools.StructuredTool.from_function(
        func=_preview,
        name="frugal_plan",
        description=(
            "Report which engines a question would reach and the most it could "
            "cost, without issuing any search. Call this before frugal_search "
            "when the budget matters."
        ),
    )


def frugal_tools(*, cache_dir: str = ".frugal-cache", replay: bool = False) -> Sequence[Any]:
    """Both tools, for handing to an agent in one go."""
    return [frugal_plan_tool(), frugal_search_tool(cache_dir=cache_dir, replay=replay)]
