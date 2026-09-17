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

from frugal._integration import (
    MAX_BUDGET,
    PLAN_DESCRIPTION,
    SEARCH_DESCRIPTION,
    cost_metadata,
    page_content,
    preview_text,
    provenance_metadata,
    render_for_agent,
    run_plan,
)
from frugal.errors import FrugalError
from frugal.planner import DEFAULT_BUDGET, PlanResult

__all__ = [
    "MAX_BUDGET",
    "build_retriever",
    "frugal_plan_tool",
    "frugal_search_tool",
    "frugal_tools",
    "to_documents",
]


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


def to_documents(result: PlanResult) -> list[Any]:
    """Convert a plan's evidence into LangChain documents.

    The first document's metadata carries the plan's cost. A chain that never
    looks at it is no worse off; one that does can report what the retrieval
    spent, which is not otherwise knowable from a list of documents.
    """
    documents, _, _ = _require_langchain()
    converted: list[Any] = []
    for index, item in enumerate(result.evidence):
        metadata = provenance_metadata(item)
        if index == 0:
            metadata["plan_cost"] = cost_metadata(result)
        converted.append(documents.Document(page_content=page_content(item), metadata=metadata))
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

    class FrugalRetriever(retrievers.BaseRetriever):  # type: ignore[misc, name-defined]
        """Retrieves by planning search across SerpApi's engines."""

        def _get_relevant_documents(
            self, query: str, *, run_manager: Any = None
        ) -> list[Any]:
            if not query.strip():
                return []
            return to_documents(
                run_plan(query, budget=budget, cache_dir=cache_dir, replay=replay)
            )

    return FrugalRetriever()


def frugal_search_tool(*, cache_dir: str = ".frugal-cache", replay: bool = False) -> Any:
    """A tool an agent can call to search under a budget.

    ``replay`` serves only from recorded responses and spends nothing, which is
    how this is demonstrated and tested without a key.
    """
    _, _, tools = _require_langchain()

    def search(question: str, budget: int = DEFAULT_BUDGET) -> str:
        if not question.strip():
            return "No question was given."
        try:
            result = run_plan(question, budget=budget, cache_dir=cache_dir, replay=replay)
        except FrugalError as exc:
            # Returned rather than raised: an agent cannot act on an exception
            # it never sees.
            return f"Search failed: {type(exc).__name__}: {exc}"
        return render_for_agent(result)

    return tools.StructuredTool.from_function(
        func=search,
        name="frugal_search",
        description=SEARCH_DESCRIPTION,
    )


def frugal_plan_tool() -> Any:
    """A tool an agent can call to price a question before searching."""
    _, _, tools = _require_langchain()
    return tools.StructuredTool.from_function(
        func=preview_text,
        name="frugal_plan",
        description=PLAN_DESCRIPTION,
    )


def frugal_tools(*, cache_dir: str = ".frugal-cache", replay: bool = False) -> Sequence[Any]:
    """Both tools, for handing to an agent in one go."""
    return [frugal_plan_tool(), frugal_search_tool(cache_dir=cache_dir, replay=replay)]
