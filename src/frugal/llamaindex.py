"""LlamaIndex adapters.

The same two shapes as the LangChain adapters, in LlamaIndex's containers: a
``BaseRetriever`` that drops in where a vector index would go, and function
tools an agent can choose between.

The tool worth having is :func:`frugal_plan_tool`. An agent with a search tool
can otherwise call it as often as it likes with no idea what it is spending;
this lets it ask the price first.

Nodes carry the engine, query and rank that produced them, so a response
synthesiser citing its sources can name the search rather than assert the claim.

A note on scores. LlamaIndex expects a relevance score per node, and a planner
has no similarity to report — there is no embedding and no distance. The score
here is reciprocal rank, which preserves the order the engines returned and is
honest about being ordinal rather than a similarity anyone computed. Do not read
it as a confidence.
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
    "to_nodes",
]


def _require_llamaindex() -> Any:
    """Import LlamaIndex, with an error that says how to get it."""
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, TextNode
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise FrugalError(
            "the LlamaIndex adapters need llama-index-core:\n\n"
            "    pip install 'frugal[llamaindex]'\n"
        ) from exc
    return BaseRetriever, NodeWithScore, TextNode, FunctionTool


def to_nodes(result: PlanResult) -> list[Any]:
    """Convert a plan's evidence into scored LlamaIndex nodes.

    The first node's metadata carries what the retrieval cost, which a list of
    nodes does not otherwise convey.
    """
    _, NodeWithScore, TextNode, _ = _require_llamaindex()

    nodes: list[Any] = []
    for index, item in enumerate(result.evidence):
        metadata = provenance_metadata(item)
        if index == 0:
            metadata["plan_cost"] = cost_metadata(result)
        nodes.append(
            NodeWithScore(
                node=TextNode(text=page_content(item), metadata=metadata),
                # Ordinal, not a similarity: see the module docstring.
                score=1.0 / (index + 1),
            )
        )
    return nodes


def build_retriever(
    *,
    budget: int = DEFAULT_BUDGET,
    cache_dir: str = ".frugal-cache",
    replay: bool = False,
) -> Any:
    """A ``BaseRetriever`` that plans its searching.

    Drops in where a vector index would go. Unlike one, it decides which engines
    to query and stops when the evidence stops improving.
    """
    BaseRetriever, _, _, _ = _require_llamaindex()

    class FrugalRetriever(BaseRetriever):  # type: ignore[misc, valid-type]
        """Retrieves by planning search across SerpApi's engines."""

        def _retrieve(self, query_bundle: Any) -> list[Any]:
            query = getattr(query_bundle, "query_str", str(query_bundle))
            if not query.strip():
                return []
            return to_nodes(
                run_plan(query, budget=budget, cache_dir=cache_dir, replay=replay)
            )

    return FrugalRetriever()


def frugal_search_tool(*, cache_dir: str = ".frugal-cache", replay: bool = False) -> Any:
    """A tool an agent can call to search under a budget.

    ``replay`` serves only from recorded responses and spends nothing, which is
    how this is demonstrated and tested without a key.
    """
    _, _, _, FunctionTool = _require_llamaindex()

    def frugal_search(question: str, budget: int = DEFAULT_BUDGET) -> str:
        if not question.strip():
            return "No question was given."
        try:
            result = run_plan(question, budget=budget, cache_dir=cache_dir, replay=replay)
        except FrugalError as exc:
            # Returned rather than raised: an agent cannot act on an exception
            # it never sees.
            return f"Search failed: {type(exc).__name__}: {exc}"
        return render_for_agent(result)

    return FunctionTool.from_defaults(
        fn=frugal_search, name="frugal_search", description=SEARCH_DESCRIPTION
    )


def frugal_plan_tool() -> Any:
    """A tool an agent can call to price a question before searching."""
    _, _, _, FunctionTool = _require_llamaindex()

    def frugal_plan(question: str, budget: int = DEFAULT_BUDGET) -> str:
        return preview_text(question, budget=budget)

    return FunctionTool.from_defaults(
        fn=frugal_plan, name="frugal_plan", description=PLAN_DESCRIPTION
    )


def frugal_tools(*, cache_dir: str = ".frugal-cache", replay: bool = False) -> Sequence[Any]:
    """Both tools, for handing to an agent in one go."""
    return [frugal_plan_tool(), frugal_search_tool(cache_dir=cache_dir, replay=replay)]
