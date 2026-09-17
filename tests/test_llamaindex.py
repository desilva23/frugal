"""Tests for the LlamaIndex adapters.

Same bar as the LangChain ones: an agent using this can otherwise search as
often as it likes with no idea what it is spending, so the cost reporting and
the budget cap matter as much as the nodes. Everything runs from committed
fixtures and spends nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frugal.llamaindex import (
    MAX_BUDGET,
    build_retriever,
    frugal_plan_tool,
    frugal_search_tool,
    frugal_tools,
    to_nodes,
)

FIXTURES = str(Path(__file__).parent.parent / "benchmarks" / "fixtures")
QUESTION = "Has search interest in electric vehicles in India been rising or falling over time?"


@pytest.fixture
def retriever() -> object:
    return build_retriever(budget=6, cache_dir=FIXTURES, replay=True)


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------


def test_the_retriever_returns_nodes(retriever: object) -> None:
    nodes = retriever.retrieve(QUESTION)  # type: ignore[attr-defined]
    assert nodes
    assert all(node.node.text for node in nodes)


def test_every_node_carries_the_search_that_produced_it(retriever: object) -> None:
    for node in retriever.retrieve(QUESTION):  # type: ignore[attr-defined]
        assert node.node.metadata["engine"]
        assert node.node.metadata["query"]
        assert node.node.metadata["rank"] >= 1


def test_the_cost_of_the_retrieval_is_reported(retriever: object) -> None:
    nodes = retriever.retrieve(QUESTION)  # type: ignore[attr-defined]
    cost = nodes[0].node.metadata["plan_cost"]
    assert set(cost) == {
        "searches_billed",
        "served_from_cache",
        "steps_planned",
        "stopped_because",
        "session_searches_billed",
        "session_limit",
        "session_remaining",
        "session_calls",
    }


def test_scores_preserve_the_order_the_engines_returned(retriever: object) -> None:
    """Reciprocal rank: ordinal, not a similarity anyone computed."""
    scores = [node.score for node in retriever.retrieve(QUESTION)]  # type: ignore[attr-defined]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0


def test_a_series_is_rendered_as_its_summary(retriever: object) -> None:
    nodes = retriever.retrieve(QUESTION)  # type: ignore[attr-defined]
    series = [n for n in nodes if n.node.metadata["kind"] == "series"]
    assert series
    assert "observations" in series[0].node.text


def test_an_empty_query_retrieves_nothing(retriever: object) -> None:
    assert retriever.retrieve("   ") == []  # type: ignore[attr-defined]


def test_nodes_convert_without_a_retriever() -> None:
    from frugal.cache import CacheMode, ResponseCache
    from frugal.client import SerpApiClient
    from frugal.planner import Planner

    client = SerpApiClient(cache=ResponseCache(FIXTURES, mode=CacheMode.REPLAY))
    result = Planner(client).run(QUESTION, budget=6)
    assert len(to_nodes(result)) == len(result.evidence)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def test_both_tools_are_offered() -> None:
    names = {tool.metadata.name for tool in frugal_tools(cache_dir=FIXTURES, replay=True)}
    assert names == {"frugal_plan", "frugal_search"}


def test_each_tool_describes_when_to_use_it() -> None:
    assert "without issuing any search" in frugal_plan_tool().metadata.description
    assert "budget" in frugal_search_tool(replay=True).metadata.description


def test_the_plan_tool_reports_cost_without_searching() -> None:
    answer = str(frugal_plan_tool().call(question=QUESTION, budget=6))
    assert "at most" in answer
    assert "No search was issued" in answer


def test_the_search_tool_reports_what_it_spent() -> None:
    tool = frugal_search_tool(cache_dir=FIXTURES, replay=True)
    assert "searches billed" in str(tool.call(question=QUESTION, budget=6))


def test_the_search_tool_numbers_its_evidence() -> None:
    tool = frugal_search_tool(cache_dir=FIXTURES, replay=True)
    assert "[1]" in str(tool.call(question=QUESTION, budget=6))


def test_an_empty_question_is_answered_not_raised() -> None:
    for tool in frugal_tools(cache_dir=FIXTURES, replay=True):
        assert "No question" in str(tool.call(question="  "))


def test_a_failure_is_returned_as_text_an_agent_can_read(tmp_path: Path) -> None:
    """An agent cannot act on an exception it never sees."""
    tool = frugal_search_tool(cache_dir=str(tmp_path / "missing"), replay=True)
    answer = str(tool.call(question="something never recorded", budget=2))
    assert "failed" in answer.lower() or "No evidence" in answer


# --------------------------------------------------------------------------
# Consistency across the adapters
# --------------------------------------------------------------------------


def test_the_budget_cap_is_shared_across_every_integration() -> None:
    """One cap, so a caller cannot route around it by picking a framework."""
    from frugal.langchain import MAX_BUDGET as LANGCHAIN_MAX
    from frugal.mcp_server import MAX_BUDGET as MCP_MAX

    assert MAX_BUDGET == LANGCHAIN_MAX == MCP_MAX


def test_both_frameworks_report_the_same_provenance() -> None:
    """The adapters should differ only in their container."""
    from frugal.langchain import build_retriever as langchain_retriever

    nodes = build_retriever(budget=6, cache_dir=FIXTURES, replay=True).retrieve(QUESTION)
    docs = langchain_retriever(budget=6, cache_dir=FIXTURES, replay=True).invoke(QUESTION)

    assert len(nodes) == len(docs)
    for node, doc in zip(nodes, docs, strict=True):
        assert node.node.metadata["engine"] == doc.metadata["engine"]
        assert node.node.text == doc.page_content
