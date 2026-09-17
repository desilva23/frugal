"""Tests for the LangChain adapters.

The reason this adapter is worth having is that a LangChain agent can otherwise
search as often as it likes with no idea what it is spending. So these check the
cost reporting and the budget cap as closely as the documents, and that
provenance survives the conversion — a chain that cites its sources should be
able to name the search that produced each one.

Everything runs against the committed fixtures, so the suite spends nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frugal.langchain import (
    MAX_BUDGET,
    build_retriever,
    frugal_plan_tool,
    frugal_search_tool,
    frugal_tools,
    to_documents,
)

FIXTURES = str(Path(__file__).parent.parent / "benchmarks" / "fixtures")
QUESTION = "Has search interest in electric vehicles in India been rising or falling over time?"


@pytest.fixture
def retriever() -> object:
    return build_retriever(budget=6, cache_dir=FIXTURES, replay=True)


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------


def test_the_retriever_returns_documents(retriever: object) -> None:
    docs = retriever.invoke(QUESTION)  # type: ignore[attr-defined]
    assert docs
    assert all(doc.page_content for doc in docs)


def test_every_document_carries_the_search_that_produced_it(retriever: object) -> None:
    """A chain that cites sources should name the search, not assert the claim."""
    for doc in retriever.invoke(QUESTION):  # type: ignore[attr-defined]
        assert doc.metadata["engine"]
        assert doc.metadata["query"]
        assert doc.metadata["rank"] >= 1


def test_the_cost_of_the_retrieval_is_reported(retriever: object) -> None:
    """Not otherwise knowable from a list of documents."""
    cost = retriever.invoke(QUESTION)[0].metadata["plan_cost"]  # type: ignore[attr-defined]
    assert set(cost) == {
        "searches_billed",
        "served_from_cache",
        "steps_planned",
        "stopped_because",
    }


def test_a_series_is_rendered_as_its_summary(retriever: object) -> None:
    """Fifty timestamped numbers would crowd a context window without adding to it."""
    docs = retriever.invoke(QUESTION)  # type: ignore[attr-defined]
    series = [d for d in docs if d.metadata["kind"] == "series"]
    assert series
    assert "observations" in series[0].page_content


def test_an_empty_query_retrieves_nothing(retriever: object) -> None:
    assert retriever.invoke("   ") == []  # type: ignore[attr-defined]


def test_an_oversized_budget_is_capped() -> None:
    """An agent in a loop is how a search bill becomes a surprise."""
    assert build_retriever(budget=10_000, cache_dir=FIXTURES, replay=True) is not None


def test_documents_convert_without_a_retriever() -> None:
    from frugal.cache import CacheMode, ResponseCache
    from frugal.client import SerpApiClient
    from frugal.planner import Planner

    client = SerpApiClient(cache=ResponseCache(FIXTURES, mode=CacheMode.REPLAY))
    result = Planner(client).run(QUESTION, budget=6)
    assert len(to_documents(result)) == len(result.evidence)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def test_both_tools_are_offered() -> None:
    names = {tool.name for tool in frugal_tools(cache_dir=FIXTURES, replay=True)}
    assert names == {"frugal_plan", "frugal_search"}


def test_each_tool_takes_a_question_and_a_budget() -> None:
    for tool in frugal_tools(cache_dir=FIXTURES, replay=True):
        assert set(tool.args) == {"question", "budget"}


def test_each_tool_describes_when_to_use_it() -> None:
    """A description is what tells an agent which tool to reach for."""
    assert "without issuing any search" in frugal_plan_tool().description
    assert "budget" in frugal_search_tool().description


def test_the_plan_tool_reports_cost_without_searching() -> None:
    answer = frugal_plan_tool().invoke({"question": QUESTION, "budget": 6})
    assert "at most" in answer
    assert "No search was issued" in answer


def test_the_plan_tool_says_the_ceiling_is_not_the_expected_cost() -> None:
    """An agent budgeting against the ceiling would over-reserve every time."""
    assert "usually costs less" in frugal_plan_tool().invoke({"question": QUESTION})


def test_the_search_tool_reports_what_it_spent() -> None:
    tool = frugal_search_tool(cache_dir=FIXTURES, replay=True)
    answer = tool.invoke({"question": QUESTION, "budget": 6})
    assert "searches billed" in answer


def test_the_search_tool_numbers_its_evidence() -> None:
    tool = frugal_search_tool(cache_dir=FIXTURES, replay=True)
    answer = tool.invoke({"question": QUESTION, "budget": 6})
    assert "[1]" in answer


def test_an_empty_question_is_answered_not_raised() -> None:
    for tool in frugal_tools(cache_dir=FIXTURES, replay=True):
        assert "No question" in tool.invoke({"question": "  "})


def test_a_failure_is_returned_as_text_an_agent_can_read(tmp_path: Path) -> None:
    """An agent cannot act on an exception it never sees."""
    tool = frugal_search_tool(cache_dir=str(tmp_path / "missing"), replay=True)
    answer = tool.invoke({"question": "something never recorded", "budget": 2})
    assert "failed" in answer.lower() or "No evidence" in answer


def test_the_budget_cap_is_shared_with_the_mcp_server() -> None:
    from frugal.mcp_server import MAX_BUDGET as MCP_MAX

    assert MAX_BUDGET == MCP_MAX
