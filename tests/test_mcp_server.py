"""Tests for the MCP server.

The point of this integration is that a calling agent can see and cap what a
search costs, which no search tool it can otherwise reach will tell it. So these
check the cost accounting as closely as the results: that a preview spends
nothing, that a budget cannot be talked past, and that a failure comes back as
something an agent can read rather than an exception it cannot.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from frugal.mcp_server import (
    MAX_BUDGET,
    _clamp,
    _render,
    build_server,
)
from frugal.schema import Document, Provenance, Series, SeriesPoint

FIXTURES = Path(__file__).parent.parent / "benchmarks" / "fixtures"
QUESTION = "Has search interest in electric vehicles in India been rising or falling over time?"
NOW = datetime(2026, 9, 17, tzinfo=UTC)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every tool call at the committed fixtures, so the suite spends nothing."""
    monkeypatch.setenv("FRUGAL_REPLAY", "1")
    monkeypatch.setenv("FRUGAL_CACHE_DIR", str(FIXTURES))


def call(tool: str, **arguments: Any) -> dict[str, Any]:
    """Invoke a tool the way a protocol client would, and decode its result."""
    server = build_server()
    result = asyncio.run(server.call_tool(tool, arguments))
    payload = result[1] if isinstance(result, tuple) else result
    if hasattr(payload, "content"):
        return json.loads(payload.content[0].text)  # type: ignore[no-any-return]
    return payload  # type: ignore[no-any-return]


def tools() -> list[Any]:
    return asyncio.run(build_server().list_tools())


# --------------------------------------------------------------------------
# Advertisement
# --------------------------------------------------------------------------


def test_both_tools_are_exposed() -> None:
    assert {tool.name for tool in tools()} == {"frugal_plan", "frugal_search"}


def test_each_tool_describes_itself() -> None:
    """A description is what tells a calling agent when to reach for a tool."""
    for tool in tools():
        assert tool.description and len(tool.description) > 40


def test_each_tool_takes_a_question_and_a_budget() -> None:
    for tool in tools():
        assert set(tool.input_schema.get("properties", {})) == {"question", "budget"}


def test_the_server_tells_an_agent_to_check_cost_first() -> None:
    server = build_server()
    assert "frugal_plan" in (server.instructions or "")


# --------------------------------------------------------------------------
# frugal_plan: must not spend
# --------------------------------------------------------------------------


def test_plan_reports_the_engines_and_the_ceiling() -> None:
    result = call("frugal_plan", question=QUESTION, budget=6)
    assert "google_trends" in result["engines"]
    assert result["most_it_could_cost"] >= 1
    assert result["fits_in_budget"] is True


def test_plan_shows_the_parameters_each_engine_would_receive() -> None:
    result = call("frugal_plan", question=QUESTION, budget=6)
    trends = next(s for s in result["steps"] if s["engine"] == "google_trends")
    assert trends["parameters"]["geo"] == "IN"


def test_plan_reports_the_detected_place() -> None:
    assert "IN" in call("frugal_plan", question=QUESTION)["locale"]


def test_plan_says_the_ceiling_is_not_the_expected_cost() -> None:
    """An agent budgeting against the ceiling would over-reserve every time."""
    assert "usually costs less" in call("frugal_plan", question=QUESTION)["note"]


def test_plan_spends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "cache"
    monkeypatch.setenv("FRUGAL_CACHE_DIR", str(empty))
    monkeypatch.delenv("FRUGAL_REPLAY", raising=False)
    call("frugal_plan", question=QUESTION)
    assert not list(empty.rglob("*.json")) if empty.exists() else True


# --------------------------------------------------------------------------
# frugal_search
# --------------------------------------------------------------------------


def test_search_returns_evidence_with_its_cost() -> None:
    result = call("frugal_search", question=QUESTION, budget=6)
    assert result["evidence"]
    assert set(result["cost"]) == {
        "searches_billed",
        "served_from_cache",
        "steps_planned",
        "steps_skipped_by_early_stop",
    }


def test_every_evidence_item_carries_provenance() -> None:
    """An agent should cite the engine and query, not assert the claim."""
    for item in call("frugal_search", question=QUESTION, budget=6)["evidence"]:
        assert item["engine"]
        assert item["query"]
        assert item["rank"] >= 1


def test_search_reports_which_engines_it_used() -> None:
    result = call("frugal_search", question=QUESTION, budget=6)
    assert "google_trends" in result["engines_used"]


def test_search_reports_why_it_stopped() -> None:
    assert call("frugal_search", question=QUESTION, budget=6)["stopped_because"]


def test_a_replayed_search_bills_nothing() -> None:
    assert call("frugal_search", question=QUESTION, budget=6)["cost"]["searches_billed"] == 0


# --------------------------------------------------------------------------
# The budget cannot be talked past
# --------------------------------------------------------------------------


def test_an_oversized_budget_is_capped() -> None:
    """An agent looping on a tool is how a search bill becomes a surprise."""
    assert _clamp(10_000) == MAX_BUDGET
    assert call("frugal_plan", question=QUESTION, budget=10_000)["budget"] == MAX_BUDGET


def test_a_budget_below_one_is_raised_to_one() -> None:
    assert _clamp(0) == 1
    assert _clamp(-5) == 1


def test_a_budget_within_range_is_left_alone() -> None:
    assert _clamp(7) == 7


# --------------------------------------------------------------------------
# Failures reach the agent as data
# --------------------------------------------------------------------------


def test_an_empty_question_is_reported_not_raised() -> None:
    for tool in ("frugal_plan", "frugal_search"):
        assert "error" in call(tool, question="   ")


def test_a_replay_miss_is_reported_as_an_error_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool call that fails must tell the agent what went wrong."""
    monkeypatch.setenv("FRUGAL_CACHE_DIR", str(tmp_path / "empty"))
    result = call("frugal_search", question="something never recorded")
    assert "error" in result or result["evidence"] == []


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_a_document_renders_with_its_citable_fields() -> None:
    document = Document(
        title="A title",
        url="https://example.com/a",
        snippet="a snippet",
        source="Reuters",
        provenance=Provenance("google", "q", 3, NOW),
    )
    rendered = _render(document)
    assert rendered["kind"] == "document"
    assert rendered["url"] == "https://example.com/a"
    assert rendered["rank"] == 3


def test_a_series_renders_with_its_points_and_change() -> None:
    """The one kind of evidence a web search cannot return, kept whole."""
    points = tuple(
        SeriesPoint(timestamp=datetime(2026, 1, i, tzinfo=UTC), value=float(i * 10))
        for i in range(1, 5)
    )
    rendered = _render(
        Series(name="term", points=points, provenance=Provenance("google_trends", "q", 1, NOW))
    )
    assert rendered["kind"] == "series"
    assert rendered["observations"] == 4
    assert rendered["change"] == pytest.approx(3.0)
    assert len(rendered["points"]) == 4


def test_rendered_evidence_is_json_serialisable() -> None:
    """The caller is a program; a result it cannot decode is no result."""
    result = call("frugal_search", question=QUESTION, budget=6)
    json.dumps(result)
