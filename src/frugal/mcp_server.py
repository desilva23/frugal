"""An MCP server exposing Frugal to any agent that speaks the protocol.

Most search tools an agent can reach hand back ten links and say nothing about
what that cost. The agent cannot tell whether a question was cheap or expensive,
cannot cap what it is willing to spend, and cannot find out in advance. It just
searches, and someone gets the bill.

This server exposes the two things that change:

``frugal_plan`` answers "what would this cost?" without issuing anything. An
agent can ask before it commits, which is not usually an option.

``frugal_search`` runs the plan under a budget the caller sets, and returns what
it spent alongside what it found — which engines were used, how many searches
were billed, how many came free from cache, and why it stopped.

Both return structured results rather than a wall of text, because the caller is
a program. Evidence keeps its provenance, so an agent can cite the engine and
the query that produced a claim rather than asserting it.
"""

from __future__ import annotations

import os
from typing import Any

from frugal import __version__, _integration
from frugal.cache import CacheMode, ResponseCache
from frugal.client import SerpApiClient
from frugal.errors import FrugalError
from frugal.planner import DEFAULT_BUDGET, Planner, PlanResult
from frugal.schema import Document, Evidence, Series

#: Where recorded responses live. Overridable so an agent can be pointed at the
#: committed fixture set and run entirely offline.
CACHE_ENV_VAR = "FRUGAL_CACHE_DIR"
DEFAULT_CACHE_DIR = ".frugal-cache"

#: Set this to serve only from recorded responses, spending nothing. Useful for
#: demonstrating the server without a key.
REPLAY_ENV_VAR = "FRUGAL_REPLAY"

#: Shared with the framework adapters rather than defined again here. Two
#: definitions kept in step by a test catch a divergence after it happens; one
#: definition cannot diverge.
MAX_BUDGET = _integration.MAX_BUDGET

INSTRUCTIONS = """Frugal plans search across SerpApi's engines under a budget.

Call frugal_plan first when cost matters: it reports which engines a question
would reach and the most it could spend, without issuing anything.

Call frugal_search to run it. Set `budget` to the most searches you are willing
to spend; the plan stops early when the evidence stops improving and reports
what it actually cost.

Evidence carries provenance, so cite the engine and query that produced a claim
rather than asserting it."""


def _cache(replay: bool | None = None) -> ResponseCache:
    """The response cache, honouring the environment overrides."""
    if replay is None:
        replay = os.environ.get(REPLAY_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}
    directory = os.environ.get(CACHE_ENV_VAR, "").strip() or DEFAULT_CACHE_DIR
    return ResponseCache(directory, mode=CacheMode.REPLAY if replay else CacheMode.AUTO)


def _clamp(budget: int) -> int:
    """Bound a call by the per-call ceiling and by what the session has left."""
    return _integration.clamp(budget)


def _render(item: Evidence) -> dict[str, Any]:
    """One evidence item, with the provenance an agent needs to cite it."""
    common = {
        "engine": item.provenance.engine,
        "query": item.provenance.query,
        "rank": item.provenance.position,
    }
    if isinstance(item, Series):
        return {
            **common,
            "kind": "series",
            "name": item.name,
            "summary": item.summarise(),
            "observations": len(item.points),
            "change": item.trend(),
            "points": [
                {"timestamp": point.timestamp.isoformat(), "value": point.value}
                for point in item.points
            ],
        }
    assert isinstance(item, Document)
    return {
        **common,
        "kind": "document",
        "title": item.title,
        "url": item.url,
        "snippet": item.snippet,
        "source": item.source,
        "published": item.published_at.isoformat() if item.published_at else None,
    }


def _result(result: PlanResult) -> dict[str, Any]:
    """A completed plan, as a structured result.

    Cost is reported beside the evidence rather than hidden, because an agent
    that cannot see what a call cost cannot manage a budget across many.
    """
    return {
        "question": result.question,
        "evidence": [_render(item) for item in result.evidence],
        "cost": {
            "searches_billed": result.searches_charged,
            "served_from_cache": result.cache_hits,
            "steps_planned": result.ceiling,
            "steps_skipped_by_early_stop": result.steps_skipped,
            # What the whole process has spent. An agent that can only see one
            # call's cost cannot manage a budget across many.
            **_integration.ledger().as_dict(),
        },
        "engines_used": sorted({step.step.engine for step in result.steps}),
        "stopped_because": result.stopped_because,
        "failures": [
            {"engine": step.step.engine, "error": step.error} for step in result.failures
        ],
    }


def build_server() -> Any:
    """Construct the MCP server.

    Imported lazily so that the rest of Frugal does not depend on the MCP SDK.
    The server is an integration, not a core component, and someone using the
    library should not have to install a protocol they are not speaking.
    """
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise FrugalError(
            "the MCP server needs the protocol SDK:\n\n    pip install 'frugal[mcp]'\n"
        ) from exc

    server = MCPServer(
        name="frugal",
        title="Frugal — cost-aware search",
        instructions=INSTRUCTIONS,
        version=__version__,
    )

    @server.tool(
        name="frugal_plan",
        description=(
            "Report which SerpApi engines a question would reach and the most it "
            "could cost, without issuing any search. Use this before frugal_search "
            "when the budget matters."
        ),
    )
    def frugal_plan(question: str, budget: int = DEFAULT_BUDGET) -> dict[str, Any]:
        """Preview a plan's cost without spending anything."""
        if not question.strip():
            return {"error": "question is empty"}

        budget = _clamp(budget)
        client = SerpApiClient(cache=_cache(replay=True))
        planner = Planner(client)
        try:
            dry = planner.dry_run(question, budget=budget)
        except FrugalError as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        locale = planner.locale_for(question)
        return {
            "question": question,
            "engines": sorted({step.engine for step in dry.steps}),
            "steps": [
                {
                    "engine": step.engine,
                    "query": step.query,
                    "strategy": step.strategy,
                    "parameters": {
                        key: value for key, value in step.params.items() if key != "q"
                    },
                }
                for step in dry.steps
            ],
            "most_it_could_cost": dry.ceiling,
            "budget": budget,
            "fits_in_budget": dry.projection.fits,
            "locale": locale.describe() if locale else None,
            "note": (
                "This is the ceiling. A real run usually costs less: it stops early "
                "when the evidence stops improving, and repeated searches are free."
            ),
        }

    @server.tool(
        name="frugal_search",
        description=(
            "Search across SerpApi's engines under a budget, routing the question "
            "to the engines that can answer it. Returns evidence with provenance, "
            "plus what the call actually cost."
        ),
    )
    def frugal_search(question: str, budget: int = DEFAULT_BUDGET) -> dict[str, Any]:
        """Run a plan and return evidence with its cost."""
        if not question.strip():
            return {"error": "question is empty"}

        try:
            return _result(
                _integration.run_plan(
                    question,
                    budget=budget,
                    cache_dir=os.environ.get(CACHE_ENV_VAR, "").strip() or DEFAULT_CACHE_DIR,
                    replay=os.environ.get(REPLAY_ENV_VAR, "").strip().lower()
                    in {"1", "true", "yes"},
                )
            )
        except FrugalError as exc:
            # Returned rather than raised: a tool call that fails should tell the
            # agent what went wrong in a form it can read and act on.
            return {"error": f"{type(exc).__name__}: {exc}", "question": question}

    return server


def main() -> int:
    """Run the server over stdio, which is what MCP clients launch."""
    build_server().run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
