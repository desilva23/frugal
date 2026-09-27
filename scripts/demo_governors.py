"""Show the budget and the stopping rule firing.

Neither fires anywhere in the benchmark, and the README says so. That is a
property of the question set, not of the code: the budget is twelve and a plan
spends two, and marginal novelty never falls because each step queries a
different index. Both facts are easy to state and easy to doubt, so this
reproduces each mechanism doing its job.

No API key and no network: a mock transport serves the responses, and it
charges, which matters. Replay cannot demonstrate a budget -- it serves from
cache, a cache hit is free, and an allowance bounding money is not consumed by
a search that cost none.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import httpx

from frugal.cache import ResponseCache
from frugal.client import SerpApiClient
from frugal.planner import Planner

QUESTION = "What did recent studies find about retrieval augmented generation?"
TREND = "Is interest in electric vehicles and solar panels growing in India over time?"


def _results(rows: list[dict[str, str]]) -> httpx.Response:
    return httpx.Response(
        200, json={"organic_results": rows, "search_metadata": {"status": "Success"}}
    )


def varied(request: httpx.Request) -> httpx.Response:
    """Every engine returns different pages, so nothing ever saturates."""
    engine = request.url.params.get("engine", "google")
    query = request.url.params.get("q", "")
    return _results(
        [
            {
                "title": f"{engine} {query[:10]} {i}",
                "link": f"https://{engine}.test/{query[:6]}/{i}",
                "snippet": "s",
            }
            for i in range(5)
        ]
    )


def identical(request: httpx.Request) -> httpx.Response:
    """Every engine returns the same page, so the second search buys nothing."""
    return _results(
        [{"title": f"same {i}", "link": f"https://same.test/{i}", "snippet": "s"}
         for i in range(5)]
    )


def _planner(handler: Any, directory: str, **kwargs: Any) -> tuple[Planner, SerpApiClient]:
    client = SerpApiClient(
        api_key="demo-key",
        cache=ResponseCache(Path(directory) / "cache"),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _seconds: None,
    )
    return Planner(client, **kwargs), client


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        planner, client = _planner(varied, directory, max_engines=3, max_rounds=2)
        ceiling = planner.dry_run(QUESTION, budget=50).ceiling
        result = planner.run(QUESTION, budget=2)
        print("The budget refuses the search that would exceed it")
        print(f"  plan wants     : {ceiling} searches")
        print("  budget given   : 2")
        print(f"  issued         : {result.steps_executed}"
              f"        skipped: {result.steps_skipped}")
        print(f"  SerpApi billed : {client.log.searches_charged}")
        print(f"  stopped        : {result.stopped_because}")
        assert result.stopped_because == "budget exhausted"
        assert client.log.searches_charged == 2

    print()

    with tempfile.TemporaryDirectory() as directory:
        planner, client = _planner(identical, directory, max_engines=1, max_rounds=4)
        ceiling = planner.dry_run(TREND, budget=50).ceiling
        result = planner.run(TREND, budget=50)
        novelty = ", ".join(f"{o.marginal_novelty:.2f}" for o in result.observations)
        print("The stopping rule stops when the next search would buy nothing")
        print(f"  plan wants     : {ceiling} searches")
        print("  budget given   : 50        (deliberately not binding)")
        print(f"  issued         : {result.steps_executed}"
              f"        skipped: {result.steps_skipped}")
        print(f"  stopped        : {result.stopped_because}")
        print(f"  novelty        : [{novelty}]")
        assert "saturated" in result.stopped_because
        assert result.steps_skipped > 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
