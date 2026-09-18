"""Tests for plan compilation and execution.

Driven through httpx.MockTransport so the whole system can be exercised —
budget exhaustion, early stopping, a failing engine mid-round — without a key or
a network. Controlling exactly what each engine returns is also the only way to
test the stopping rule, which depends on how much a batch overlaps with the last.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from frugal.cache import ResponseCache
from frugal.client import SerpApiClient
from frugal.planner import Planner, keyword_naive_run, naive_run, parameterised_naive_run
from frugal.schema import Document


def results(*names: str) -> list[dict[str, Any]]:
    return [
        {"title": n, "link": f"https://example.com/{n}", "snippet": f"about {n}", "position": i + 1}
        for i, n in enumerate(names)
    ]


def responder(
    payloads: dict[str, list[dict[str, Any]]] | None = None,
    *,
    default: list[dict[str, Any]] | None = None,
    fail_engine: str | None = None,
    counter: list[str] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Serve per-engine results, optionally failing one engine."""

    def handler(request: httpx.Request) -> httpx.Response:
        engine = request.url.params.get("engine", "google")
        if counter is not None:
            counter.append(engine)
        if engine == fail_engine:
            return httpx.Response(200, json={"error": f"{engine} is unavailable"})
        if engine == "google_trends":
            return httpx.Response(
                200,
                json={
                    "search_metadata": {"id": "x", "status": "Success"},
                    "interest_over_time": {
                        "timeline_data": [
                            {"timestamp": "1757808000", "values": [
                                {"query": "t", "extracted_value": 5}
                            ]}
                        ]
                    },
                },
            )
        rows = (payloads or {}).get(engine, default if default is not None else results("a", "b"))
        key = {
            "google_news": "news_results",
            "google_jobs": "jobs_results",
            "google_maps": "local_results",
        }.get(engine, "organic_results")
        return httpx.Response(
            200, json={"search_metadata": {"id": "x", "status": "Success"}, key: rows}
        )

    return handler


def make_planner(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
    **kwargs: Any,
) -> tuple[Planner, SerpApiClient]:
    client = SerpApiClient(
        api_key="test-key",
        cache=ResponseCache(tmp_path / "cache"),
        http=httpx.Client(transport=httpx.MockTransport(handler or responder())),
        sleep=lambda _s: None,
    )
    return Planner(client, **kwargs), client


QUESTION = "What did recent studies find about retrieval augmented generation?"


# --------------------------------------------------------------------------
# Planning costs nothing
# --------------------------------------------------------------------------


def test_planning_issues_no_requests(tmp_path: Path) -> None:
    calls: list[str] = []
    planner, _ = make_planner(tmp_path, responder(counter=calls))
    planner.plan(QUESTION)
    planner.dry_run(QUESTION, budget=12)
    assert calls == []


def test_a_plan_is_deterministic(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path)
    first = [(s.engine, s.query) for s in planner.plan(QUESTION)]
    for _ in range(10):
        assert [(s.engine, s.query) for s in planner.plan(QUESTION)] == first


def test_a_plan_respects_the_engine_limit(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path, max_engines=2)
    assert len({s.engine for s in planner.plan(QUESTION)}) <= 2


def test_a_plan_respects_the_round_limit(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path, max_rounds=1)
    assert {s.round_number for s in planner.plan(QUESTION)} == {1}


def test_a_plan_stops_when_reformulation_runs_dry(tmp_path: Path) -> None:
    """More rounds would only reissue queries the plan has already made."""
    planner, _ = make_planner(tmp_path, max_rounds=20)
    steps = planner.plan(QUESTION)
    assert max(s.round_number for s in steps) < 20


def test_a_plan_never_repeats_a_query_on_one_engine(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path, max_rounds=6)
    issued = [(s.engine, s.query) for s in planner.plan(QUESTION)]
    assert len(issued) == len(set(issued))


def test_dry_run_reports_the_ceiling(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path)
    dry = planner.dry_run(QUESTION, budget=12)
    assert dry.ceiling == len(dry.steps)
    assert dry.projection.fits
    assert "at most" in dry.describe()


def test_dry_run_flags_a_plan_that_will_not_fit(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path)
    assert not planner.dry_run(QUESTION, budget=1).projection.fits


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def test_a_run_returns_normalised_evidence(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path)
    result = planner.run(QUESTION, budget=12)
    assert result.evidence
    assert all(isinstance(item, Document) for item in result.evidence)


def test_evidence_is_deduplicated_across_engines(tmp_path: Path) -> None:
    """Every engine returning the same page is one piece of evidence."""
    planner, _ = make_planner(tmp_path, responder(default=results("same")))
    assert len(planner.run(QUESTION, budget=12).evidence) == 1


def test_results_are_truncated_to_a_common_depth(tmp_path: Path) -> None:
    """Without this, an engine with a bigger page looks like a better plan."""
    wide = responder(default=results(*[f"r{i}" for i in range(100)]))
    planner, _ = make_planner(tmp_path, wide, results_per_search=5)
    assert all(step.evidence_count <= 5 for step in planner.run(QUESTION, budget=12).steps)


def test_step_order_is_the_plan_order_despite_concurrency(tmp_path: Path) -> None:
    """A run must reproduce regardless of which engine answers first."""
    planner, _ = make_planner(tmp_path, concurrency=4)
    first = [s.step.query for s in planner.run(QUESTION, budget=12).steps]
    for _ in range(5):
        planner2, _ = make_planner(tmp_path, concurrency=4)
        assert [s.step.query for s in planner2.run(QUESTION, budget=12).steps] == first


def test_an_empty_question_is_rejected(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path)
    with pytest.raises(ValueError, match="empty question"):
        planner.run("  ", budget=5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_engines": 0}, "max_engines"),
        ({"max_rounds": 0}, "max_rounds"),
        ({"concurrency": 0}, "concurrency"),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path, kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_planner(tmp_path, **kwargs)


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_a_run_never_bills_more_than_its_budget(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path)
    result = planner.run(QUESTION, budget=2)
    assert result.searches_charged <= 2
    assert result.budget.spent <= 2


def test_an_exhausted_budget_stops_the_plan(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path)
    result = planner.run(QUESTION, budget=1)
    assert "budget" in result.stopped_because
    assert result.budget.was_binding


def test_cache_hits_do_not_consume_the_budget(tmp_path: Path) -> None:
    """The budget is a spend limit, not a call limit; a cached search is free."""
    planner, client = make_planner(tmp_path)
    planner.run(QUESTION, budget=12)
    billed_first = client.log.searches_charged

    second = planner.run(QUESTION, budget=1)
    assert second.cache_hits == second.steps_executed
    assert second.searches_charged == 0
    assert client.log.searches_charged == billed_first


def test_spend_is_reported_per_engine(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path)
    assert planner.run(QUESTION, budget=12).budget.by_engine


def test_the_listener_sees_each_billed_search(tmp_path: Path) -> None:
    seen: list[int] = []
    planner, _ = make_planner(tmp_path)
    result = planner.run(
        QUESTION, budget=12, listener=lambda *, spent, limit, engine: seen.append(spent)
    )
    assert seen == list(range(1, result.searches_charged + 1))


# --------------------------------------------------------------------------
# The two savings, kept apart
# --------------------------------------------------------------------------


def test_skipped_steps_measure_early_stopping_not_caching(tmp_path: Path) -> None:
    """Conflating them would report the cache's work as the planner's."""
    planner, _ = make_planner(tmp_path)
    warm = planner.run(QUESTION, budget=12)
    again = planner.run(QUESTION, budget=12)
    assert again.cache_hits == again.steps_executed
    assert again.steps_skipped == warm.steps_skipped


def test_a_saturating_plan_stops_before_its_last_round(tmp_path: Path) -> None:
    """Every engine returning the same page means there is nothing left to buy.

    Rounds are explicit here: the measured default is a single round, in which
    there is no second batch for saturation to observe.
    """
    planner, _ = make_planner(tmp_path, responder(default=results("a", "b")), max_rounds=2)
    result = planner.run(QUESTION, budget=50)
    assert "saturated" in result.stopped_because


def test_a_saturating_plan_skips_its_remaining_steps(tmp_path: Path) -> None:
    """Skipping needs a plan with a round left to skip.

    Reformulation exhausts most engines after two distinct queries, so a typical
    plan saturates on its final round and has nothing left to skip. Trends is
    the exception -- it accepts only term queries, and narrower term sets give a
    third round -- which makes it the case where the saving is visible.
    """
    planner, _ = make_planner(tmp_path, max_engines=1, max_rounds=4)
    result = planner.run(
        "Is interest in electric vehicles growing in India over time?", budget=50
    )
    assert "saturated" in result.stopped_because
    assert result.steps_skipped > 0
    assert result.steps_executed < result.ceiling


def test_a_plan_still_finding_evidence_does_not_stop(tmp_path: Path) -> None:
    counter: list[str] = []

    def varied(request: httpx.Request) -> httpx.Response:
        counter.append("x")
        rows = results(*[f"r{len(counter)}_{i}" for i in range(5)])
        return httpx.Response(
            200,
            json={"search_metadata": {"id": "x", "status": "Success"}, "organic_results": rows},
        )

    planner, _ = make_planner(tmp_path, varied)
    result = planner.run(QUESTION, budget=50)
    assert result.steps_skipped == 0
    assert not result.saturation.saturated


# --------------------------------------------------------------------------
# A failing engine must not take the plan down
# --------------------------------------------------------------------------


def test_one_failing_engine_does_not_abort_the_plan(tmp_path: Path) -> None:
    """Scholar is routed first for this question, so failing it is felt."""
    planner, _ = make_planner(tmp_path, responder(fail_engine="google_scholar"))
    result = planner.run(QUESTION, budget=12)
    assert result.failures
    assert all(f.step.engine == "google_scholar" for f in result.failures)
    assert result.evidence


def test_a_failed_search_is_not_billed(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path, responder(fail_engine="google_scholar"))
    result = planner.run(QUESTION, budget=12)
    assert all(f.charged == 0 for f in result.failures)


def test_schema_drift_on_one_engine_is_reported_not_swallowed(tmp_path: Path) -> None:
    """Returning nothing silently would look like an engine with thin coverage."""

    def drifted(request: httpx.Request) -> httpx.Response:
        engine = request.url.params.get("engine")
        if engine == "google_scholar":
            return httpx.Response(
                200,
                json={"search_metadata": {"id": "x"}, "renamed_results": results("a")},
            )
        return responder()(request)

    planner, _ = make_planner(tmp_path, drifted)
    result = planner.run(QUESTION, budget=12)
    drift = [f for f in result.failures if f.step.engine == "google_scholar"]
    assert drift and "SchemaDrift" in str(drift[0].error)


def test_a_drifted_search_is_still_counted_as_billed(tmp_path: Path) -> None:
    """SerpApi charged for it; pretending otherwise would understate the cost."""

    def drifted(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"search_metadata": {"id": "x"}, "renamed": results("a")}
        )

    planner, _ = make_planner(tmp_path, drifted)
    result = planner.run(QUESTION, budget=12)
    assert result.searches_charged > 0


# --------------------------------------------------------------------------
# The baseline
# --------------------------------------------------------------------------


def test_the_baseline_issues_exactly_one_search(tmp_path: Path) -> None:
    calls: list[str] = []
    _, client = make_planner(tmp_path, responder(counter=calls))
    result = naive_run(client, QUESTION)
    assert len(calls) == 1
    assert result.searches_charged == 1
    assert result.ceiling == 1


def test_the_baseline_uses_web_search_verbatim(tmp_path: Path) -> None:
    _, client = make_planner(tmp_path)
    step = naive_run(client, QUESTION).steps[0]
    assert step.step.engine == "google"
    assert step.step.query == QUESTION


def test_the_baseline_is_truncated_to_the_same_depth(tmp_path: Path) -> None:
    """A baseline allowed a longer page would be a rigged comparison."""
    wide = responder(default=results(*[f"r{i}" for i in range(100)]))
    _, client = make_planner(tmp_path, wide)
    assert len(naive_run(client, QUESTION, results=5).evidence) == 5


def test_the_baseline_survives_a_failing_engine(tmp_path: Path) -> None:
    _, client = make_planner(tmp_path, responder(fail_engine="google"))
    result = naive_run(client, QUESTION)
    assert result.steps[0].error
    assert result.evidence == []


def test_results_serialise_for_the_benchmark(tmp_path: Path) -> None:
    planner, _ = make_planner(tmp_path)
    payload = planner.run(QUESTION, budget=12).as_dict()
    for key in ("searches_charged", "ceiling", "steps_skipped", "cache_hits", "budget"):
        assert key in payload


# --------------------------------------------------------------------------
# Later rounds adapt to what earlier ones found
# --------------------------------------------------------------------------


def test_a_second_round_asks_something_the_first_round_taught_it(tmp_path: Path) -> None:
    """The answer to "this is a static DAG, not an agent".

    Round two is not a rewording of the question; it carries vocabulary that only
    appeared in round one's results.
    """
    seen: list[str] = []
    rows = results("quantum annealing breakthrough", "quantum annealing result")
    inner = responder(default=rows)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url.params.get("q", "")))
        return inner(request)

    planner, _ = make_planner(tmp_path, handler, max_rounds=2, max_engines=1)
    planner.run("What was the latest computing breakthrough?", budget=12)

    assert any("quantum" in q or "annealing" in q for q in seen[1:]), seen


def test_adaptation_is_deterministic(tmp_path: Path) -> None:
    """A plan that adapts must still reproduce, or the benchmark is worthless."""

    handler = responder(default=results("photonic lattice study", "photonic lattice review"))
    question = "What was the latest computing breakthrough?"
    runs = []
    for _ in range(3):
        planner, _ = make_planner(tmp_path, handler, max_rounds=2, max_engines=1)
        runs.append([s.step.query for s in planner.run(question, budget=12).steps])
    assert runs[0] == runs[1] == runs[2]


def test_a_term_only_engine_is_not_given_feedback_terms(tmp_path: Path) -> None:
    """Appending a found term to a trends query invents a term nobody searched."""
    planner, _ = make_planner(tmp_path, max_rounds=3, max_engines=1)
    result = planner.run("Is interest in electric vehicles growing in India over time?", budget=20)
    for step in result.steps:
        if step.step.engine == "google_trends":
            assert "feedback" not in step.step.strategy


def test_a_round_with_nothing_new_to_add_keeps_its_query(tmp_path: Path) -> None:
    """No feedback terms means the planned query stands, not an empty one."""

    planner, _ = make_planner(
        tmp_path, responder(default=[]), max_rounds=2, max_engines=1
    )
    result = planner.run("What was the latest computing breakthrough?", budget=12)
    assert all(step.step.query for step in result.steps)


# --------------------------------------------------------------------------
# The parameterised baseline
# --------------------------------------------------------------------------
#
# It exists to separate two mechanisms that were being credited to one. Without
# it the gap between a keyword search and a full plan contains both routing and
# the engine parameters a plan carries, and the README attributed all of it to
# routing.


def test_the_parameterised_baseline_issues_exactly_one_search(tmp_path: Path) -> None:
    calls: list[str] = []
    _, client = make_planner(tmp_path, responder(counter=calls))
    result = parameterised_naive_run(client, QUESTION)
    assert len(calls) == 1
    assert result.ceiling == 1


def test_it_uses_web_search_only(tmp_path: Path) -> None:
    """One engine. The only thing it adds over the keyword baseline is parameters."""
    _, client = make_planner(tmp_path)
    assert parameterised_naive_run(client, QUESTION).steps[0].step.engine == "google"


def test_it_reformulates_like_the_keyword_baseline(tmp_path: Path) -> None:
    _, client = make_planner(tmp_path)
    plain = keyword_naive_run(client, QUESTION).steps[0].step.query
    assert parameterised_naive_run(client, QUESTION).steps[0].step.query == plain


def test_it_receives_the_parameters_the_keyword_baseline_does_not(tmp_path: Path) -> None:
    """The one difference between the two arms, and therefore what they measure."""
    question = "Which companies are hiring Python developers in Chennai?"
    _, client = make_planner(tmp_path)

    plain = keyword_naive_run(client, question).steps[0].step.params
    parameterised = parameterised_naive_run(client, question).steps[0].step.params

    assert set(plain) == {"q", "num"}
    assert parameterised["gl"] == "in"
    assert parameterised["hl"] == "en"


def test_it_narrows_scholar_style_recency_when_asked(tmp_path: Path) -> None:
    _, client = make_planner(tmp_path)
    result = parameterised_naive_run(client, "What recent research addresses hallucination?")
    assert result.steps[0].step.params["q"]


def test_a_failing_search_does_not_take_it_down(tmp_path: Path) -> None:
    _, client = make_planner(tmp_path, responder(fail_engine="google"))
    result = parameterised_naive_run(client, QUESTION)
    assert result.steps[0].error
    assert result.evidence == []


def test_an_empty_search_spends_the_budget_it_really_spent(tmp_path: Path) -> None:
    """The budget must see what SerpApi billed, including searches that found nothing."""

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "search_metadata": {"id": "e", "status": "Success"},
                "error": "Google hasn't returned any results for this query.",
            },
        )

    planner, client = make_planner(tmp_path, empty)
    result = planner.run(QUESTION, budget=12)

    assert result.searches_charged == result.steps_executed
    assert result.budget.spent == result.steps_executed
    assert client.log.empty_results == result.steps_executed
    assert result.failures == []
