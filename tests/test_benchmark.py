"""Tests for benchmark scoring.

Scoring decides what the published table says, so the properties that matter are
that a correct answer phrased differently still counts, that a near miss is not
recorded as a hit, and that the whole thing is exact rather than judged.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from frugal.benchmark import (
    Question,
    StrategySummary,
    load_questions,
    score_evidence,
    searchable_text,
)
from frugal.schema import Document, Provenance, Series, SeriesPoint

NOW = datetime(2026, 9, 17, tzinfo=UTC)


def doc(title: str, **kwargs: object) -> Document:
    return Document(
        title=title,
        provenance=Provenance("google", "q", 1, NOW),
        **kwargs,  # type: ignore[arg-type]
    )


def question(markers: list[list[str]], *, expects_series: bool = False) -> Question:
    return Question(
        id="q",
        question="?",
        category="test",
        markers=tuple(tuple(g) for g in markers),
        rationale="",
        expects_series=expects_series,
    )


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_an_answer_in_the_title_counts() -> None:
    assert score_evidence(question([["bengaluru"]]), [doc("Bengaluru is the capital")]).found


def test_an_answer_in_the_snippet_counts() -> None:
    evidence = [doc("Karnataka", snippet="The capital is Bengaluru.")]
    assert score_evidence(question([["bengaluru"]]), evidence).found


def test_an_answer_in_the_url_counts() -> None:
    evidence = [doc("A page", url="https://example.com/bengaluru")]
    assert score_evidence(question([["bengaluru"]]), evidence).found


def test_any_alternative_satisfies_a_group() -> None:
    """A correct answer phrased differently must still count."""
    group = question([["bengaluru", "bangalore"]])
    assert score_evidence(group, [doc("Bangalore")]).found
    assert score_evidence(group, [doc("Bengaluru")]).found


def test_every_group_must_be_satisfied() -> None:
    q = question([["python"], ["chennai"]])
    assert not score_evidence(q, [doc("Python jobs in Mumbai")]).found
    assert score_evidence(q, [doc("Python jobs in Chennai")]).found


def test_groups_may_be_satisfied_by_different_results() -> None:
    q = question([["python"], ["chennai"]])
    assert score_evidence(q, [doc("Python roles"), doc("Openings in Chennai")]).found


def test_matching_is_case_insensitive() -> None:
    assert score_evidence(question([["bengaluru"]]), [doc("BENGALURU")]).found


def test_no_evidence_finds_nothing() -> None:
    score = score_evidence(question([["anything"]]), [])
    assert not score.found
    assert score.depth is None


def test_unmatched_groups_are_reported() -> None:
    """A missing marker should say which, so a miss can be diagnosed."""
    score = score_evidence(question([["python"], ["chennai"]]), [doc("Python")])
    assert score.missing == ("chennai",)
    assert score.completeness == 0.5


# --------------------------------------------------------------------------
# Depth
# --------------------------------------------------------------------------


def test_depth_is_where_the_answer_completed() -> None:
    q = question([["alpha"], ["beta"]])
    evidence = [doc("alpha"), doc("nothing"), doc("beta"), doc("more")]
    assert score_evidence(q, evidence).depth == 3


def test_depth_is_one_when_a_single_result_answers_it() -> None:
    assert score_evidence(question([["alpha"]]), [doc("alpha")]).depth == 1


def test_depth_is_none_when_the_answer_never_completes() -> None:
    q = question([["alpha"], ["absent"]])
    assert score_evidence(q, [doc("alpha")]).depth is None


# --------------------------------------------------------------------------
# Series evidence
# --------------------------------------------------------------------------


def series(name: str) -> Series:
    points = tuple(
        SeriesPoint(timestamp=datetime(2026, 1, i + 1, tzinfo=UTC), value=float(i))
        for i in range(3)
    )
    return Series(name=name, points=points, provenance=Provenance("google_trends", "q", 1, NOW))


def test_prose_can_answer_a_question_a_series_answers_better() -> None:
    """The correction to an engineered metric.

    Requiring a Series to count made it arithmetically impossible for a
    web-search baseline to score on the trend question, since web search returns
    documents and never a series. Its results plainly did answer it, carrying
    headlines like "Why India is Seeing EV Interest Rise".
    """
    q = question([["electric vehicles"]], expects_series=True)
    score = score_evidence(q, [doc("electric vehicles interest is rising")])
    assert score.found
    assert not score.structured


def test_a_series_satisfies_a_question_that_needs_one() -> None:
    q = question([["electric vehicles"]], expects_series=True)
    score = score_evidence(q, [series("electric vehicles")])
    assert score.found
    assert score.structured


def test_a_series_contributes_its_summary_text() -> None:
    assert "observations" in searchable_text(series("term"))


def test_which_kind_of_evidence_came_back_is_recorded_but_not_scored() -> None:
    """A score implies a measurement; this is a description of the engines used.

    Both prose and a series answer the question, and both count as answered.
    Whether a series came back is recorded because it is worth knowing, not
    because it distinguishes one strategy's performance from another's.
    """
    q = question([["electric vehicles"]], expects_series=True)
    prose = score_evidence(q, [doc("electric vehicles")])
    structured = score_evidence(q, [series("electric vehicles")])

    assert prose.found and structured.found
    assert not prose.structured
    assert structured.structured


def test_the_scored_modality_metric_is_gone() -> None:
    """It reported which engines a configuration calls, not how it performed.

    Only google_trends emits a Series and only the routed strategy calls it, so
    the figure was thirty minus the count of expects_series questions for the
    baselines and thirty for the planner -- known before running anything.
    """
    score = score_evidence(question([["x"]], expects_series=True), [doc("x")])
    assert not hasattr(score, "answered_in_the_right_modality")


# --------------------------------------------------------------------------
# The shipped question set
# --------------------------------------------------------------------------


def test_the_question_set_loads() -> None:
    questions = load_questions()
    assert len(questions) >= 12


def test_every_question_has_markers_and_a_rationale() -> None:
    for q in load_questions():
        assert q.markers, f"{q.id} has no markers"
        assert all(group for group in q.markers), f"{q.id} has an empty marker group"
        assert q.rationale, f"{q.id} has no stated rationale"


def test_question_ids_are_unique() -> None:
    ids = [q.id for q in load_questions()]
    assert len(set(ids)) == len(ids)


def test_the_set_includes_controls_the_baseline_should_answer() -> None:
    """A set the planner wins outright would be a set chosen to make it win."""
    assert sum(1 for q in load_questions() if q.category == "simple") >= 3


def test_markers_are_lowercased_on_load() -> None:
    for q in load_questions():
        for group in q.markers:
            assert all(alt == alt.lower() for alt in group)


def test_a_malformed_question_set_fails_loudly(tmp_path: Path) -> None:
    bad = tmp_path / "questions.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception, match="not valid JSON"):
        load_questions(bad)


def test_a_missing_question_set_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="no question set"):
        load_questions(tmp_path / "absent.json")


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def test_recall_and_efficiency_are_reported() -> None:
    summary = StrategySummary(strategy="x", questions=4, answered=3, searches=8)
    assert summary.recall == 0.75
    assert summary.searches_per_question == 2.0
    assert summary.answers_per_search == 0.375


def test_an_empty_summary_does_not_divide_by_zero() -> None:
    summary = StrategySummary(strategy="x", questions=0, answered=0, searches=0)
    assert summary.recall == 0.0
    assert summary.answers_per_search == 0.0
    assert summary.median_depth is None


def test_median_depth_handles_an_even_count() -> None:
    summary = StrategySummary(strategy="x", questions=4, answered=4, searches=4)
    summary.depths = [1, 2, 3, 4]
    assert summary.median_depth == 2.5


# --------------------------------------------------------------------------
# The reporting path
# --------------------------------------------------------------------------
#
# These tables are the artifact the project is judged on. The scoring underneath
# them was tested from the start; the code that aggregates and renders them was
# not, which left the published numbers resting on the least-covered module in
# the repository.

FIXTURES = str(Path(__file__).parent.parent / "benchmarks" / "fixtures")


def run_two_questions() -> list[object]:
    """Run the real harness over two questions, from committed fixtures."""
    from frugal.benchmark import run_benchmark
    from frugal.cache import CacheMode, ResponseCache
    from frugal.client import SerpApiClient

    questions = load_questions()[:2]
    cache = ResponseCache(FIXTURES, mode=CacheMode.REPLAY)
    with SerpApiClient(cache=cache) as client:
        return run_benchmark(client, questions, budget=12, verbose=False)  # type: ignore[return-value]


def test_the_harness_runs_every_strategy_over_every_question() -> None:
    from frugal.benchmark import STRATEGIES

    outcomes = run_two_questions()
    assert len(outcomes) == 2 * len(STRATEGIES)
    assert {o.strategy for o in outcomes} == set(STRATEGIES)  # type: ignore[attr-defined]


def test_the_harness_spends_nothing_replaying_fixtures() -> None:
    """If this ever bills, the benchmark is not reproducible as claimed."""
    for outcome in run_two_questions():
        assert outcome.billed_this_run == 0  # type: ignore[attr-defined]


def test_cost_is_reported_as_steps_not_as_billing() -> None:
    """Billing depends on cache warmth; a replayed run would read as free."""
    for outcome in run_two_questions():
        assert outcome.searches >= 1  # type: ignore[attr-defined]


def test_the_headline_table_names_every_strategy() -> None:
    from frugal.benchmark import STRATEGIES, render_table, summarise

    table = render_table(summarise(run_two_questions()))  # type: ignore[arg-type]
    for strategy in STRATEGIES:
        assert strategy in table


def test_the_headline_table_is_markdown_a_readme_can_carry() -> None:
    from frugal.benchmark import render_table, summarise

    lines = render_table(summarise(run_two_questions())).splitlines()  # type: ignore[arg-type]
    assert lines[0].startswith("|") and lines[0].endswith("|")
    assert set(lines[1].replace("|", "").strip()) <= {"-", " "}
    assert all(line.count("|") == lines[0].count("|") for line in lines)


def test_the_table_reports_recall_as_a_percentage() -> None:
    from frugal.benchmark import render_table, summarise

    assert "%" in render_table(summarise(run_two_questions()))  # type: ignore[arg-type]


def test_a_strategy_that_produced_nothing_is_omitted_not_shown_as_zero() -> None:
    """A row of zeros would read as a measured result rather than an absent one."""
    from frugal.benchmark import render_table

    assert "planned" not in render_table({})


def test_the_ablation_table_lists_the_configurations_it_ran() -> None:
    from frugal.benchmark import StrategySummary, render_ablation

    summaries = {
        "naive": StrategySummary(strategy="naive", questions=2, answered=1, searches=2),
        "routed-2x1": StrategySummary(strategy="routed-2x1", questions=2, answered=2, searches=3),
    }
    table = render_ablation(summaries)
    assert "naive" in table and "routed-2x1" in table
    assert "fixed-news" not in table


def test_the_deep_sweeps_can_be_skipped() -> None:
    """They cost more than the benchmark they check at this question count."""
    from frugal.benchmark import ABLATIONS, DEEP_ABLATIONS

    assert {name for name, _, _ in ABLATIONS} > DEEP_ABLATIONS


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def test_the_dry_run_issues_nothing_and_reports_a_ceiling(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from frugal.benchmark import main

    assert main(["--dry-run", "--limit", "2", "--cache", FIXTURES]) == 0
    out = capsys.readouterr().out
    assert "ceiling for the full run" in out
    assert "worst case" in out


def test_a_replayed_run_writes_its_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from frugal.benchmark import main

    out_path = tmp_path / "results.json"
    assert main(["--replay", "--cache", FIXTURES, "--limit", "2", "--out", str(out_path)]) == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["questions"] == 2
    assert payload["searches_billed_this_run"] == 0
    assert payload["summaries"]
    assert payload["outcomes"]


def test_the_written_results_carry_what_the_readme_quotes(tmp_path: Path) -> None:
    """A reader checking the table should find the same fields behind it."""
    from frugal.benchmark import main

    out_path = tmp_path / "results.json"
    main(["--replay", "--cache", FIXTURES, "--limit", "2", "--out", str(out_path)])
    summary = next(iter(json.loads(out_path.read_text(encoding="utf-8"))["summaries"].values()))
    for field in ("recall", "searches_per_question", "answered", "questions"):
        assert field in summary


# --------------------------------------------------------------------------
# Marker calibration
# --------------------------------------------------------------------------
#
# The question set is the measuring instrument for every number this project
# publishes, and it was the only component with no calibration test. Substring
# matching made several questions unfalsifiable and nothing in CI could notice.


@pytest.mark.parametrize(
    ("marker", "text"),
    [
        ("ai", "said the chairman"),
        ("ai", "available maintenance"),
        ("upi", "the building was occupied"),
        ("rs", "it took years"),
        ("rs", "special offers"),
        ("100", "priced at 21000 rupees"),
        ("gil", "a fragile agreement"),
    ],
)
def test_a_marker_does_not_match_a_word_that_merely_contains_it(
    marker: str, text: str
) -> None:
    from frugal.benchmark import marker_matches

    assert not marker_matches(marker, text)


@pytest.mark.parametrize(
    ("marker", "text"),
    [
        ("ai", "AI research"),
        ("upi", "UPI payments"),
        ("rs", "priced at Rs. 2999"),
        ("100", "boils at 100 degrees"),
        ("gil", "the GIL blocks threads"),
        ("retrieval-augmented generation", "on Retrieval-Augmented Generation for NLP"),
    ],
)
def test_a_marker_still_matches_the_thing_it_is_for(marker: str, text: str) -> None:
    from frugal.benchmark import marker_matches

    assert marker_matches(marker, text)


@pytest.mark.parametrize("marker", ["₹", "°c"])
def test_a_marker_of_punctuation_is_not_given_word_boundaries(marker: str) -> None:
    """A boundary next to a symbol would never match; the rupee sign is a marker."""
    from frugal.benchmark import marker_matches

    assert marker_matches("₹", "costs ₹2999")
    assert marker_matches("°c", "boils at 100 °c")


def test_regex_metacharacters_in_a_marker_are_literal() -> None:
    from frugal.benchmark import marker_matches

    assert marker_matches("c++", "written in c++")
    assert not marker_matches("c++", "written in c")


def test_no_shipped_marker_matches_arbitrary_prose() -> None:
    """A marker satisfied by unrelated text makes its question unfalsifiable."""
    from frugal.benchmark import marker_matches

    filler = (
        "The quick brown fox jumps over the lazy dog while years of available "
        "maintenance offers occupied the chairman and said nothing at 21000 rupees."
    )
    for question in load_questions():
        for group in question.markers:
            for alt in group:
                assert not marker_matches(alt, filler), (
                    f"{question.id}: marker {alt!r} matches unrelated prose"
                )
