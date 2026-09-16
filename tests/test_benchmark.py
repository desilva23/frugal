"""Tests for benchmark scoring.

Scoring decides what the published table says, so the properties that matter are
that a correct answer phrased differently still counts, that a near miss is not
recorded as a hit, and that the whole thing is exact rather than judged.
"""

from __future__ import annotations

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
    headlines like "Why India is Seeing EV Interest Rise". Modality is now
    reported beside recall instead of gating it.
    """
    q = question([["electric vehicles"]], expects_series=True)
    score = score_evidence(q, [doc("electric vehicles interest is rising")])
    assert score.found
    assert not score.answered_in_the_right_modality


def test_a_series_satisfies_a_question_that_needs_one() -> None:
    q = question([["electric vehicles"]], expects_series=True)
    score = score_evidence(q, [series("electric vehicles")])
    assert score.found
    assert score.structured


def test_a_series_contributes_its_summary_text() -> None:
    assert "observations" in searchable_text(series("term"))


def test_modality_is_reported_separately_from_recall() -> None:
    q = question([["electric vehicles"]], expects_series=True)
    prose = score_evidence(q, [doc("electric vehicles")])
    structured = score_evidence(q, [series("electric vehicles")])

    assert prose.found and structured.found
    assert not prose.answered_in_the_right_modality
    assert structured.answered_in_the_right_modality


def test_a_question_not_wanting_a_series_is_always_in_the_right_modality() -> None:
    score = score_evidence(question([["bengaluru"]]), [doc("Bengaluru")])
    assert score.answered_in_the_right_modality


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
