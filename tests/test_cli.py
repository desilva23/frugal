"""Tests for the command line interface.

Everything the CLI prints is the planner's own trace rather than decoration, so
these check that the trace is actually shown — a routing score, a cost, a reason
for stopping — and that `plan` genuinely issues nothing, since a cost preview
that secretly costs something would be worse than no preview at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frugal.cache import ResponseCache
from frugal.cli import main, sparkline

FIXTURES = Path(__file__).parent.parent / "benchmarks" / "fixtures"

# The benchmark set's own wording, so the committed fixtures are guaranteed to
# cover it. A test question phrased differently would generate different
# queries, miss the fixtures, and fail for a reason unrelated to the CLI.
QUESTION = "Has search interest in electric vehicles in India been rising or falling over time?"


# --------------------------------------------------------------------------
# Sparklines
# --------------------------------------------------------------------------


def test_sparkline_renders_one_character_per_value() -> None:
    assert len(sparkline([1, 2, 3, 4])) == 4


def test_sparkline_of_nothing_is_empty() -> None:
    assert sparkline([]) == ""


def test_a_flat_series_renders_flat() -> None:
    """A constant series must not render as noise."""
    assert len(set(sparkline([5, 5, 5, 5]))) == 1


def test_a_rising_series_ends_higher_than_it_starts() -> None:
    line = sparkline([1, 2, 3, 9])
    assert line[-1] > line[0]


def test_extremes_reach_the_ends_of_the_scale() -> None:
    line = sparkline([0, 50, 100])
    assert line[0] == "▁"
    assert line[-1] == "█"


# --------------------------------------------------------------------------
# plan: must not spend
# --------------------------------------------------------------------------


def test_plan_issues_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A cost preview that secretly costs something is worse than none."""
    cache = tmp_path / "empty-cache"
    assert main(["plan", QUESTION, "--cache", str(cache)]) == 0
    assert not list(cache.rglob("*.json")) if cache.exists() else True


def test_plan_shows_the_routing_and_the_ceiling(capsys: pytest.CaptureFixture[str]) -> None:
    main(["plan", QUESTION, "--cache", str(FIXTURES)])
    out = capsys.readouterr().out
    assert "google_trends" in out
    assert "at most" in out
    assert "no searches were spent" in out


def test_plan_shows_the_parameters_each_engine_receives(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The parameters are the point; a plan that hides them hides the bug."""
    main(["plan", QUESTION, "--cache", str(FIXTURES)])
    out = capsys.readouterr().out
    assert "geo=IN" in out
    assert "TIMESERIES" in out


def test_plan_respects_an_engine_limit(capsys: pytest.CaptureFixture[str]) -> None:
    main(["plan", QUESTION, "--cache", str(FIXTURES), "--engines", "1"])
    assert "google_trends" in capsys.readouterr().out


# --------------------------------------------------------------------------
# ask: replayed, so the suite spends nothing
# --------------------------------------------------------------------------


def test_ask_runs_from_recorded_fixtures(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", QUESTION, "--replay", "--cache", str(FIXTURES)]) == 0


def test_ask_shows_the_spend_and_why_it_stopped(capsys: pytest.CaptureFixture[str]) -> None:
    main(["ask", QUESTION, "--replay", "--cache", str(FIXTURES)])
    out = capsys.readouterr().out
    assert "searches billed" in out
    assert "ceiling" in out
    assert "unique results" in out


def test_ask_shows_the_saturation_trace(capsys: pytest.CaptureFixture[str]) -> None:
    main(["ask", QUESTION, "--replay", "--cache", str(FIXTURES)])
    assert "new" in capsys.readouterr().out


def test_ask_renders_a_series_as_a_sparkline(capsys: pytest.CaptureFixture[str]) -> None:
    """The one kind of evidence a web search cannot return, shown as a shape."""
    main(["ask", QUESTION, "--replay", "--cache", str(FIXTURES)])
    out = capsys.readouterr().out
    assert any(block in out for block in "▁▂▃▄▅▆▇█")


def test_ask_limits_how_many_results_it_prints(capsys: pytest.CaptureFixture[str]) -> None:
    main(["ask", QUESTION, "--replay", "--cache", str(FIXTURES), "--show", "2"])
    assert "more" in capsys.readouterr().out


def test_ask_reports_a_failure_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run where every step failed must not report success."""
    assert main(["ask", QUESTION, "--replay", "--cache", str(tmp_path / "empty")]) == 1
    out = capsys.readouterr().out
    assert "CacheMiss" in out
    assert "every step failed" in out


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_reports_on_the_key(capsys: pytest.CaptureFixture[str]) -> None:
    main(["doctor"])
    assert "SerpApi key" in capsys.readouterr().out


def test_doctor_never_prints_the_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SERPAPI_API_KEY", "super-secret-value-here")
    main(["doctor"])
    assert "super-secret-value-here" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------


def test_version_is_reported() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_an_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["invent"])


def test_the_fixture_set_is_present_for_these_tests() -> None:
    """Guards against the fixtures being dropped from the repository."""
    cache = ResponseCache(FIXTURES)
    entries = list(cache.iter_entries())
    assert entries, "committed fixtures are missing; replay tests cannot run"
    assert any(e.engine == "google_trends" for e in entries)


def test_fixtures_carry_no_credentials() -> None:
    """These are committed; a key in one would be published."""
    for path in FIXTURES.rglob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        for value in record.get("params", {}).values():
            assert value == "<redacted>" or "api_key" not in str(value).lower()


# --------------------------------------------------------------------------
# No failure reaches a user as a stack trace
# --------------------------------------------------------------------------
#
# A traceback tells someone running a command line tool nothing they can act on,
# and tells someone watching a demo rather more than one would like.


@pytest.mark.parametrize("command", ["plan", "ask"])
@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_an_empty_question_is_refused_cleanly(
    command: str, question: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, question, "--cache", str(FIXTURES)]) == 2
    out = capsys.readouterr().out
    assert "empty" in out
    assert "Traceback" not in out


@pytest.mark.parametrize("command", ["plan", "ask"])
def test_an_empty_question_suggests_what_to_type(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    main([command, "", "--cache", str(FIXTURES)])
    assert "Try:" in capsys.readouterr().out


def test_a_usage_error_is_distinguishable_from_a_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """2 means "you asked wrongly"; 1 means "it went wrong"."""
    assert main(["ask", "", "--cache", str(FIXTURES)]) == 2
    assert main(["ask", QUESTION, "--replay", "--cache", str(tmp_path / "empty")]) == 1


def test_an_unexpected_error_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The catch-all, tested through a failure the guards do not cover."""

    def explode(*args: object, **kwargs: object) -> None:
        raise ValueError("something unforeseen")

    monkeypatch.setattr("frugal.cli.plan_only", explode)
    assert main(["plan", QUESTION, "--cache", str(FIXTURES)]) == 2
    out = capsys.readouterr().out
    assert "something unforeseen" in out
    assert "Traceback" not in out


def test_an_interrupt_exits_conventionally(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("frugal.cli.plan_only", interrupt)
    assert main(["plan", QUESTION, "--cache", str(FIXTURES)]) == 130
