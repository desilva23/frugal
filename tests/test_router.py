"""Tests for engine routing.

Routing is the project's central claim, so the bar here is that each question
shape reaches the engine that actually answers it — and, just as importantly,
does *not* reach engines that merely look plausible. A wasted search is the
failure this whole project exists to eliminate, so the negative cases below
carry as much weight as the positive ones.

The suite also pins determinism. A model in this loop would make the published
benchmark unrepeatable; these tests are what stops one creeping in.
"""

from __future__ import annotations

import pytest

from frugal.router import (
    PROFILES,
    SIGNALS,
    detect_signals,
    profile_for,
    route,
)


def engines(question: str, limit: int | None = None) -> list[str]:
    return [d.engine for d in route(question, limit=limit)]


def top(question: str) -> str:
    return route(question)[0].engine


# --------------------------------------------------------------------------
# Each question shape reaches its engine
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What did recent studies find about transformer scaling?", "google_scholar"),
        ("Which paper first described retrieval augmented generation?", "google_scholar"),
        ("What is the cheapest wireless earbuds under 3000 rupees?", "google_shopping"),
        ("Best price on a mechanical keyboard", "google_shopping"),
        ("Which companies are hiring python developers?", "google_jobs"),
        ("What salary do data engineers get in Bangalore?", "google_jobs"),
        ("Find a coffee shop near me that is open now", "google_maps"),
        ("Give me directions to the nearest hospital", "google_maps"),
        ("Who holds patents on perovskite solar cells?", "google_patents"),
        ("Is interest in electric vehicles growing over time?", "google_trends"),
        ("What is the latest news on railway electrification?", "google_news"),
    ],
)
def test_question_shapes_reach_their_engine(question: str, expected: str) -> None:
    assert top(question) == expected


def test_a_question_with_no_shape_falls_back_to_web_search() -> None:
    """Returning nothing would be worse than returning something unspecialised."""
    assert engines("How does photosynthesis work?") == ["google"]


def test_web_search_is_always_available() -> None:
    for question in [
        "cheapest laptop deals",
        "patents on solar cells",
        "jobs in Chennai",
        "how does a transformer work",
    ]:
        assert "google" in engines(question)


# --------------------------------------------------------------------------
# Negative cases — the wasted searches this project exists to remove
# --------------------------------------------------------------------------


def test_a_local_question_does_not_route_to_news() -> None:
    """Regression: bare "now" in "open now" used to fire the recency signal."""
    assert "google_news" not in engines("Find a coffee shop near me that is open now")


def test_an_ip_address_question_does_not_route_to_patents() -> None:
    """Regression: "ip" as a patent pattern sent network questions to patents."""
    assert "google_patents" not in engines("What is my IP address?")


def test_an_ip_address_question_does_not_route_to_maps() -> None:
    """Regression: bare "address" fired the local signal on "IP address"."""
    assert engines("What is my IP address?") == ["google"]


def test_an_email_address_question_does_not_route_to_maps() -> None:
    assert engines("What is my email address?") == ["google"]


def test_a_street_address_question_still_routes_to_maps() -> None:
    """The scoped pattern must not have broken the genuine case."""
    assert top("What is the address of Chennai airport?") == "google_maps"


def test_a_scholarly_question_does_not_route_to_shopping() -> None:
    assert "google_shopping" not in engines("What does the research say about sleep cycles?")


def test_a_patent_question_does_not_route_to_shopping() -> None:
    assert "google_shopping" not in engines("Who patented the lithium iron phosphate cathode?")


def test_a_jobs_question_does_not_route_to_shopping() -> None:
    assert "google_shopping" not in engines("What is the salary for a backend role?")


# --------------------------------------------------------------------------
# Signal detection
# --------------------------------------------------------------------------


def test_signals_respect_word_boundaries() -> None:
    """"jobless" is not an employment question and "priceless" is not commerce."""
    assert "employment" not in detect_signals("the jobless rate fell")
    assert "commerce" not in detect_signals("the view was priceless")


def test_signal_matching_is_case_insensitive() -> None:
    assert "patent" in detect_signals("Who holds PATENTS on this?")


def test_reasons_report_the_questions_own_words_not_patterns() -> None:
    """A plan trace showing a regex helps nobody read it."""
    decision = route("Find a cafe near me")[0]
    assert decision.reasons
    assert "(?<" not in decision.explain()
    assert "near me" in decision.explain()


def test_multiple_signals_are_all_reported() -> None:
    fired = detect_signals("What is the cheapest laptop and the latest deals?")
    assert "commerce" in fired
    assert "recency" in fired


def test_explain_reads_as_a_sentence() -> None:
    assert "score" in route("patents on graphene")[0].explain()


def test_fallback_explains_itself() -> None:
    assert "fallback" in route("how does photosynthesis work")[0].explain()


# --------------------------------------------------------------------------
# Ranking, limits and determinism
# --------------------------------------------------------------------------


def test_results_are_ordered_by_descending_score() -> None:
    scores = [d.score for d in route("latest research on battery chemistry")]
    assert scores == sorted(scores, reverse=True)


def test_limit_caps_the_number_of_engines() -> None:
    assert len(route("latest cheapest research jobs near me", limit=2)) == 2


def test_limit_of_one_still_returns_something() -> None:
    assert len(route("anything at all", limit=1)) == 1


def test_a_high_floor_still_leaves_web_search() -> None:
    """The plan must never come back empty."""
    assert engines_above_floor("how does photosynthesis work", floor=99.0) == ["google"]


def engines_above_floor(question: str, floor: float) -> list[str]:
    return [d.engine for d in route(question, floor=floor)]


def test_routing_is_deterministic() -> None:
    """A model in this loop would make the published benchmark unrepeatable."""
    question = "What is the cheapest electric scooter and is interest growing?"
    first = [(d.engine, d.score) for d in route(question)]
    for _ in range(20):
        assert [(d.engine, d.score) for d in route(question)] == first


def test_an_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty question"):
        route("   ")


# --------------------------------------------------------------------------
# Profile integrity
# --------------------------------------------------------------------------


def test_every_profile_references_real_signals() -> None:
    for profile in PROFILES:
        for name in (*profile.serves, *profile.suppressed_by):
            assert name in SIGNALS, f"{profile.engine} references unknown signal {name!r}"


def test_no_profile_both_serves_and_suppresses_one_signal() -> None:
    for profile in PROFILES:
        overlap = set(profile.serves) & set(profile.suppressed_by)
        assert not overlap, f"{profile.engine} both serves and suppresses {overlap}"


def test_profiles_are_unique_per_engine() -> None:
    names = [p.engine for p in PROFILES]
    assert len(names) == len(set(names))


def test_every_engine_costs_at_least_one_search() -> None:
    assert all(p.cost >= 1 for p in PROFILES)


def test_profile_lookup_is_case_insensitive() -> None:
    assert profile_for("GOOGLE_NEWS") is not None
    assert profile_for("not_an_engine") is None


def test_every_routable_engine_can_be_normalised() -> None:
    """The invariant tying the two halves together.

    Routing an engine the normaliser cannot parse would spend a search and then
    throw the result away, which is precisely the waste this project exists to
    remove.
    """
    from frugal.adapters import ADAPTERS

    for profile in PROFILES:
        assert profile.engine in ADAPTERS, f"{profile.engine} is routable but has no adapter"
