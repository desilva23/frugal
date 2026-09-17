"""Tests for engine-specific request parameters.

These exist because their absence was a correctness bug, not a missing feature:
a question about interest in electric vehicles *in India* was answered with
worldwide trends data, because the planner sent no `geo`. The series was real,
well formed, and about the wrong place.
"""

from __future__ import annotations

import pytest

from frugal.parameters import build_params, detect_locale, wants_recent

# --------------------------------------------------------------------------
# Place detection
# --------------------------------------------------------------------------


def test_a_country_is_detected() -> None:
    locale = detect_locale("Is interest in electric vehicles growing in India?")
    assert locale is not None and locale.country == "IN"


def test_a_city_is_detected_with_its_full_place_string() -> None:
    locale = detect_locale("Which companies are hiring in Chennai?")
    assert locale is not None
    assert locale.location == "Chennai, Tamil Nadu, India"
    assert locale.country == "IN"


def test_a_city_wins_over_the_country_it_is_in() -> None:
    """The more specific answer is the more useful one."""
    locale = detect_locale("Python jobs in Chennai, India")
    assert locale is not None and locale.location is not None
    assert "Chennai" in locale.location


def test_alternative_city_names_resolve_to_one_place() -> None:
    a = detect_locale("jobs in Bangalore")
    b = detect_locale("jobs in Bengaluru")
    assert a is not None and b is not None
    assert a.location == b.location


def test_detection_is_case_insensitive() -> None:
    assert detect_locale("jobs in CHENNAI") is not None


def test_word_boundaries_are_respected() -> None:
    """"indiana" is not "india"."""
    assert detect_locale("universities in Indiana") is None


def test_a_question_with_no_place_has_no_locale() -> None:
    assert detect_locale("How does photosynthesis work?") is None


def test_the_locale_reports_what_matched() -> None:
    locale = detect_locale("hiring in Chennai")
    assert locale is not None and "chennai" in locale.describe()


# --------------------------------------------------------------------------
# Trends — the engine the bug was found on
# --------------------------------------------------------------------------


def test_trends_receives_the_country() -> None:
    """Without this, a question about India is answered for the world."""
    locale = detect_locale("electric vehicles in India")
    params = build_params("google_trends", "electric vehicles", locale=locale)
    assert params["geo"] == "IN"


def test_trends_always_requests_a_timeseries() -> None:
    assert build_params("google_trends", "x")["data_type"] == "TIMESERIES"


def test_trends_is_not_sent_a_page_size_it_has_no_use_for() -> None:
    assert "num" not in build_params("google_trends", "x")


def test_trends_without_a_place_sends_no_geo() -> None:
    """Better to answer for the world than to guess at a country."""
    assert "geo" not in build_params("google_trends", "x", locale=None)


# --------------------------------------------------------------------------
# Location-sensitive engines
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["google_jobs", "google_maps"])
def test_location_reaches_the_engines_built_around_it(engine: str) -> None:
    locale = detect_locale("hiring in Chennai")
    params = build_params(engine, "python developer", locale=locale)
    assert params["location"] == "Chennai, Tamil Nadu, India"


def test_a_country_only_locale_sets_gl_but_not_location() -> None:
    """SerpApi's location matches place strings, not country codes."""
    locale = detect_locale("jobs in India")
    params = build_params("google_jobs", "python", locale=locale)
    assert params["gl"] == "in"
    assert "location" not in params


def test_shopping_gets_the_country_so_prices_are_local() -> None:
    locale = detect_locale("earbuds under 3000 rupees in India")
    assert build_params("google_shopping", "earbuds", locale=locale)["gl"] == "in"


def test_web_search_gets_the_country_and_language() -> None:
    locale = detect_locale("news from India")
    params = build_params("google", "x", locale=locale)
    assert params["gl"] == "in"
    assert params["hl"] == "en"


# --------------------------------------------------------------------------
# Engines that should not be narrowed
# --------------------------------------------------------------------------


def test_patents_is_not_given_a_country() -> None:
    """Patents is jurisdictional; a gl would narrow it without being asked."""
    locale = detect_locale("solar patents in India")
    assert "gl" not in build_params("google_patents", "solar", locale=locale)


def test_scholar_narrows_by_year_only_when_recency_is_asked_for() -> None:
    assert "as_ylo" not in build_params("google_scholar", "x", recent=False, current_year=2026)
    params = build_params("google_scholar", "x", recent=True, current_year=2026)
    assert params["as_ylo"] == 2021


def test_scholar_needs_a_year_to_narrow_by() -> None:
    assert "as_ylo" not in build_params("google_scholar", "x", recent=True, current_year=None)


# --------------------------------------------------------------------------
# Recency detection
# --------------------------------------------------------------------------


def test_recency_words_are_detected() -> None:
    assert wants_recent("What are the latest findings?")
    assert wants_recent("recent studies on sleep")


def test_a_timeless_question_is_not_recent() -> None:
    assert not wants_recent("How does photosynthesis work?")


def test_recency_respects_word_boundaries() -> None:
    assert not wants_recent("the renewal process")


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def test_every_engine_receives_the_query() -> None:
    for engine in ["google", "google_news", "google_trends", "google_jobs", "google_patents"]:
        assert build_params(engine, "the query")["q"] == "the query"


def test_engine_names_are_matched_case_insensitively() -> None:
    assert build_params("GOOGLE_TRENDS", "x")["data_type"] == "TIMESERIES"


# --------------------------------------------------------------------------
# A constraint expressed twice
# --------------------------------------------------------------------------
#
# Setting a parameter *and* leaving the same constraint in the query text
# over-constrains the index. A jobs search for "python developers chennai" with
# location set to Chennai returned no results at all, while either alone
# returned plenty. The search was spent and bought nothing, which is the exact
# waste this project exists to remove.


def test_the_place_is_dropped_from_the_query_when_it_becomes_a_parameter() -> None:
    locale = detect_locale("Which companies are hiring Python developers in Chennai?")
    params = build_params(
        "google_jobs", "companies hiring python developers chennai", locale=locale
    )
    assert params["location"] == "Chennai, Tamil Nadu, India"
    assert "chennai" not in params["q"].lower()
    assert "python" in params["q"]


def test_the_country_is_dropped_from_a_shopping_query() -> None:
    locale = detect_locale("What laptops are available under 50000 rupees in India?")
    params = build_params("google_shopping", "laptops available india", locale=locale)
    assert params["gl"] == "in"
    assert "india" not in params["q"].lower()


def test_the_place_is_dropped_from_a_trends_query() -> None:
    locale = detect_locale("Is interest in electric vehicles growing in India?")
    params = build_params("google_trends", "electric vehicles india", locale=locale)
    assert params["geo"] == "IN"
    assert "india" not in params["q"].lower()


def test_web_search_keeps_the_place_in_the_query() -> None:
    """Web search is not over-constrained by it, and it helps the match."""
    locale = detect_locale("news from India")
    assert "india" in build_params("google", "railways india", locale=locale)["q"].lower()


def test_stripping_never_empties_a_query() -> None:
    """A query stripped to nothing retrieves nothing, which is strictly worse."""
    locale = detect_locale("India")
    assert build_params("google_shopping", "india", locale=locale)["q"]


# --------------------------------------------------------------------------
# Qualifiers are not product names
# --------------------------------------------------------------------------


def test_prices_and_qualifiers_are_dropped_from_a_shopping_query() -> None:
    """Shopping matches product names; it does not apply a price as a filter."""
    params = build_params("google_shopping", "wireless earbuds available under 3000 rupees")
    assert params["q"] == "wireless earbuds"


def test_a_shopping_query_keeps_the_product() -> None:
    assert "laptops" in build_params("google_shopping", "best laptops under 50000")["q"]


def test_other_engines_keep_their_qualifiers() -> None:
    """Only shopping treats these as noise; web search matches on them."""
    assert "under" in build_params("google", "laptops under 50000")["q"]
