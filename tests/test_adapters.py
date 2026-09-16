"""Tests for per-engine normalisation.

Half of this file runs against fixtures trimmed from live responses, committed
under tests/fixtures. That is deliberate: an adapter written against
documentation passes tests written against documentation while failing on the
actual payload. These fixtures are what the engines really returned, so a future
rename breaks the suite rather than silently emptying an evidence set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from frugal.adapters import ADAPTERS, EngineAdapter, TrendsAdapter, _as_text, normalise
from frugal.errors import SchemaDrift
from frugal.schema import Document, Series

FIXTURES = Path(__file__).parent / "fixtures"

DOCUMENT_ENGINES = [
    "google",
    "google_news",
    "google_scholar",
    "google_shopping",
    "google_jobs",
    "google_maps",
    "google_patents",
]


def load(engine: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{engine}.json").read_text(encoding="utf-8"))


def parse(engine: str) -> list[Any]:
    return normalise(engine, load(engine), query="test query")


# --------------------------------------------------------------------------
# Against real payloads
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", DOCUMENT_ENGINES)
def test_every_engine_yields_documents(engine: str) -> None:
    evidence = parse(engine)
    assert evidence, f"{engine} produced no evidence from its recorded payload"
    assert all(isinstance(item, Document) for item in evidence)


@pytest.mark.parametrize("engine", DOCUMENT_ENGINES)
def test_every_document_has_a_title(engine: str) -> None:
    """The one field the survey found on every engine."""
    assert all(doc.title.strip() for doc in parse(engine))


@pytest.mark.parametrize("engine", DOCUMENT_ENGINES)
def test_provenance_records_the_engine_and_query(engine: str) -> None:
    for doc in parse(engine):
        assert doc.provenance.engine == engine
        assert doc.provenance.query == "test query"
        assert doc.provenance.position >= 1


def test_jobs_resolves_a_link_despite_having_no_link_field() -> None:
    """The finding that motivated candidate URL paths: jobs has no 'link'."""
    docs = parse("google_jobs")
    assert all(doc.url for doc in docs)
    raw = load("google_jobs")["jobs_results"][0]
    assert "link" not in raw
    assert docs[0].url == raw["apply_options"][0]["link"]


def test_jobs_uses_the_company_as_its_source() -> None:
    assert parse("google_jobs")[0].source


def test_maps_uses_the_website_field_for_its_url() -> None:
    docs = parse("google_maps")
    raw = load("google_maps")["local_results"][0]
    assert docs[0].url == raw.get("website")


def test_maps_tolerates_a_result_with_no_website() -> None:
    """A business without a website is ordinary, not schema drift."""
    payload = load("google_maps")
    for result in payload["local_results"]:
        result.pop("website", None)
    docs = normalise("google_maps", payload, query="q")
    assert docs and all(doc.url is None for doc in docs)


def test_news_parses_its_publication_date() -> None:
    """News is the only surveyed engine supplying a parsed iso_date."""
    assert any(doc.published_at for doc in parse("google_news"))


def test_shopping_keeps_price_in_extra() -> None:
    """No snippet field exists; price and seller are what distinguish a result."""
    doc = parse("google_shopping")[0]
    assert doc.snippet is None
    assert "price" in doc.extra


def test_scholar_keeps_its_publication_summary() -> None:
    doc = parse("google_scholar")[0]
    assert doc.source
    assert "publication_info.summary" in doc.extra


def test_patents_dates_the_document_by_publication() -> None:
    docs = parse("google_patents")
    assert all(doc.published_at for doc in docs)


def test_trends_produces_a_series_not_documents() -> None:
    evidence = parse("google_trends")
    assert evidence and all(isinstance(item, Series) for item in evidence)
    series = evidence[0]
    assert len(series.points) > 1
    assert series.points[0].timestamp < series.points[-1].timestamp


def test_trends_series_summarises() -> None:
    summary = parse("google_trends")[0].summarise()
    assert "observations" in summary


# --------------------------------------------------------------------------
# Drift detection
# --------------------------------------------------------------------------


def test_a_renamed_result_key_raises_rather_than_returning_nothing() -> None:
    """The failure this module exists to prevent: silence on a rename."""
    payload = {"unexpected_results": [{"title": "x", "link": "https://example.com"}]}
    with pytest.raises(SchemaDrift, match="organic_results"):
        normalise("google", payload, query="q")


def test_drift_message_points_at_the_survey_script() -> None:
    with pytest.raises(SchemaDrift, match=r"survey_engines\.py"):
        normalise("google", {"nothing": 1}, query="q")


def test_a_genuinely_empty_result_set_is_not_drift() -> None:
    """A query nobody has written about legitimately returns nothing."""
    payload = {"search_metadata": {"status": "Success"}}
    assert normalise("google", payload, query="q") == []


def test_results_without_titles_raise() -> None:
    payload = {"organic_results": [{"link": "https://example.com"} for _ in range(3)]}
    with pytest.raises(SchemaDrift, match="none carried a title"):
        normalise("google", payload, query="q")


def test_a_renamed_link_field_raises() -> None:
    """Most results losing their URL means a rename, not a run of odd results."""
    payload = {"organic_results": [{"title": f"t{i}", "href": "x"} for i in range(4)]}
    with pytest.raises(SchemaDrift, match="resolved a URL"):
        normalise("google", payload, query="q")


def test_a_single_result_without_a_url_is_tolerated() -> None:
    """One odd result should not cry wolf."""
    results: list[dict[str, Any]] = [
        {"title": f"t{i}", "link": f"https://example.com/{i}"} for i in range(4)
    ]
    results.append({"title": "no link"})
    docs = normalise("google", {"organic_results": results}, query="q")
    assert len(docs) == 5


def test_malformed_entries_are_skipped_not_fatal() -> None:
    payload = {
        "organic_results": [
            "a bare string",
            {"title": "good", "link": "https://example.com/a"},
            None,
        ]
    }
    docs = normalise("google", payload, query="q")
    assert [d.title for d in docs] == ["good"]


def test_trends_without_its_timeline_raises() -> None:
    with pytest.raises(SchemaDrift, match="interest_over_time"):
        normalise("google_trends", {"something_else": {}}, query="q")


def test_trends_with_unusable_observations_raises() -> None:
    payload = {"interest_over_time": {"timeline_data": [{"timestamp": "x", "values": []}]}}
    with pytest.raises(SchemaDrift):
        normalise("google_trends", payload, query="q")


# --------------------------------------------------------------------------
# Field flattening
# --------------------------------------------------------------------------


def test_source_is_read_whether_string_or_object() -> None:
    """'source' is a string on web search and an object with 'name' on news."""
    assert _as_text("Reuters") == "Reuters"
    assert _as_text({"name": "Reuters", "icon": "..."}) == "Reuters"


def test_source_object_without_a_name_yields_none() -> None:
    assert _as_text({"icon": "..."}) is None


def test_lists_are_joined() -> None:
    assert _as_text(["a", "b"]) == "a, b"


def test_booleans_are_not_text() -> None:
    assert _as_text(True) is None


def test_numbers_render_as_text() -> None:
    assert _as_text(4.5) == "4.5"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_an_unknown_engine_falls_back_to_the_web_search_mapping() -> None:
    """Most SerpApi engines follow the organic_results convention."""
    payload = {"organic_results": [{"title": "x", "link": "https://example.com"}]}
    docs = normalise("some_new_engine", payload, query="q")
    assert len(docs) == 1
    assert docs[0].provenance.engine == "some_new_engine"


def test_engine_names_are_matched_case_insensitively() -> None:
    assert normalise("GOOGLE", load("google"), query="q")


def test_every_surveyed_engine_has_an_adapter() -> None:
    for engine in [*DOCUMENT_ENGINES, "google_trends"]:
        assert engine in ADAPTERS


def test_adapters_declare_the_engine_they_serve() -> None:
    for name, adapter in ADAPTERS.items():
        assert adapter.engine == name


def test_document_adapters_declare_at_least_one_result_key() -> None:
    for adapter in ADAPTERS.values():
        if isinstance(adapter, EngineAdapter):
            assert adapter.result_keys
        else:
            assert isinstance(adapter, TrendsAdapter)
