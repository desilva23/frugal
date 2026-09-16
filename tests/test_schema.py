"""Tests for the evidence model and its primitives.

Deduplication compares canonical URLs, so the rules in canonical_url decide
whether the benchmark's recall numbers mean anything: an over-eager rule merges
distinct pages, an under-eager one lets the same page count twice across engines.
Most of this file is about that boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime

from frugal.schema import (
    Document,
    Provenance,
    Series,
    SeriesPoint,
    canonical_url,
    deduplicate,
    dig,
    first_present,
    parse_timestamp,
)

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def provenance(engine: str = "google", position: int = 1) -> Provenance:
    return Provenance(engine=engine, query="q", position=position, retrieved_at=NOW)


def document(**kwargs: object) -> Document:
    return Document(title=kwargs.pop("title", "t"), provenance=provenance(), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# dig
# --------------------------------------------------------------------------


def test_dig_reads_a_flat_key() -> None:
    assert dig({"a": 1}, "a") == 1


def test_dig_reads_a_nested_path() -> None:
    assert dig({"a": {"b": {"c": 3}}}, "a.b.c") == 3


def test_dig_indexes_into_a_list() -> None:
    assert dig({"a": [{"b": 1}, {"b": 2}]}, "a[1].b") == 2


def test_dig_supports_a_negative_index() -> None:
    assert dig({"a": [1, 2, 3]}, "a[-1]") == 3


def test_dig_returns_default_for_a_missing_key() -> None:
    assert dig({"a": 1}, "b", default="fallback") == "fallback"


def test_dig_returns_default_for_an_out_of_range_index() -> None:
    assert dig({"a": [1]}, "a[9]") is None


def test_dig_returns_default_when_a_segment_is_the_wrong_type() -> None:
    """Live payloads change shape; a TypeError from inside a parse helps nobody."""
    assert dig({"a": "string"}, "a.b") is None
    assert dig({"a": {"b": 1}}, "a[0]") is None


def test_dig_returns_default_for_a_null_along_the_path() -> None:
    assert dig({"a": None}, "a.b.c") is None


def test_first_present_skips_empty_values() -> None:
    payload = {"a": "", "b": [], "c": {}, "d": "found"}
    assert first_present(payload, ("a", "b", "c", "d")) == "found"


def test_first_present_returns_default_when_nothing_matches() -> None:
    assert first_present({"a": 1}, ("x", "y"), default="none") == "none"


def test_first_present_keeps_a_falsy_but_meaningful_zero() -> None:
    assert first_present({"rating": 0}, ("rating",)) == 0


# --------------------------------------------------------------------------
# canonical_url
# --------------------------------------------------------------------------


def test_identical_urls_canonicalise_identically() -> None:
    assert canonical_url("https://example.com/a") == canonical_url("https://example.com/a")


def test_www_is_dropped() -> None:
    assert canonical_url("https://www.example.com/a") == canonical_url("https://example.com/a")


def test_host_case_is_normalised_but_path_case_is_not() -> None:
    """Hosts are case-insensitive; paths are not, and /A may differ from /a."""
    assert canonical_url("https://EXAMPLE.com/a") == canonical_url("https://example.com/a")
    assert canonical_url("https://example.com/A") != canonical_url("https://example.com/a")


def test_fragment_is_dropped() -> None:
    assert canonical_url("https://example.com/a#section") == canonical_url("https://example.com/a")


def test_trailing_slash_is_dropped() -> None:
    assert canonical_url("https://example.com/a/") == canonical_url("https://example.com/a")


def test_bare_root_keeps_its_slash() -> None:
    assert canonical_url("https://example.com") == "https://example.com/"


def test_campaign_parameters_are_stripped() -> None:
    tracked = "https://example.com/a?utm_source=news&utm_campaign=x&gclid=123"
    assert canonical_url(tracked) == canonical_url("https://example.com/a")


def test_meaningful_parameters_are_kept() -> None:
    assert canonical_url("https://example.com/a?id=7") != canonical_url("https://example.com/a")


def test_parameter_order_does_not_matter() -> None:
    assert canonical_url("https://example.com/a?b=2&a=1") == canonical_url(
        "https://example.com/a?a=1&b=2"
    )


def test_default_ports_are_dropped() -> None:
    assert canonical_url("http://example.com:80/a") == canonical_url("http://example.com/a")
    assert canonical_url("https://example.com:443/a") == canonical_url("https://example.com/a")


def test_non_default_port_is_kept() -> None:
    assert canonical_url("https://example.com:8443/a") != canonical_url("https://example.com/a")


def test_non_http_schemes_are_rejected() -> None:
    assert canonical_url("mailto:someone@example.com") is None
    assert canonical_url("javascript:alert(1)") is None
    assert canonical_url("ftp://example.com/file") is None


def test_relative_and_empty_input_is_rejected() -> None:
    assert canonical_url("/relative/path") is None
    assert canonical_url("") is None
    assert canonical_url(None) is None
    assert canonical_url("   ") is None


def test_non_string_input_is_rejected() -> None:
    assert canonical_url(12345) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# parse_timestamp
# --------------------------------------------------------------------------


def test_parses_iso_with_zulu_suffix() -> None:
    parsed = parse_timestamp("2026-09-17T12:00:00Z")
    assert parsed == datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def test_parses_a_bare_date() -> None:
    assert parse_timestamp("2026-09-17") == datetime(2026, 9, 17, tzinfo=UTC)


def test_parses_a_unix_timestamp_given_as_a_string() -> None:
    """Trends returns its timestamps this way."""
    assert parse_timestamp("1757808000") is not None


def test_parses_a_human_month_format() -> None:
    assert parse_timestamp("Sep 17, 2026") == datetime(2026, 9, 17, tzinfo=UTC)


def test_naive_input_is_assumed_utc() -> None:
    assert parse_timestamp("2026-09-17T12:00:00").tzinfo is UTC  # type: ignore[union-attr]


def test_relative_text_yields_none_rather_than_a_guess() -> None:
    """A wrong date is worse than a missing one when reasoning about freshness."""
    assert parse_timestamp("2 hours ago") is None
    assert parse_timestamp("yesterday") is None


def test_implausible_timestamps_are_rejected() -> None:
    """Guards a millisecond timestamp being read as the year 55000."""
    assert parse_timestamp(99_999_999_999_999) is None
    assert parse_timestamp(-1) is None


def test_booleans_and_junk_yield_none() -> None:
    assert parse_timestamp(True) is None
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None
    assert parse_timestamp(["2026-09-17"]) is None


# --------------------------------------------------------------------------
# Identity and deduplication
# --------------------------------------------------------------------------


def test_documents_with_equivalent_urls_share_an_identity() -> None:
    a = document(url="https://www.example.com/a?utm_source=x")
    b = document(url="https://example.com/a")
    assert a.identity == b.identity


def test_document_without_a_url_falls_back_to_its_title() -> None:
    a = document(title="Some  Title")
    b = document(title="some title")
    assert a.identity == b.identity


def test_deduplicate_keeps_the_first_occurrence() -> None:
    first = document(title="first", url="https://example.com/a")
    second = document(title="second", url="https://www.example.com/a/")
    assert [d.title for d in deduplicate([first, second])] == ["first"]


def test_deduplicate_collapses_across_engines() -> None:
    """The redundancy the planner exists to eliminate is cross-engine."""
    web = Document(title="x", url="https://example.com/a", provenance=provenance("google"))
    news = Document(title="x", url="https://example.com/a", provenance=provenance("google_news"))
    assert len(deduplicate([web, news])) == 1


def test_deduplicate_preserves_distinct_documents() -> None:
    a = document(url="https://example.com/a")
    b = document(url="https://example.com/b")
    assert len(deduplicate([a, b])) == 2


def test_deduplicate_of_nothing_is_nothing() -> None:
    assert deduplicate([]) == []


# --------------------------------------------------------------------------
# Series
# --------------------------------------------------------------------------


def series(values: list[float]) -> Series:
    points = tuple(
        SeriesPoint(timestamp=datetime(2026, 1, i + 1, tzinfo=UTC), value=v)
        for i, v in enumerate(values)
    )
    return Series(name="term", points=points, provenance=provenance("google_trends"))


def test_trend_reports_relative_change() -> None:
    assert series([50, 100]).trend() == 1.0
    assert series([100, 50]).trend() == -0.5


def test_trend_is_undefined_for_too_few_points() -> None:
    assert series([1]).trend() is None
    assert series([]).trend() is None


def test_trend_is_undefined_when_the_first_value_is_zero() -> None:
    """The ratio has no meaning, and reporting an infinite rise would be false."""
    assert series([0, 50]).trend() is None


def test_span_covers_first_to_last_observation() -> None:
    span = series([1, 2, 3]).span
    assert span is not None and span[0] < span[1]


def test_span_of_an_empty_series_is_none() -> None:
    assert series([]).span is None


def test_summary_is_a_single_line_an_llm_can_consume() -> None:
    summary = series([40, 100]).summarise()
    assert "2 observations" in summary
    assert "up" in summary


def test_flat_series_is_described_as_flat() -> None:
    assert "flat" in series([100, 101]).summarise()


def test_empty_series_summarises_without_raising() -> None:
    assert "no observations" in series([]).summarise()


def test_series_serialises_with_its_summary() -> None:
    payload = series([1, 2]).as_dict()
    assert payload["kind"] == "series"
    assert len(payload["points"]) == 2
    assert "summary" in payload


def test_document_serialises_with_provenance() -> None:
    payload = document(url="https://example.com/a").as_dict()
    assert payload["kind"] == "document"
    assert payload["provenance"]["engine"] == "google"
