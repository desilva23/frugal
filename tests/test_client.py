"""Tests for the SerpApi transport.

Driven entirely through httpx.MockTransport: the suite must pass with no network
and no credentials, which is the same property that lets the benchmark reproduce
from committed fixtures.

Weighted toward the failure modes that cost money or corrupt a cache — chiefly
that an error response must never be recorded, since a cached error would be
served forever.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from frugal.cache import CacheMode, ResponseCache
from frugal.client import SerpApiClient, backoff_seconds, parse_retry_after
from frugal.errors import (
    AuthenticationError,
    RateLimited,
    SerpApiError,
    TransportError,
)

OK_BODY = {
    "search_metadata": {"id": "abc123", "status": "Success"},
    "organic_results": [{"title": "result", "link": "https://example.com"}],
}


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    tmp_path: Path,
    **kwargs: object,
) -> SerpApiClient:
    """A client wired to a mock transport and a temporary cache."""
    return SerpApiClient(
        api_key="test-key",
        cache=ResponseCache(tmp_path / "cache"),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _seconds: None,
        **kwargs,  # type: ignore[arg-type]
    )


def always(
    status: int, json: object = None, text: str = "", **headers: str
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_request: httpx.Request) -> httpx.Response:
        if json is not None:
            return httpx.Response(status, json=json, headers=headers)
        return httpx.Response(status, text=text, headers=headers)

    return handler


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_successful_search_returns_the_payload(tmp_path: Path) -> None:
    client = make_client(always(200, OK_BODY), tmp_path)
    response = client.search("google", q="python")
    assert response.raw == OK_BODY
    assert response.engine == "google"
    assert response.status == "Success"
    assert response.search_id == "abc123"
    assert response.searches_charged == 1
    assert not response.from_cache


def test_engine_and_key_are_sent_as_query_parameters(tmp_path: Path) -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=OK_BODY)

    make_client(handler, tmp_path).search("google_news", q="x")
    assert seen[0].params["engine"] == "google_news"
    assert seen[0].params["api_key"] == "test-key"


def test_none_parameters_are_not_sent(tmp_path: Path) -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=OK_BODY)

    make_client(handler, tmp_path).search("google", q="x", num=None)
    assert "num" not in seen[0].params


def test_top_level_keys_surveys_an_unfamiliar_engine(tmp_path: Path) -> None:
    response = make_client(always(200, OK_BODY), tmp_path).search("google", q="x")
    assert response.top_level_keys() == ["organic_results", "search_metadata"]


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_second_identical_search_is_served_from_cache_and_costs_nothing(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=OK_BODY)

    client = make_client(handler, tmp_path)
    client.search("google", q="python")
    second = client.search("google", q="python")

    assert calls == 1
    assert second.from_cache
    assert second.searches_charged == 0
    assert client.log.searches_charged == 1
    assert client.log.cache_hits == 1


def test_error_responses_are_never_cached(tmp_path: Path) -> None:
    """A recorded error would be replayed forever, so it must not be stored."""
    cache = ResponseCache(tmp_path / "cache")
    client = SerpApiClient(
        api_key="test-key",
        cache=cache,
        http=httpx.Client(
            transport=httpx.MockTransport(always(200, {"error": "Unsupported location"}))
        ),
        sleep=lambda _s: None,
    )
    with pytest.raises(SerpApiError):
        client.search("google", q="x")
    assert list(cache.directory.rglob("*.json")) == []


def test_replay_mode_needs_no_credentials(tmp_path: Path) -> None:
    """Reproducing a benchmark must not require a key."""
    cache = ResponseCache(tmp_path / "cache", mode=CacheMode.REPLAY)
    client = SerpApiClient(cache=cache, http=httpx.Client())
    assert client is not None


# --------------------------------------------------------------------------
# SerpApi-level errors carried on HTTP 200
# --------------------------------------------------------------------------


def test_error_field_on_a_200_is_raised(tmp_path: Path) -> None:
    client = make_client(always(200, {"error": "Google hasn't returned any results"}), tmp_path)
    with pytest.raises(SerpApiError, match="hasn't returned any results"):
        client.search("google", q="x")


def test_error_list_on_a_200_is_joined(tmp_path: Path) -> None:
    client = make_client(always(200, {"error": ["first problem", "second problem"]}), tmp_path)
    with pytest.raises(SerpApiError, match="first problem; second problem"):
        client.search("google", q="x")


def test_error_status_in_metadata_is_detected(tmp_path: Path) -> None:
    body = {"search_metadata": {"status": "Error", "error": "backend failure"}}
    with pytest.raises(SerpApiError, match="backend failure"):
        make_client(always(200, body), tmp_path).search("google", q="x")


def test_invalid_key_reported_on_a_200_is_an_auth_error_not_a_search_error(tmp_path: Path) -> None:
    """Distinguished so it is not retried and the message is actionable."""
    client = make_client(always(200, {"error": "Invalid API key"}), tmp_path)
    with pytest.raises(AuthenticationError):
        client.search("google", q="x")


def test_blank_error_field_is_not_treated_as_an_error(tmp_path: Path) -> None:
    body = {"error": "   ", "organic_results": []}
    assert make_client(always(200, body), tmp_path).search("google", q="x").raw == body


# --------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------


def test_401_fails_immediately_without_retrying(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "no key"})

    client = make_client(handler, tmp_path)
    with pytest.raises(AuthenticationError):
        client.search("google", q="x")
    assert calls == 1


def test_a_400_is_not_retried(tmp_path: Path) -> None:
    """A malformed request fails identically however often it is sent."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(TransportError, match="HTTP 400"):
        make_client(handler, tmp_path).search("google", q="x")
    assert calls == 1


def test_a_500_is_retried_and_then_succeeds(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(500, text="server error")
        return httpx.Response(200, json=OK_BODY)

    client = make_client(handler, tmp_path)
    assert client.search("google", q="x").raw == OK_BODY
    assert calls == 3
    assert client.log.retries == 2


def test_persistent_429_raises_rate_limited(tmp_path: Path) -> None:
    client = make_client(always(429, text="slow down"), tmp_path, max_attempts=3)
    with pytest.raises(RateLimited) as excinfo:
        client.search("google", q="x")
    assert excinfo.value.attempts == 3


def test_timeouts_are_retried(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise httpx.TimeoutException("timed out")
        return httpx.Response(200, json=OK_BODY)

    assert make_client(handler, tmp_path).search("google", q="x").raw == OK_BODY
    assert calls == 2


def test_exhausted_retries_raise_transport_error(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(TransportError, match="after 2 attempts"):
        make_client(handler, tmp_path, max_attempts=2).search("google", q="x")


# --------------------------------------------------------------------------
# Malformed bodies
# --------------------------------------------------------------------------


def test_non_json_body_raises_transport_error(tmp_path: Path) -> None:
    client = make_client(always(200, text="<html>maintenance</html>"), tmp_path)
    with pytest.raises(TransportError, match="non-JSON"):
        client.search("google", q="x")


def test_json_array_at_top_level_raises_transport_error(tmp_path: Path) -> None:
    client = make_client(always(200, [1, 2, 3]), tmp_path)
    with pytest.raises(TransportError, match="expected an object"):
        client.search("google", q="x")


def test_response_without_metadata_reports_no_status(tmp_path: Path) -> None:
    response = make_client(always(200, {"organic_results": []}), tmp_path).search("google", q="x")
    assert response.status is None
    assert response.search_id is None


# --------------------------------------------------------------------------
# Backoff
# --------------------------------------------------------------------------


def test_backoff_grows_and_stays_jittered() -> None:
    """Jitter stops concurrent workers throttled together from retrying together."""
    samples = [backoff_seconds(3) for _ in range(40)]
    assert len(set(samples)) > 1
    assert all(2.0 <= s <= 4.0 for s in samples)


def test_backoff_is_capped() -> None:
    assert backoff_seconds(40) <= 30.0


def test_retry_after_is_honoured_when_shorter() -> None:
    assert backoff_seconds(8, retry_after=1.5) == 1.5


def test_retry_after_is_capped_too() -> None:
    assert backoff_seconds(1, retry_after=9999.0) == 30.0


def test_unparseable_retry_after_is_ignored() -> None:
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse_retry_after(None) is None
    assert parse_retry_after("-5") is None
    assert parse_retry_after("12") == 12.0


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_an_injected_http_client_is_not_closed(tmp_path: Path) -> None:
    """Its owner closes it; closing it here would break a shared connection pool."""
    http = httpx.Client(transport=httpx.MockTransport(always(200, OK_BODY)))
    with SerpApiClient(api_key="k", cache=ResponseCache(tmp_path / "c"), http=http):
        pass
    assert not http.is_closed


def test_log_accounting_reflects_spend(tmp_path: Path) -> None:
    client = make_client(always(200, OK_BODY), tmp_path)
    client.search("google", q="a")
    client.search("google", q="b")
    client.search("google", q="a")
    assert client.log.as_dict()["searches_charged"] == 2
    assert client.log.as_dict()["cache_hits"] == 1


# --------------------------------------------------------------------------
# A completed search that found nothing
# --------------------------------------------------------------------------
#
# Measured against the SerpApi account counter: three searches returning "no
# results" cost three searches. They arrive with an `error` key AND with
# search_metadata.status == "Success" -- the search completed, which is why it
# was billed. This client used to treat every `error` key as a failure, so those
# searches were recorded as free and their budget reservations released.

EMPTY_BODY = {
    "search_metadata": {"id": "e", "status": "Success"},
    "error": "Google News hasn't returned any results for this query.",
}


def test_a_completed_empty_search_is_counted_as_billed(tmp_path: Path) -> None:
    client = make_client(always(200, EMPTY_BODY), tmp_path)
    response = client.search("google_news", q="nothing to find")
    assert response.searches_charged == 1
    assert client.log.searches_charged == 1
    assert client.log.empty_results == 1


def test_a_completed_empty_search_is_not_raised(tmp_path: Path) -> None:
    """Finding nothing is an answer, not a failure."""
    client = make_client(always(200, EMPTY_BODY), tmp_path)
    assert client.search("google_news", q="nothing to find").status == "Success"


def test_a_completed_empty_search_is_recorded(tmp_path: Path) -> None:
    """So an identical search is not paid for twice to be told the same thing."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=EMPTY_BODY)

    client = make_client(handler, tmp_path)
    client.search("google_news", q="nothing to find")
    second = client.search("google_news", q="nothing to find")
    assert calls == 1
    assert second.from_cache


def test_an_error_without_a_success_status_is_still_a_failure(tmp_path: Path) -> None:
    """Only SerpApi's own completion signal marks a search as billed."""
    client = make_client(always(200, {"error": "Unsupported location"}), tmp_path)
    with pytest.raises(SerpApiError):
        client.search("google", q="x")
    assert client.log.searches_charged == 0


def test_an_error_status_is_still_a_failure(tmp_path: Path) -> None:
    body = {"search_metadata": {"status": "Error", "error": "backend failure"}}
    client = make_client(always(200, body), tmp_path)
    with pytest.raises(SerpApiError):
        client.search("google", q="x")
    assert client.log.searches_charged == 0


def test_a_failed_search_is_still_not_recorded(tmp_path: Path) -> None:
    """The original rule stands for genuine failures: a cached failure is permanent."""
    cache = ResponseCache(tmp_path / "cache")
    client = SerpApiClient(
        api_key="k",
        cache=cache,
        http=httpx.Client(transport=httpx.MockTransport(always(200, {"error": "bad param"}))),
        sleep=lambda _s: None,
    )
    with pytest.raises(SerpApiError):
        client.search("google", q="x")
    assert list(cache.directory.rglob("*.json")) == []


def test_an_invalid_key_is_still_an_auth_error_even_with_a_success_status(
    tmp_path: Path,
) -> None:
    """Checked defensively: a credential problem must never be recorded as a result."""
    body = {"search_metadata": {"status": "Error"}, "error": "Invalid API key"}
    with pytest.raises(AuthenticationError):
        make_client(always(200, body), tmp_path).search("google", q="x")
