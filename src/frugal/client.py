"""HTTP transport for SerpApi.

The client is thin by design — planning decisions live in the planner, not here.
Its one job is to turn a request into a recorded response exactly once, and to
be honest about what that cost.

Two behaviours are worth knowing about. Cache hits never touch the network and
never charge, so :attr:`SearchResponse.searches_charged` is the number to trust
when reporting spend. And SerpApi reports most search-level problems with HTTP
200 plus an ``error`` field rather than a failure status, so a response that
looks fine to the transport still has to be inspected before it is recorded — a
recorded error would otherwise be served from cache forever.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from frugal.cache import CacheMode, ResponseCache
from frugal.config import resolve_api_key
from frugal.errors import (
    AuthenticationError,
    RateLimited,
    SerpApiError,
    TransportError,
)

DEFAULT_BASE_URL = "https://serpapi.com/search.json"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_ATTEMPTS = 4

#: Retried with backoff. 408 and 429 are transient by definition; 5xx are the
#: server's problem and usually clear. Everything else is a request that will
#: fail identically however many times it is sent.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: Capped so a long Retry-After cannot stall a plan indefinitely.
_MAX_BACKOFF_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """One SerpApi response, with the metadata needed to account for it."""

    engine: str
    params: dict[str, Any]
    raw: dict[str, Any]
    from_cache: bool
    elapsed_ms: float
    searches_charged: int

    @property
    def status(self) -> str | None:
        """SerpApi's own status string, when it supplied one."""
        metadata = self.raw.get("search_metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get("status")
            return str(value) if value is not None else None
        return None

    @property
    def search_id(self) -> str | None:
        """SerpApi's archive id for this search, useful when reporting a problem."""
        metadata = self.raw.get("search_metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get("id")
            return str(value) if value is not None else None
        return None

    def top_level_keys(self) -> list[str]:
        """Sorted top-level keys. Used when surveying an unfamiliar engine."""
        return sorted(self.raw)


@dataclass(slots=True)
class RequestLog:
    """Per-client counters, for reporting what a run actually cost."""

    requests: int = 0
    cache_hits: int = 0
    retries: int = 0
    searches_charged: int = 0
    total_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "searches_charged": self.searches_charged,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "errors": list(self.errors),
        }


def _looks_like_auth_failure(message: str) -> bool:
    """Whether an error message describes a credential problem.

    SerpApi returns 401 for a missing key but 200 with an error string for an
    invalid one, so the text has to be inspected to avoid retrying a key that
    will never work.
    """
    lowered = message.lower()
    return "api key" in lowered or "invalid key" in lowered or "unauthor" in lowered


def _extract_error(payload: Mapping[str, Any]) -> str | None:
    """Return SerpApi's error message from a 200 response, if there is one."""
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    if isinstance(error, list) and error:
        return "; ".join(str(item) for item in error)

    metadata = payload.get("search_metadata")
    if isinstance(metadata, Mapping) and str(metadata.get("status", "")).lower() == "error":
        return str(metadata.get("error") or "search failed")
    return None


class SerpApiClient:
    """Issues SerpApi searches, resolving them from cache where possible.

    Usable as a context manager. When no ``http`` client is supplied one is
    created and owned; an injected client is left for its owner to close, which
    is what lets tests drive this with a mock transport and no network.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache: ResponseCache | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        http: httpx.Client | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.cache = cache if cache is not None else ResponseCache(".frugal-cache")
        self.base_url = base_url
        self.max_attempts = max(1, max_attempts)
        self.log = RequestLog()
        self._sleep = sleep
        self._owns_http = http is None
        self._http = http if http is not None else httpx.Client(timeout=timeout)

        # Resolved eagerly so a misconfigured key fails at construction, where
        # the traceback still points at the caller, rather than mid-plan.
        # REPLAY never reaches the network, so it must work without one.
        if api_key is None and self.cache.mode is CacheMode.REPLAY:
            self._api_key = ""
        else:
            self._api_key = resolve_api_key(api_key)

    def search(self, engine: str, **params: Any) -> SearchResponse:
        """Run one search, serving from cache when possible."""
        engine = engine.strip().lower()
        request_params = {k: v for k, v in params.items() if v is not None}

        cached = self.cache.load(engine, request_params)
        if cached is not None:
            self.log.cache_hits += 1
            return SearchResponse(
                engine=engine,
                params=dict(cached.params),
                raw=dict(cached.response),
                from_cache=True,
                elapsed_ms=cached.elapsed_ms,
                searches_charged=0,
            )

        started = time.perf_counter()
        payload = self._fetch(engine, request_params)
        elapsed_ms = (time.perf_counter() - started) * 1000

        # Inspected before recording. Caching an error would serve it forever.
        message = _extract_error(payload)
        if message is not None:
            self.log.errors.append(f"{engine}: {message}")
            if _looks_like_auth_failure(message):
                raise AuthenticationError(message)
            raise SerpApiError(engine, message, status=self._status_of(payload))

        self.cache.store(engine, request_params, payload, elapsed_ms=elapsed_ms)
        self.log.requests += 1
        self.log.searches_charged += 1
        self.log.total_latency_ms += elapsed_ms

        return SearchResponse(
            engine=engine,
            params=dict(request_params),
            raw=payload,
            from_cache=False,
            elapsed_ms=elapsed_ms,
            searches_charged=1,
        )

    def cached(self, engine: str, **params: Any) -> SearchResponse | None:
        """Return this search from cache, or ``None`` if it is not recorded.

        Never touches the network and never charges. Lets a caller establish
        that a search is free before reserving budget for it.
        """
        engine = engine.strip().lower()
        request_params = {k: v for k, v in params.items() if v is not None}
        entry = self.cache.peek(engine, request_params)
        if entry is None:
            return None
        return SearchResponse(
            engine=engine,
            params=dict(entry.params),
            raw=dict(entry.response),
            from_cache=True,
            elapsed_ms=entry.elapsed_ms,
            searches_charged=0,
        )

    @staticmethod
    def _status_of(payload: Mapping[str, Any]) -> str | None:
        metadata = payload.get("search_metadata")
        if isinstance(metadata, Mapping) and metadata.get("status") is not None:
            return str(metadata["status"])
        return None

    def _fetch(self, engine: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Perform the HTTP request, retrying transient failures."""
        query = {**params, "engine": engine, "api_key": self._api_key, "output": "json"}
        last_error: Exception | None = None
        retry_after: float | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._http.get(self.base_url, params=query)
            except httpx.TimeoutException as exc:
                last_error = exc
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code == 401:
                    raise AuthenticationError(
                        "SerpApi rejected the key (401). Check SERPAPI_API_KEY in your .env."
                    )

                if response.status_code in _RETRYABLE_STATUS:
                    retry_after = parse_retry_after(response.headers.get("retry-after"))
                    last_error = TransportError(f"HTTP {response.status_code}")
                elif response.is_error:
                    # A 4xx that is not 401 or 408/429 will fail identically on
                    # every retry; failing now keeps the message readable.
                    raise TransportError(
                        f"SerpApi returned HTTP {response.status_code} for engine {engine!r}: "
                        f"{response.text[:200]}"
                    )
                else:
                    return self._decode(response, engine)

            if attempt < self.max_attempts:
                self.log.retries += 1
                self._sleep(backoff_seconds(attempt, retry_after))

        if isinstance(last_error, TransportError) and "429" in str(last_error):
            raise RateLimited(self.max_attempts, retry_after)
        raise TransportError(
            f"SerpApi request for engine {engine!r} failed after {self.max_attempts} attempts: "
            f"{last_error}"
        ) from last_error

    @staticmethod
    def _decode(response: httpx.Response, engine: str) -> dict[str, Any]:
        """Decode a successful response, rejecting anything that is not an object."""
        try:
            payload = response.json()
        except ValueError as exc:
            raise TransportError(
                f"SerpApi returned a non-JSON body for engine {engine!r}: {response.text[:200]}"
            ) from exc
        if not isinstance(payload, dict):
            raise TransportError(
                f"SerpApi returned {type(payload).__name__} for engine {engine!r}, "
                f"expected an object"
            )
        return payload

    def close(self) -> None:
        """Close the HTTP client, if this instance created it."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> SerpApiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"SerpApiClient(base_url={self.base_url!r}, "
            f"charged={self.log.searches_charged}, cache_hits={self.log.cache_hits})"
        )


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header expressed in seconds.

    The HTTP-date form is ignored deliberately: it is rare from this API, and a
    clock-skewed date could produce a nonsensical wait.
    """
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def backoff_seconds(attempt: int, retry_after: float | None = None) -> float:
    """Exponential backoff with jitter, honouring Retry-After when it is shorter.

    Jitter matters when a plan fans out across engines concurrently: without it,
    every worker throttled at the same moment retries at the same moment.
    """
    if retry_after is not None:
        return min(retry_after, _MAX_BACKOFF_SECONDS)
    base = min(2.0 ** (attempt - 1), _MAX_BACKOFF_SECONDS)
    return base * (0.5 + random.random() * 0.5)
