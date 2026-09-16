"""Content-addressed record/replay cache for SerpApi responses.

Frugal charges itself for every search it issues, so the cache is not an
optimisation bolted on afterwards — it is load-bearing. Three properties matter:

*Each unique search is paid for exactly once.* Entries are addressed by a hash of
the engine plus its canonicalised parameters, so an identical search issued from
anywhere in a plan resolves to the same entry.

*Benchmarks re-run at zero cost.* :attr:`CacheMode.REPLAY` serves only from disk
and raises :class:`~frugal.errors.CacheMiss` rather than silently reaching for
the network, so a recorded fixture set either covers a workload completely or
fails loudly. Committed fixtures let anyone reproduce a result table without an
API key of their own.

*Entries are inspectable and diffable.* Records are plain JSON with the response
verbatim under ``response``, so a cache directory can be read, grepped and
version-controlled.

Credentials never reach the cache: they are stripped both from the key
derivation and from the stored record.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from frugal.errors import CacheCorrupt, CacheMiss

CACHE_FORMAT_VERSION = 1

#: Parameter names stripped before hashing and before writing to disk. Matched
#: case-insensitively. SerpApi accepts the key as ``api_key``; the others guard
#: against wrappers that pass a credential under a different name.
_REDACTED_PARAMS = frozenset(
    {"api_key", "apikey", "serp_api_key", "secret_api_key", "token", "access_token"}
)

_REDACTION_PLACEHOLDER = "<redacted>"

#: Default freshness windows in seconds, by engine. Volatility differs by orders
#: of magnitude across engines — a news result goes stale in minutes, a patent
#: record does not move for weeks — and a single global TTL would either serve
#: stale prices or re-buy stable scholarly results on every run.
DEFAULT_TTLS: Mapping[str, float] = {
    "google_news": 15 * 60,
    "google_finance": 15 * 60,
    "google_flights": 30 * 60,
    "google_hotels": 30 * 60,
    "google_trends": 60 * 60,
    "google_shopping": 60 * 60,
    "google_jobs": 6 * 60 * 60,
    "google_maps": 24 * 60 * 60,
    "google": 6 * 60 * 60,
    "google_scholar": 7 * 24 * 60 * 60,
    "google_patents": 7 * 24 * 60 * 60,
}

#: Applied to any engine absent from :data:`DEFAULT_TTLS`.
FALLBACK_TTL = 24 * 60 * 60


class CacheMode(Enum):
    """How the cache resolves a lookup.

    :cvar AUTO: Serve fresh entries from disk; fetch and record on a miss or a
        stale entry. The normal development mode.
    :cvar REPLAY: Serve from disk only. A miss raises
        :class:`~frugal.errors.CacheMiss` and TTLs are ignored, because a fixture
        recorded for a benchmark does not expire. This is the mode that makes a
        published result table reproducible without credentials.
    :cvar RECORD: Always fetch, overwriting any existing entry. Used to refresh a
        fixture set deliberately.
    """

    AUTO = "auto"
    REPLAY = "replay"
    RECORD = "record"


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A recorded SerpApi response and the metadata describing its capture."""

    engine: str
    key: str
    params: dict[str, Any]
    response: dict[str, Any]
    fetched_at: datetime
    elapsed_ms: float
    searches_charged: int

    def age_seconds(self, *, now: datetime | None = None) -> float:
        """Seconds elapsed since capture.

        Clamped at zero: a fixture recorded on a machine whose clock ran ahead
        would otherwise report a negative age and read as perpetually fresh.
        """
        reference = now or datetime.now(UTC)
        return max(0.0, (reference - self.fetched_at).total_seconds())

    def is_fresh(self, ttl: float, *, now: datetime | None = None) -> bool:
        """Whether this entry is within ``ttl`` seconds of capture."""
        if ttl <= 0:
            return False
        return self.age_seconds(now=now) < ttl

    def to_record(self) -> dict[str, Any]:
        """Serialise to the on-disk envelope."""
        return {
            "frugal_cache_version": CACHE_FORMAT_VERSION,
            "engine": self.engine,
            "key": self.key,
            "params": self.params,
            "fetched_at": self.fetched_at.isoformat(),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "searches_charged": self.searches_charged,
            "response": self.response,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, path: Path) -> CacheEntry:
        """Rebuild from an on-disk envelope, validating required fields."""
        version = record.get("frugal_cache_version")
        if version != CACHE_FORMAT_VERSION:
            raise CacheCorrupt(
                str(path),
                f"unsupported format version {version!r} (expected {CACHE_FORMAT_VERSION})",
            )
        try:
            fetched_at = datetime.fromisoformat(str(record["fetched_at"]))
            entry = cls(
                engine=str(record["engine"]),
                key=str(record["key"]),
                params=dict(record["params"]),
                response=dict(record["response"]),
                fetched_at=fetched_at if fetched_at.tzinfo else fetched_at.replace(tzinfo=UTC),
                elapsed_ms=float(record.get("elapsed_ms", 0.0)),
                searches_charged=int(record.get("searches_charged", 1)),
            )
        except CacheCorrupt:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheCorrupt(str(path), f"{type(exc).__name__}: {exc}") from exc
        return entry


@dataclass(slots=True)
class CacheStats:
    """Running counters for one cache instance.

    ``searches_charged`` is the number that matters: it is real money, and the
    planner's budget governor reads it to decide whether a plan may continue.
    """

    hits: int = 0
    misses: int = 0
    stale: int = 0
    writes: int = 0
    corrupt: int = 0
    searches_charged: int = 0

    @property
    def lookups(self) -> int:
        """Total resolved lookups."""
        return self.hits + self.misses + self.stale

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from disk, or 0.0 when nothing was looked up."""
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Flatten for logging and benchmark reports."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stale": self.stale,
            "writes": self.writes,
            "corrupt": self.corrupt,
            "searches_charged": self.searches_charged,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 4),
        }


def canonicalise_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce request parameters to a form two equivalent calls always share.

    Credentials are dropped, ``None`` values are dropped (SerpApi omits them
    anyway, so passing ``num=None`` must not produce a different key from
    omitting ``num``), keys are lowercased and sorted, and values are coerced to
    JSON scalars. Without this, ``{"q": "x", "hl": "en"}`` and
    ``{"hl": "en", "q": "x"}`` would be billed as two distinct searches.
    """
    canonical: dict[str, Any] = {}
    for raw_name, value in params.items():
        name = str(raw_name).strip().lower()
        if name in _REDACTED_PARAMS or value is None:
            continue
        canonical[name] = _coerce_value(value, path=name)
    return dict(sorted(canonical.items()))


def _coerce_value(value: Any, *, path: str) -> Any:
    """Coerce a parameter value to something JSON-serialisable and stable."""
    if isinstance(value, bool):
        # Checked before int: bool is a subclass of int, and SerpApi expects the
        # lowercase string form rather than 0/1.
        return "true" if value else "false"
    if isinstance(value, str | int | float):
        return value
    if isinstance(value, Mapping):
        return {str(k): _coerce_value(v, path=f"{path}.{k}") for k, v in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_coerce_value(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    raise TypeError(
        f"parameter {path!r} has type {type(value).__name__}, which cannot be part of a "
        f"cache key; pass a string, number, bool, list or mapping"
    )


def redact_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``params`` with credential values replaced by a placeholder.

    Used for the stored record and for log lines. Unlike
    :func:`canonicalise_params` this keeps the key present, so a reader can see
    that a credential was supplied without learning its value.
    """
    return {
        str(name): (
            _REDACTION_PLACEHOLDER if str(name).strip().lower() in _REDACTED_PARAMS else value
        )
        for name, value in params.items()
    }


def compute_key(engine: str, params: Mapping[str, Any]) -> str:
    """Derive the content address for a search.

    The engine is included in the hashed payload rather than relied upon as a
    directory name alone, so two engines that accept identical parameters can
    never collide even if entries are moved between directories.
    """
    payload = json.dumps(
        {"engine": engine.strip().lower(), "params": canonicalise_params(params)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Reads and writes recorded SerpApi responses beneath a directory.

    The cache resolves lookups but never performs I/O against SerpApi itself;
    the transport layer supplies responses to :meth:`store`. That split keeps
    this class trivially testable without a network or a key.

    Instances are safe to share across threads. Writes are atomic — a record is
    written to a temporary file in the destination directory and moved into
    place with :func:`os.replace` — so an interrupted run leaves either the old
    entry or the new one, never a truncated file.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        mode: CacheMode = CacheMode.AUTO,
        ttls: Mapping[str, float] | None = None,
        fallback_ttl: float = FALLBACK_TTL,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.mode = mode
        self._ttls = dict(DEFAULT_TTLS if ttls is None else ttls)
        self.fallback_ttl = fallback_ttl
        self.stats = CacheStats()
        self._stats_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    def ttl_for(self, engine: str) -> float:
        """Freshness window for ``engine``, falling back to :attr:`fallback_ttl`."""
        return self._ttls.get(engine.strip().lower(), self.fallback_ttl)

    def path_for(self, engine: str, key: str) -> Path:
        """Location of the entry for ``key``.

        Entries are sharded by the first two characters of the digest. A flat
        directory degrades badly on most filesystems once a benchmark has
        recorded tens of thousands of fixtures.
        """
        safe_engine = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in engine.strip().lower()
        )
        return self.directory / (safe_engine or "unknown") / key[:2] / f"{key}.json"

    def lock_for(self, key: str) -> threading.Lock:
        """Per-key lock, so concurrent plans touching one search serialise."""
        with self._key_locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def load(self, engine: str, params: Mapping[str, Any]) -> CacheEntry | None:
        """Return a usable entry for this search, or ``None`` if there is none.

        Honours :attr:`mode`: ``REPLAY`` ignores TTLs and raises on a miss,
        ``RECORD`` always reports a miss so the caller refetches. A corrupt entry
        is counted and treated as a miss under AUTO and RECORD — a bad file
        should cost one search to repair, not abort a long run — but raises under
        REPLAY, where silently skipping it would corrupt a published result.
        """
        key = compute_key(engine, params)

        if self.mode is CacheMode.RECORD:
            self._bump("misses")
            return None

        path = self.path_for(engine, key)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._bump("misses")
            if self.mode is CacheMode.REPLAY:
                raise CacheMiss(engine, key) from None
            return None
        except OSError as exc:
            # Unreadable for some other reason (permissions, a truncated mount).
            # Treat as corrupt rather than crashing a long-running plan.
            self._bump("corrupt")
            if self.mode is CacheMode.REPLAY:
                raise CacheCorrupt(str(path), f"{type(exc).__name__}: {exc}") from exc
            return None

        try:
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise CacheCorrupt(
                    str(path), f"expected a JSON object, found {type(record).__name__}"
                )
            entry = CacheEntry.from_record(record, path=path)
        except json.JSONDecodeError as exc:
            self._bump("corrupt")
            if self.mode is CacheMode.REPLAY:
                raise CacheCorrupt(str(path), f"invalid JSON: {exc}") from exc
            return None
        except CacheCorrupt:
            self._bump("corrupt")
            if self.mode is CacheMode.REPLAY:
                raise
            return None

        if self.mode is CacheMode.REPLAY:
            # Fixtures do not expire. A benchmark recorded last week must produce
            # the same numbers today or it is not a benchmark.
            self._bump("hits")
            return entry

        if not entry.is_fresh(self.ttl_for(engine)):
            self._bump("stale")
            return None

        self._bump("hits")
        return entry

    def peek(self, engine: str, params: Mapping[str, Any]) -> CacheEntry | None:
        """Look up an entry without recording a hit or a miss.

        Callers that need to know whether a search is free *before* committing
        to it use this. Going through :meth:`load` would count the lookup twice
        once the real fetch follows, and the hit rate is a reported number.
        """
        saved = self.stats
        try:
            self.stats = CacheStats()
            return self.load(engine, params)
        finally:
            self.stats = saved

    def store(
        self,
        engine: str,
        params: Mapping[str, Any],
        response: Mapping[str, Any],
        *,
        elapsed_ms: float = 0.0,
        searches_charged: int = 1,
        fetched_at: datetime | None = None,
    ) -> CacheEntry:
        """Record a freshly fetched response and return the stored entry."""
        key = compute_key(engine, params)
        entry = CacheEntry(
            engine=engine.strip().lower(),
            key=key,
            params=redact_params(params),
            response=dict(response),
            fetched_at=fetched_at or datetime.now(UTC),
            elapsed_ms=elapsed_ms,
            searches_charged=searches_charged,
        )
        self._write(entry)
        with self._stats_lock:
            self.stats.writes += 1
            self.stats.searches_charged += searches_charged
        return entry

    def _write(self, entry: CacheEntry) -> None:
        """Atomically persist ``entry``."""
        path = self.path_for(entry.engine, entry.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(entry.to_record(), indent=2, sort_keys=True, ensure_ascii=False)

        # Written into the destination directory so os.replace stays on one
        # filesystem; a cross-device rename is not atomic.
        handle, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{entry.key[:8]}-", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def iter_entries(self) -> Iterator[CacheEntry]:
        """Yield every readable entry, skipping corrupt files.

        Used by the fixture-set inspection commands; a corrupt file should not
        stop an audit of the other ten thousand.
        """
        for path in sorted(self.directory.rglob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict):
                    yield CacheEntry.from_record(record, path=path)
            except (OSError, json.JSONDecodeError, CacheCorrupt):
                continue

    def _bump(self, counter: str) -> None:
        with self._stats_lock:
            setattr(self.stats, counter, getattr(self.stats, counter) + 1)

    def __repr__(self) -> str:
        return (
            f"ResponseCache(directory={str(self.directory)!r}, mode={self.mode.value!r}, "
            f"entries_written={self.stats.writes})"
        )
