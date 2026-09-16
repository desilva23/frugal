"""One evidence model across heterogeneous engines.

Surveying eight SerpApi engines produced the requirement for this module. Two
findings drove the design:

*``title`` is the only field every engine supplies.* Not ``link`` — ``google_jobs``
has no link field at all, carrying ``apply_options`` and ``share_link`` instead,
and ``google_maps`` carries ``website``, which many local results lack entirely. A
normaliser that assumes ``title``/``link``/``snippet`` silently drops both engines,
and silence is the problem: the caller sees an empty evidence set rather than an
error.

*Not every engine returns documents.* ``google_trends`` returns a time series.
There is no list of documents to normalise, so evidence has two kinds —
:class:`Document` and :class:`Series` — and a planner that can reach for the
second is doing something a link-shaped model cannot express.

Engine-specific fields are preserved on :attr:`Document.extra` rather than
discarded. Normalisation here means giving every result a common surface, not
flattening away what makes an engine worth calling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

#: Query parameters that identify a marketing campaign rather than a document.
#: Two URLs differing only in these point at the same page, and treating them as
#: distinct results would inflate every recall measurement in the benchmark.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_source_platform", "gclid", "fbclid", "msclkid", "dclid",
        "mc_cid", "mc_eid", "igshid", "ref", "ref_src", "referrer", "source",
        "_ga", "_gl", "yclid", "wbraid", "gbraid", "si",
    }
)

_INDEX_PATTERN = re.compile(r"^(?P<name>[^\[\]]*)\[(?P<index>-?\d+)\]$")


def dig(payload: Any, path: str, default: Any = None) -> Any:
    """Read a nested value by dotted path, returning ``default`` if it is absent.

    Supports list indexing: ``"apply_options[0].link"``. Every engine buries at
    least one field the normaliser needs — Scholar's link lives under
    ``inline_links``, Jobs' under ``apply_options`` — and this keeps the adapters
    declarative rather than a thicket of nested ``if`` statements.

    Never raises for a missing or wrongly-typed segment. Adapters run against
    live payloads from an API that adds and removes fields without notice, and a
    :class:`TypeError` from deep inside a parse is not useful to anyone.
    """
    current = payload
    for segment in path.split("."):
        if current is None:
            return default

        match = _INDEX_PATTERN.match(segment)
        if match:
            name = match.group("name")
            if name:
                current = current.get(name) if isinstance(current, dict) else None
            if not isinstance(current, (list, tuple)):
                return default
            index = int(match.group("index"))
            try:
                current = current[index]
            except IndexError:
                return default
            continue

        if isinstance(current, dict):
            current = current.get(segment)
        else:
            return default

    return default if current is None else current


def first_present(payload: Any, paths: tuple[str, ...], default: Any = None) -> Any:
    """Return the first path in ``paths`` that yields a value.

    Adapters declare an ordered list of candidate locations rather than one, so
    that an engine offering several kinds of link — Jobs offers three, meaning
    different things — resolves to the most useful one available.
    """
    for path in paths:
        value = dig(payload, path)
        if value not in (None, "", [], {}):
            return value
    return default


def canonical_url(url: str | None) -> str | None:
    """Reduce a URL to a form two links to the same page will share.

    Lowercases the scheme and host, drops ``www.`` and the fragment, strips
    campaign parameters, sorts what remains, and removes a trailing slash. This
    is what deduplication compares, so an over-eager rule here would merge
    distinct pages and an under-eager one would let the same page count twice
    across engines.

    Returns ``None`` for anything that is not an absolute http(s) URL, which is
    the honest answer for a relative path or a ``mailto:``.
    """
    if not url or not isinstance(url, str):
        return None

    candidate = url.strip()
    if not candidate:
        return None

    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None

    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return None

    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    # Default ports carry no meaning and appear inconsistently.
    if host.endswith(":80") or host.endswith(":443"):
        host = host.rsplit(":", 1)[0]

    kept = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if name.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))

    path = parts.path.rstrip("/") or "/"

    return urlunsplit((parts.scheme.lower(), host, path, query, ""))


def parse_timestamp(value: Any) -> datetime | None:
    """Best-effort conversion of the several date forms SerpApi returns.

    News supplies a parsed ``iso_date``, Patents supplies ``YYYY-MM-DD``, Trends
    supplies a Unix timestamp as a string, and several engines supply human text
    like ``"2 hours ago"`` that carries no absolute date at all. Unparseable
    input yields ``None`` rather than a guess: a wrong date is worse than a
    missing one when the planner is reasoning about freshness.
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return _from_unix(float(value))

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if text.isdigit() and len(text) >= 9:
        return _from_unix(float(text))

    iso_candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
    except ValueError:
        pass
    else:
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue

    return None


def _from_unix(value: float) -> datetime | None:
    """Convert a Unix timestamp, rejecting values outside a plausible range."""
    # Guards against a millisecond timestamp or a stray integer field being read
    # as a date in the year 55000.
    if not 0 < value < 4_102_444_800:  # up to the year 2100
        return None
    try:
        return datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where one piece of evidence came from.

    Carried on every result so that an answer can cite not just a source but the
    search that surfaced it — which engine, which reformulation, which rank. The
    benchmark reads ``engine`` and ``query`` to attribute a contribution to the
    routing decision that produced it.
    """

    engine: str
    query: str
    position: int
    retrieved_at: datetime
    search_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "query": self.query,
            "position": self.position,
            "retrieved_at": self.retrieved_at.isoformat(),
            "search_id": self.search_id,
        }


@dataclass(frozen=True, slots=True)
class Document:
    """A retrievable item, whatever engine produced it.

    ``title`` is the only field guaranteed present, because it is the only one
    every surveyed engine supplies. ``url`` is optional and genuinely so: a local
    result without a website has no URL, and inventing one would be worse than
    admitting it.
    """

    title: str
    provenance: Provenance
    url: str | None = None
    snippet: str | None = None
    source: str | None = None
    published_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        """The key deduplication compares.

        Falls back to the normalised title when there is no URL, so that results
        without links still collapse rather than each counting as new evidence.
        """
        canonical = canonical_url(self.url)
        if canonical:
            return canonical
        return f"{self.provenance.engine}:{_normalise_title(self.title)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "document",
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "provenance": self.provenance.as_dict(),
            "extra": self.extra,
        }


@dataclass(frozen=True, slots=True)
class SeriesPoint:
    """One observation in a time series."""

    timestamp: datetime
    value: float
    label: str | None = None


@dataclass(frozen=True, slots=True)
class Series:
    """A time series returned as evidence.

    Exists because ``google_trends`` answers questions no list of links can.
    "Interest in this doubled over six months" is a different kind of fact from
    ten more pages, and a planner able to reach for it can answer questions that
    document retrieval cannot.
    """

    name: str
    points: tuple[SeriesPoint, ...]
    provenance: Provenance

    @property
    def identity(self) -> str:
        return f"series:{self.provenance.engine}:{_normalise_title(self.name)}"

    @property
    def span(self) -> tuple[datetime, datetime] | None:
        """First and last observation times, or ``None`` when empty."""
        if not self.points:
            return None
        return self.points[0].timestamp, self.points[-1].timestamp

    def trend(self) -> float | None:
        """Relative change from the first observation to the last.

        ``0.5`` means the value rose by half. ``None`` when there are too few
        points, or when the first value is zero and the ratio is undefined.
        """
        if len(self.points) < 2:
            return None
        first, last = self.points[0].value, self.points[-1].value
        if first == 0:
            return None
        return (last - first) / abs(first)

    def summarise(self) -> str:
        """A one-line description an LLM can consume alongside document snippets."""
        if not self.points:
            return f"{self.name}: no observations"
        values = [p.value for p in self.points]
        change = self.trend()
        direction = (
            "flat"
            if change is None or abs(change) < 0.05
            else ("up" if change > 0 else "down")
        )
        detail = f" ({change:+.0%})" if change is not None else ""
        return (
            f"{self.name}: {len(self.points)} observations, "
            f"min {min(values):g}, max {max(values):g}, trending {direction}{detail}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "series",
            "name": self.name,
            "points": [
                {"timestamp": p.timestamp.isoformat(), "value": p.value, "label": p.label}
                for p in self.points
            ],
            "provenance": self.provenance.as_dict(),
            "summary": self.summarise(),
        }


#: Anything an engine can contribute to a plan's evidence set.
Evidence = Document | Series


def _normalise_title(title: str) -> str:
    """Collapse a title to a comparable form for identity fallback."""
    return re.sub(r"\s+", " ", title).strip().casefold()


def deduplicate(evidence: list[Evidence]) -> list[Evidence]:
    """Collapse results that point at the same thing, keeping the first seen.

    Order is preserved, so the earliest-ranked occurrence survives. This runs
    across engines, not only within one: the same page surfacing in both web
    search and news is one piece of evidence, and counting it twice is exactly
    the redundancy the planner exists to eliminate.
    """
    seen: set[str] = set()
    unique: list[Evidence] = []
    for item in evidence:
        key = item.identity
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
