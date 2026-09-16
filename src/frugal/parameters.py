"""Engine-specific request parameters.

Until this module existed, every engine received the same two parameters — the
query and a page size — and that was a correctness bug rather than a missing
feature. Asked whether interest in electric vehicles *in India* was rising,
Frugal called ``google_trends`` with no ``geo``, which answers for the world.
The series it returned was real, well-formed, and about the wrong place.

SerpApi's value is structured access to engines that each take different
parameters, and treating them all as a generic query box throws that away. A
jobs search without ``location`` returns listings from anywhere; a shopping
search without ``gl`` prices in the wrong currency; a trends search without
``geo`` answers a question nobody asked.

Place detection works from a curated gazetteer rather than a general named-entity
model, for the same reason the router is lexical: a model here would make the
benchmark unrepeatable. The list is small and deliberately weighted towards
India, and extending it is a data change rather than a code change. Its limits
are real — an unlisted town is simply not detected, and the query still carries
the place name, so the engine is no worse off than before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Countries, mapped to the two-letter code SerpApi expects for ``gl``/``geo``.
COUNTRIES: dict[str, str] = {
    "india": "IN",
    "united states": "US",
    "usa": "US",
    "america": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "britain": "GB",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "japan": "JP",
    "china": "CN",
    "singapore": "SG",
    "brazil": "BR",
}

#: Indian cities, mapped to the fuller place string SerpApi's ``location``
#: parameter matches against. Weighted towards India because this is where the
#: questions come from; other countries resolve at country level only.
CITIES: dict[str, tuple[str, str]] = {
    "chennai": ("Chennai, Tamil Nadu, India", "IN"),
    "bengaluru": ("Bengaluru, Karnataka, India", "IN"),
    "bangalore": ("Bengaluru, Karnataka, India", "IN"),
    "mumbai": ("Mumbai, Maharashtra, India", "IN"),
    "delhi": ("Delhi, India", "IN"),
    "new delhi": ("New Delhi, Delhi, India", "IN"),
    "hyderabad": ("Hyderabad, Telangana, India", "IN"),
    "kolkata": ("Kolkata, West Bengal, India", "IN"),
    "pune": ("Pune, Maharashtra, India", "IN"),
    "ahmedabad": ("Ahmedabad, Gujarat, India", "IN"),
    "jaipur": ("Jaipur, Rajasthan, India", "IN"),
    "kochi": ("Kochi, Kerala, India", "IN"),
    "thiruvananthapuram": ("Thiruvananthapuram, Kerala, India", "IN"),
    "trivandrum": ("Thiruvananthapuram, Kerala, India", "IN"),
    "coimbatore": ("Coimbatore, Tamil Nadu, India", "IN"),
    "kerala": ("Kerala, India", "IN"),
    "tamil nadu": ("Tamil Nadu, India", "IN"),
    "karnataka": ("Karnataka, India", "IN"),
    "maharashtra": ("Maharashtra, India", "IN"),
}

#: Recency words that justify narrowing a scholarly search to recent years.
_RECENT = frozenset({"recent", "recently", "latest", "current", "new", "newest"})

#: How far back "recent" reaches for Scholar. Broad on purpose: a tighter window
#: silently hides the foundational paper a question is often really about.
_SCHOLAR_RECENT_YEARS = 5


@dataclass(frozen=True, slots=True)
class Locale:
    """A place a question is about."""

    #: Full place string for SerpApi's ``location`` parameter, when known.
    location: str | None
    #: Two-letter country code for ``gl`` and ``geo``.
    country: str
    #: What in the question produced this, for the plan trace.
    matched: str

    def describe(self) -> str:
        return f"{self.location or self.country} (from {self.matched!r})"


def detect_locale(question: str) -> Locale | None:
    """Find the place a question is about, or ``None``.

    Cities are checked before countries so that "jobs in Chennai, India" resolves
    to the city rather than stopping at the country, which is the more specific
    and therefore more useful answer.
    """
    lowered = question.lower()

    for name, (location, country) in sorted(CITIES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return Locale(location=location, country=country, matched=name)

    for name, code in sorted(COUNTRIES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return Locale(location=None, country=code, matched=name)

    return None


def build_params(
    engine: str,
    query: str,
    *,
    locale: Locale | None = None,
    results: int = 10,
    recent: bool = False,
    current_year: int | None = None,
) -> dict[str, Any]:
    """Build the request parameters for one engine.

    Each engine gets what it can actually use. Passing ``num`` to an engine that
    ignores it is harmless; failing to pass ``geo`` to one that needs it is not.
    """
    engine = engine.strip().lower()
    params: dict[str, Any] = {"q": query}

    if engine == "google_trends":
        # Without data_type the engine's default is relied upon, and without geo
        # it answers for the world. Both are how the wrong-country bug happened.
        params["data_type"] = "TIMESERIES"
        if locale is not None:
            params["geo"] = locale.country
        return params

    if engine == "google_patents":
        # Patents is jurisdictional rather than locale-sensitive, and a gl here
        # would narrow results without being asked to.
        params["num"] = results
        return params

    params["num"] = results

    if engine == "google_scholar":
        if recent and current_year is not None:
            params["as_ylo"] = current_year - _SCHOLAR_RECENT_YEARS
        return params

    if locale is not None:
        params["gl"] = locale.country.lower()
        params["hl"] = "en"

        # location is the parameter these two engines are built around: a jobs
        # search without it returns listings from anywhere at all.
        if engine in {"google_jobs", "google_maps"} and locale.location:
            params["location"] = locale.location

    return params


def wants_recent(question: str) -> bool:
    """Whether a question asks for current work rather than the canonical work."""
    lowered = question.lower()
    return any(re.search(rf"\b{word}\b", lowered) for word in _RECENT)
