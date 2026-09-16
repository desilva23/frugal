"""Per-engine normalisation into the common evidence model.

Each adapter declares where its engine keeps the fields Frugal needs, rather
than one mapper guessing across all of them. The survey showed why: ``title`` is
the only field common to every engine, so a single mapping would have to be so
permissive that it could not tell a renamed field from an absent one.

The declarative form also makes drift detectable. An adapter states which result
key it expects and whether its results normally carry a URL, so a batch that
arrives without either is an error rather than an empty list — see
:class:`~frugal.errors.SchemaDrift`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from frugal.errors import SchemaDrift
from frugal.schema import (
    Document,
    Evidence,
    Provenance,
    Series,
    SeriesPoint,
    first_present,
    parse_timestamp,
)

#: Below this fraction of results carrying a resolvable URL, an engine that
#: normally supplies links is assumed to have changed shape. Tolerant enough to
#: absorb the occasional result without one, strict enough to catch a rename.
_URL_DRIFT_THRESHOLD = 0.5


def _as_text(value: Any) -> str | None:
    """Flatten a field that an engine may return as a string or an object.

    ``source`` is a plain string on web search and an object with ``name`` on
    news, and either can appear as an engine evolves.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Mapping):
        for key in ("name", "title", "summary", "text"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        return None
    if isinstance(value, Sequence):
        parts = [_as_text(item) for item in value]
        joined = ", ".join(p for p in parts if p)
        return joined or None
    return None


@dataclass(frozen=True, slots=True)
class EngineAdapter:
    """Declares where one engine keeps each normalised field.

    :param result_keys: Candidate top-level keys holding the result list, in
        preference order. More than one because engines sometimes rename or
        duplicate this between response variants.
    :param url_paths: Candidate locations for a result's link, in preference
        order. Jobs offers three that mean different things; the order encodes
        which is most useful.
    :param url_optional: ``True`` where results legitimately lack a URL. Local
        results for a business with no website are ordinary, and treating that
        as drift would make the adapter cry wolf.
    """

    engine: str
    result_keys: tuple[str, ...]
    title_paths: tuple[str, ...] = ("title",)
    url_paths: tuple[str, ...] = ("link",)
    snippet_paths: tuple[str, ...] = ("snippet",)
    source_paths: tuple[str, ...] = ("source",)
    date_paths: tuple[str, ...] = ()
    extra_paths: tuple[str, ...] = ()
    url_optional: bool = False

    def parse(
        self,
        payload: Mapping[str, Any],
        *,
        query: str,
        retrieved_at: datetime | None = None,
        search_id: str | None = None,
    ) -> list[Evidence]:
        """Normalise one response into evidence."""
        retrieved_at = retrieved_at or datetime.now(UTC)
        results = self._locate_results(payload)

        documents: list[Document] = []
        for index, raw in enumerate(results):
            if not isinstance(raw, Mapping):
                continue
            document = self._parse_one(raw, index, query, retrieved_at, search_id)
            if document is not None:
                documents.append(document)

        self._check_for_drift(documents, len(results))
        return list(documents)

    def _locate_results(self, payload: Mapping[str, Any]) -> list[Any]:
        """Find the result list, or raise if no declared key holds one."""
        for key in self.result_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value

        # An empty result set is a legitimate answer — a query nobody has written
        # about returns nothing — and is distinguished from drift by SerpApi
        # having reported the search succeeded.
        if self._reported_success(payload):
            return []

        present = ", ".join(sorted(k for k in payload if not k.startswith("search_"))) or "nothing"
        raise SchemaDrift(
            self.engine,
            f"none of {list(self.result_keys)} holds a list; payload carries {present}",
        )

    @staticmethod
    def _reported_success(payload: Mapping[str, Any]) -> bool:
        metadata = payload.get("search_metadata")
        if not isinstance(metadata, Mapping):
            return False
        return str(metadata.get("status", "")).lower() == "success"

    def _parse_one(
        self,
        raw: Mapping[str, Any],
        index: int,
        query: str,
        retrieved_at: datetime,
        search_id: str | None,
    ) -> Document | None:
        title = _as_text(first_present(raw, self.title_paths))
        if not title:
            # Every surveyed engine supplies a title. One result without one is
            # malformed rather than informative, and a document with no title
            # cannot be presented to a reader or compared for identity.
            return None

        position = raw.get("position")
        return Document(
            title=title,
            url=_as_text(first_present(raw, self.url_paths)),
            snippet=_as_text(first_present(raw, self.snippet_paths)),
            source=_as_text(first_present(raw, self.source_paths)),
            published_at=parse_timestamp(first_present(raw, self.date_paths)),
            provenance=Provenance(
                engine=self.engine,
                query=query,
                position=int(position) if isinstance(position, int) else index + 1,
                retrieved_at=retrieved_at,
                search_id=search_id,
            ),
            extra={
                path: value
                for path in self.extra_paths
                if (value := first_present(raw, (path,))) is not None
            },
        )

    def _check_for_drift(self, documents: Sequence[Document], raw_count: int) -> None:
        """Raise when a whole batch lacks something this engine always supplies."""
        if not documents:
            if raw_count:
                raise SchemaDrift(
                    self.engine,
                    f"{raw_count} results present but none carried a title at "
                    f"{list(self.title_paths)}",
                )
            return

        if self.url_optional:
            return

        with_url = sum(1 for d in documents if d.url)
        if with_url / len(documents) < _URL_DRIFT_THRESHOLD:
            raise SchemaDrift(
                self.engine,
                f"only {with_url} of {len(documents)} results resolved a URL from "
                f"{list(self.url_paths)}",
            )


@dataclass(frozen=True, slots=True)
class TrendsAdapter:
    """Normalises ``google_trends`` into a :class:`~frugal.schema.Series`.

    Kept separate from :class:`EngineAdapter` because trends is not
    document-shaped: it returns observations over time under
    ``interest_over_time.timeline_data``, with no list of results to map.
    """

    engine: str = "google_trends"

    def parse(
        self,
        payload: Mapping[str, Any],
        *,
        query: str,
        retrieved_at: datetime | None = None,
        search_id: str | None = None,
    ) -> list[Evidence]:
        retrieved_at = retrieved_at or datetime.now(UTC)
        timeline = payload.get("interest_over_time")
        if not isinstance(timeline, Mapping):
            raise SchemaDrift(self.engine, "no interest_over_time object in payload")

        raw_points = timeline.get("timeline_data")
        if not isinstance(raw_points, list):
            raise SchemaDrift(self.engine, "interest_over_time carries no timeline_data list")

        # One series per query term: trends compares several at once, and each
        # term's observations share an index across every timeline entry.
        by_term: dict[str, list[SeriesPoint]] = {}
        for entry in raw_points:
            if not isinstance(entry, Mapping):
                continue
            timestamp = parse_timestamp(entry.get("timestamp")) or parse_timestamp(
                entry.get("date")
            )
            if timestamp is None:
                continue
            for value in entry.get("values") or []:
                if not isinstance(value, Mapping):
                    continue
                term = _as_text(value.get("query")) or query
                extracted = value.get("extracted_value")
                if not isinstance(extracted, (int, float)) or isinstance(extracted, bool):
                    continue
                by_term.setdefault(term, []).append(
                    SeriesPoint(timestamp=timestamp, value=float(extracted), label=term)
                )

        if not by_term:
            raise SchemaDrift(
                self.engine, f"{len(raw_points)} timeline entries yielded no usable observations"
            )

        return [
            Series(
                name=term,
                points=tuple(sorted(points, key=lambda p: p.timestamp)),
                provenance=Provenance(
                    engine=self.engine,
                    query=query,
                    position=index + 1,
                    retrieved_at=retrieved_at,
                    search_id=search_id,
                ),
            )
            for index, (term, points) in enumerate(by_term.items())
        ]


#: Field mappings captured from live responses by scripts/survey_engines.py.
#: The per-engine oddities below are all real, not defensive guesses.
ADAPTERS: dict[str, EngineAdapter | TrendsAdapter] = {
    "google": EngineAdapter(
        engine="google",
        result_keys=("organic_results",),
        source_paths=("source", "displayed_link"),
        extra_paths=("displayed_link", "snippet_highlighted_words"),
    ),
    "google_news": EngineAdapter(
        engine="google_news",
        result_keys=("news_results",),
        # News carries no snippet at all; the headline and source are the text.
        snippet_paths=(),
        source_paths=("source.name", "source"),
        date_paths=("iso_date", "date"),
        extra_paths=("thumbnail",),
    ),
    "google_scholar": EngineAdapter(
        engine="google_scholar",
        result_keys=("organic_results",),
        # A paywalled paper has no direct link; resources holds the PDF mirror.
        url_paths=("link", "resources[0].link"),
        source_paths=("publication_info.summary",),
        extra_paths=("result_id", "inline_links.cited_by.total", "publication_info.summary"),
    ),
    "google_shopping": EngineAdapter(
        engine="google_shopping",
        result_keys=("shopping_results",),
        url_paths=("product_link", "link"),
        # No snippet field exists; price and seller are what distinguishes a result.
        snippet_paths=(),
        extra_paths=("price", "extracted_price", "product_id", "rating", "reviews"),
    ),
    "google_jobs": EngineAdapter(
        engine="google_jobs",
        result_keys=("jobs_results",),
        title_paths=("title", "job_title"),
        # Jobs has no link field. apply_options is the useful one — it goes to the
        # application — while share_link and source_link go elsewhere.
        url_paths=("apply_options[0].link", "share_link", "source_link"),
        snippet_paths=("description",),
        source_paths=("company_name", "via"),
        extra_paths=("location", "company_name", "detected_extensions", "via"),
    ),
    "google_maps": EngineAdapter(
        engine="google_maps",
        result_keys=("local_results", "place_results"),
        # A business without a website is ordinary, not drift. There is no
        # fallback on purpose: photos_link points at a Google gallery, not at
        # the business, and citing it as the source for a claim would mislead.
        url_paths=("website",),
        snippet_paths=("address", "type"),
        source_paths=("type",),
        extra_paths=("address", "rating", "reviews", "phone", "gps_coordinates", "place_id"),
        url_optional=True,
    ),
    "google_patents": EngineAdapter(
        engine="google_patents",
        result_keys=("organic_results",),
        url_paths=("patent_link", "serpapi_link", "pdf"),
        source_paths=("assignee", "inventor"),
        # Patents carry four dates; publication is the one that dates the document.
        date_paths=("publication_date", "grant_date", "filing_date", "priority_date"),
        extra_paths=("patent_id", "inventor", "assignee", "priority_date", "grant_date"),
    ),
    "google_trends": TrendsAdapter(),
}


def normalise(
    engine: str,
    payload: Mapping[str, Any],
    *,
    query: str,
    retrieved_at: datetime | None = None,
    search_id: str | None = None,
) -> list[Evidence]:
    """Normalise a response from ``engine`` into evidence.

    Falls back to the web-search mapping for an engine with no adapter of its
    own. Most SerpApi engines follow the ``organic_results`` convention, so an
    unrecognised engine is more likely to work than to fail — and if it does not,
    :class:`~frugal.errors.SchemaDrift` says so rather than returning nothing.
    """
    key = engine.strip().lower()
    adapter = ADAPTERS.get(key)
    if adapter is None:
        adapter = EngineAdapter(engine=key, result_keys=("organic_results", "results"))
    return adapter.parse(payload, query=query, retrieved_at=retrieved_at, search_id=search_id)
