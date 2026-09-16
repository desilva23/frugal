"""Survey the response shape of each SerpApi engine Frugal routes across.

Run this once to capture what each engine actually returns, then design against
the captured shapes rather than against the documentation. Re-running is free:
responses are served from the cache, so this doubles as a drift check when an
engine changes its payload.

    python scripts/survey_engines.py            # survey, using cache where possible
    python scripts/survey_engines.py --refresh  # refetch everything (costs searches)

One engine failing does not stop the others, and failures are never cached, so a
failed engine can be retried later without having wasted anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frugal.cache import CacheMode, ResponseCache  # noqa: E402
from frugal.client import SerpApiClient  # noqa: E402
from frugal.errors import FrugalError  # noqa: E402

SURVEY_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "engine_survey.json"

#: One representative query per engine. Chosen to return a populated result set
#: — an empty one tells us nothing about the shape we have to normalise.
PROBES: list[tuple[str, dict[str, Any]]] = [
    ("google", {"q": "monsoon onset date India 2026", "gl": "in", "hl": "en"}),
    ("google_news", {"q": "Indian Railways", "gl": "in", "hl": "en"}),
    ("google_scholar", {"q": "retrieval augmented generation"}),
    ("google_shopping", {"q": "wireless earbuds", "gl": "in", "hl": "en"}),
    ("google_jobs", {"q": "python developer Chennai", "gl": "in", "hl": "en"}),
    ("google_maps", {"q": "coffee", "ll": "@13.0827,80.2707,14z", "type": "search"}),
    ("google_trends", {"q": "electric vehicles", "data_type": "TIMESERIES", "geo": "IN"}),
    ("google_patents", {"q": "solar panel efficiency"}),
]

#: Keys present on essentially every response; listing them per engine is noise.
_UNIVERSAL_KEYS = {"search_metadata", "search_parameters", "search_information",
                   "serpapi_pagination", "pagination"}


def describe(value: Any, *, depth: int = 0) -> str:
    """Render a compact type sketch of a decoded payload."""
    if isinstance(value, dict):
        if depth >= 1:
            return f"object({len(value)} keys)"
        inner = ", ".join(f"{k}: {describe(v, depth=depth + 1)}" for k, v in list(value.items())[:8])
        return f"{{{inner}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        return f"[{len(value)} x {describe(value[0], depth=depth + 1)}]"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    return "string"


def result_bearing_keys(payload: dict[str, Any]) -> list[tuple[str, int]]:
    """Top-level keys holding a non-empty list, with their lengths.

    This is the thing that differs most between engines: each one names its
    results something else — organic_results, news_results, shopping_results,
    local_results, jobs_results — and the normaliser has to know which.
    """
    found = [
        (key, len(value))
        for key, value in payload.items()
        if isinstance(value, list) and value and key not in _UNIVERSAL_KEYS
    ]
    return sorted(found, key=lambda pair: -pair[1])


def survey_engine(client: SerpApiClient, engine: str, params: dict[str, Any]) -> dict[str, Any]:
    """Probe one engine and reduce its response to a structural summary."""
    response = client.search(engine, **params)
    payload = response.raw
    bearing = result_bearing_keys(payload)

    record: dict[str, Any] = {
        "engine": engine,
        "ok": True,
        "from_cache": response.from_cache,
        "elapsed_ms": round(response.elapsed_ms, 1),
        "top_level_keys": [k for k in response.top_level_keys() if k not in _UNIVERSAL_KEYS],
        "result_keys": [{"key": k, "count": n} for k, n in bearing],
    }

    if bearing:
        primary_key = bearing[0][0]
        first = payload[primary_key][0]
        record["primary_key"] = primary_key
        record["item_fields"] = (
            {k: describe(v, depth=1) for k, v in first.items()}
            if isinstance(first, dict)
            else describe(first)
        )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="refetch every engine, ignoring the cache"
    )
    args = parser.parse_args(argv)

    mode = CacheMode.RECORD if args.refresh else CacheMode.AUTO
    cache = ResponseCache(".frugal-cache", mode=mode)
    records: list[dict[str, Any]] = []

    with SerpApiClient(cache=cache) as client:
        for engine, params in PROBES:
            try:
                record = survey_engine(client, engine, params)
            except FrugalError as exc:
                # Not cached, so this engine can be retried later for free.
                record = {"engine": engine, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            _print_record(record)

        print()
        print(f"searches charged this run: {client.log.searches_charged}")
        print(f"served from cache:         {client.log.cache_hits}")

    SURVEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SURVEY_PATH.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    print(f"survey written to {SURVEY_PATH.relative_to(Path.cwd())}")

    return 0 if all(r["ok"] for r in records) else 1


def _print_record(record: dict[str, Any]) -> None:
    engine = record["engine"]
    if not record["ok"]:
        print(f"\n{engine:<18} FAILED  {record['error']}")
        return

    source = "cache" if record["from_cache"] else "live"
    keys = ", ".join(f"{r['key']}({r['count']})" for r in record["result_keys"]) or "none"
    print(f"\n{engine:<18} [{source}]")
    print(f"  result arrays : {keys}")
    fields = record.get("item_fields")
    if isinstance(fields, dict):
        print(f"  item fields   : {', '.join(sorted(fields))}")


if __name__ == "__main__":
    raise SystemExit(main())
